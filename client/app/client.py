import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import requests
import os
import socket
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report

# Import DB & Models Local
from app.db.db import SessionLocal
from app.db.models import UserLocal, Embedding
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import SimpleClassifier

CLIENT_ID = os.getenv("HOSTNAME", "client-unknown")
SERVER_URL = "http://server:8080"
MAX_USERS_CAPACITY = 100

def get_global_label(nim: str, name: str = ""):
    try:
        payload = {"nim": nim, "name": name}
        response = requests.post(f"{SERVER_URL}/api/training/get_label", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()["label"]
    except Exception as e:
        print(f"[SYNC ERROR] {e}")
    return None

def sync_users_from_server():
    print("[SYNC] Memulai sinkronisasi user dari Server...")
    try:
        response = requests.get(f"{SERVER_URL}/api/training/global_users", timeout=5)
        if response.status_code != 200:
            return
        
        global_users = response.json() 
        db = SessionLocal()
        for u_data in global_users:
            nim = u_data['nim']
            name = u_data['name']
            exists = db.query(UserLocal).filter(UserLocal.nim == nim).first()
            if not exists:
                print(f"[SYNC] Menambahkan user baru: {name}")
                new_user = UserLocal(name=name, nim=nim)
                db.add(new_user)
        db.commit()
        db.close()
        print("[SYNC] Sinkronisasi Selesai.")
    except Exception as e:
        print(f"[SYNC FAILED] {e}")
        
def report_status(status_msg, metrics=None):
    try:
        payload = {
            "id": CLIENT_ID,
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "fl_status": status_msg,
            "last_seen": datetime.now().strftime("%H:%M:%S")
        }
        if metrics:
            payload["metrics"] = metrics
        requests.post(f"{SERVER_URL}/api/clients/register", json=payload, timeout=2)
    except Exception as e:
        print(f"[CLIENT REPORT] Gagal lapor status: {e}")

def save_confusion_matrix(y_true, y_pred):
    """Menyimpan Confusion Matrix sebagai gambar untuk debugging"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {CLIENT_ID}')
        plt.ylabel('Label Asli')
        plt.xlabel('Prediksi')
        
        # Simpan di folder static agar bisa diakses web (opsional)
        output_path = "app/static/confusion_matrix.png"
        os.makedirs("app/static", exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"[METRICS] Confusion Matrix disimpan di {output_path}")
    except Exception as e:
        print(f"[METRICS] Gagal simpan confusion matrix: {e}")

def validate_model(net, test_loader):
    """Validasi model + Generate Confusion Matrix"""
    criterion = nn.CrossEntropyLoss()
    net.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            output = net(images)
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    if len(test_loader) > 0:
        val_loss /= len(test_loader)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Simpan CM jika ini adalah tahap evaluasi
    save_confusion_matrix(all_labels, all_preds)
    
    return val_loss, accuracy

def train_model(net, train_loader, val_loader=None, epochs=10, patience=3):
    """Training dengan Early Stopping & Scheduler"""
    criterion = nn.CrossEntropyLoss()
    # L2 Regularization (weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    net.train()
    
    best_loss = float('inf')
    patience_counter = 0
    final_loss = 0.0
    final_acc = 0.0
    
    # Simpan state awal
    best_state = net.state_dict()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        
        # --- EARLY STOPPING LOGIC ---
        # Jika ada data validasi, gunakan loss validasi. Jika tidak, gunakan loss training.
        current_val_loss = epoch_loss
        if val_loader:
            v_loss, _ = validate_model(net, val_loader)
            current_val_loss = v_loss
            net.train() # Kembalikan ke mode train

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_state = net.state_dict()
            patience_counter = 0
            final_loss = epoch_loss
            final_acc = epoch_acc
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Berhenti di epoch {epoch+1}")
            break
    
    # Kembalikan bobot terbaik yang ditemukan
    net.load_state_dict(best_state)
    return final_loss, final_acc

def load_data_from_db():
    db = SessionLocal()
    encryptor = EmbeddingEncryptor()
    
    try:  
        users = db.query(UserLocal).all()
        user_to_label = {}

        print(f"[CLIENT DATA] Sinkronisasi {len(users)} user lokal...")
        for u in users:
            global_label = get_global_label(u.nim)
            if global_label is not None:
                user_to_label[u.user_id] = global_label

        embeddings = db.query(Embedding).all()
        X_list, y_list = [], []
        
        for emb in embeddings:
            if emb.user_id not in user_to_label: continue
            vec = encryptor.decrypt_embedding(emb.encrypted_embedding, emb.iv)
            label = user_to_label[emb.user_id]
            X_list.append(vec)
            y_list.append(label)
            
        if len(X_list) == 0:
            return None, None
            
        X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.long)
        full_dataset = TensorDataset(X_tensor, y_tensor)
        
        # --- SPLIT DATA (80% Train, 20% Test) ---
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        if test_size == 0 and len(full_dataset) > 1:
            test_size = 1; train_size = len(full_dataset) - 1
            
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"[CLIENT DATA] Error loading data: {e}")
        return None, None
    finally:
        db.close()

class RealClient(fl.client.NumPyClient):
    def __init__(self):
        self.net = SimpleClassifier(num_classes=MAX_USERS_CAPACITY) 

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        rnd = config.get('round', '?')
        print(f"[CLIENT] Round {rnd} dimulai.")
        report_status(f"Sedang Training (Ronde {rnd})")

        # Load Global Model
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

        # Load & Split Data
        trainloader, valloader = load_data_from_db()
        if trainloader is None:
            return parameters, 0, {}

        # Training dengan Early Stopping
        # Kita set max epochs 10, tapi bisa berhenti lebih cepat jika loss validasi naik
        loss, accuracy = train_model(self.net, trainloader, valloader, epochs=10, patience=3)

        # Simpan Model Lokal
        torch.save(self.net.state_dict(), "local_head.pth")
        
        # --- DIFFERENTIAL PRIVACY (Noise Addition) ---
        print("[CLIENT] Menambahkan DP Noise...")
        params = self.get_parameters(config={})
        noise_multiplier = 0.005 # Tingkat noise (sesuaikan kebutuhan)
        noisy_params = []
        for p in params:
            noise = np.random.normal(0, noise_multiplier, p.shape)
            noisy_params.append(p + noise)
        
        status_msg = f"Selesai (L:{loss:.3f}, A:{accuracy:.1%})"
        report_status(status_msg, metrics={"loss": loss, "accuracy": accuracy})
        
        return noisy_params, len(trainloader.dataset), {"loss": loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

        _, testloader = load_data_from_db()
        
        if testloader is None or len(testloader.dataset) == 0:
            return 0.0, 0, {"accuracy": 0.0}

        loss, accuracy = validate_model(self.net, testloader)
        
        print(f"[CLIENT] Evaluasi (Test Set). Loss: {loss:.4f}, Acc: {accuracy:.1%}")
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

def start_flower_client():
    report_status("Online (Menunggu Server)")
    while True:
        try:
            print("[CLIENT] Connecting to FL Server...")
            fl.client.start_numpy_client(server_address="server:8085", client=RealClient())
            break
        except Exception as e:
            print(f"[CLIENT] Connection failed. Retrying...")
            time.sleep(5)