import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import requests
import os
import socket
from datetime import datetime
from collections import OrderedDict

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
        
        global_users = response.json() # List of {label, name, nim}
        
        db = SessionLocal()
        for u_data in global_users:
            nim = u_data['nim']
            name = u_data['name']
            
            # Cek apakah user sudah ada di lokal
            exists = db.query(UserLocal).filter(UserLocal.nim == nim).first()
            
            if not exists:
                print(f"[SYNC] Menambahkan user baru dari server: {name}")
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
        
def train_model(net, train_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    
    final_loss = 0.0
    final_acc = 0.0
    
    for _ in range(epochs):
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
            
        if len(train_loader) > 0:
            final_loss = running_loss / len(train_loader)
        if total > 0:
            final_acc = correct / total
            
    return final_loss, final_acc

def load_data_from_db():
    db = SessionLocal()
    encryptor = EmbeddingEncryptor()
    
    try:  
        users = db.query(UserLocal).all()
        user_to_label = {}

        print(f"[CLIENT DATA] Sinkronisasi {len(users)} user lokal ke Server...")
        for u in users:
            global_label = get_global_label(u.nim)
            if global_label is not None:
                user_to_label[u.user_id] = global_label
            else:
                print(f"[CLIENT DATA] Skip user {u.name} (Gagal Sync Label)")

        embeddings = db.query(Embedding).all()
        
        X_list = []
        y_list = []
        
        for emb in embeddings:
            if emb.user_id not in user_to_label:
                continue
                
            vec = encryptor.decrypt_embedding(emb.encrypted_embedding, emb.iv)
            label = user_to_label[emb.user_id] # Menggunakan Global Label
            
            X_list.append(vec)
            y_list.append(label)
            
        if len(X_list) == 0:
            return None
            
        X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_list), dtype=torch.long)
        
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), 
            batch_size=8, 
            shuffle=True
        )
        
    except Exception as e:
        print(f"[CLIENT DATA] Error loading data: {e}")
        return None
    finally:
        db.close()

class RealClient(fl.client.NumPyClient):
    def __init__(self):
        self.net = SimpleClassifier(num_classes=100) 

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        rnd = config.get('round', '?')
        print(f"[CLIENT] Round {rnd} dimulai.")
        
        report_status(f"Sedang Training (Ronde {rnd})")

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

        trainloader = load_data_from_db()
        if trainloader is None:
            print("[CLIENT] Data kosong! Skip.")
            report_status(f"Idle (Data Kosong - Ronde {rnd})")
            return parameters, 0, {}

        loss, accuracy = train_model(self.net, trainloader, epochs=50) 

        torch.save(self.net.state_dict(), "local_head.pth")
        
        status_msg = f"Selesai (Loss: {loss:.4f}, Acc: {accuracy:.1%})"
        report_status(status_msg, metrics={"loss": loss, "accuracy": accuracy})
        
        print(f"[CLIENT] Round {rnd} selesai. {status_msg}")
        
        return self.get_parameters(config={}), len(trainloader.dataset), {"loss": loss, "accuracy": accuracy}
    

    def evaluate(self, parameters, config):
        return 0.0, 1, {"accuracy": 0.0}

def start_flower_client():
    report_status("Online (Menunggu Server)")
    
    while True:
        try:
            print("[CLIENT] Connecting to FL Server...")
            fl.client.start_numpy_client(
                server_address="server:8085",
                client=RealClient()
            )
            break
        except Exception as e:
            print(f"[CLIENT] Connection failed. Retrying...")
            time.sleep(5)