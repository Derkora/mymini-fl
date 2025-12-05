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
SERVER_API_URL = "http://server:8080/api/clients" 

def report_status(status_msg):
    """Kirim status ke Server Dashboard via HTTP API"""
    try:
        payload = {
            "id": CLIENT_ID,
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "fl_status": status_msg,
            "last_seen": datetime.now().strftime("%H:%M:%S")
        }

        requests.post(f"{SERVER_API_URL}/register", json=payload, timeout=2)
    except Exception as e:
        print(f"[CLIENT REPORT] Gagal lapor status: {e}")
        
def train_model(net, train_loader, epochs=1):
    """Training loop lokal standard PyTorch"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    
    for _ in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

def load_data_from_db():
    """
    Ambil semua embedding dari DB lokal, decrypt, dan format jadi Tensor.
    Return: DataLoader atau (X_train, y_train)
    """
    db = SessionLocal()
    encryptor = EmbeddingEncryptor()
    
    try:
        # Ambil user mapping (user_id -> class index 0..9)
        
        users = db.query(UserLocal).all()
        user_to_label = {u.user_id: (u.user_id - 1) % 10 for u in users}

        embeddings = db.query(Embedding).all()
        
        X_list = []
        y_list = []
        
        for emb in embeddings:
            if emb.user_id not in user_to_label:
                continue
                
            # Decrypt
            vec = encryptor.decrypt_embedding(emb.encrypted_embedding, emb.iv)
            label = user_to_label[emb.user_id]
            
            X_list.append(vec)
            y_list.append(label)
            
        if len(X_list) == 0:
            return None
            
        X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_list), dtype=torch.long)
        
        # Buat DataLoader sederhana
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
    except Exception as e:
        print(f"[CLIENT DATA] Error loading data: {e}")
        return None
    finally:
        db.close()

class RealClient(fl.client.NumPyClient):
    def __init__(self):
        self.net = SimpleClassifier(num_classes=10)

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

        train_model(self.net, trainloader, epochs=5)

        torch.save(self.net.state_dict(), "local_head.pth")
        
        report_status(f"Selesai Training (Ronde {rnd})")
        print(f"[CLIENT] Round {rnd} selesai.")
        
        return self.get_parameters(config={}), len(trainloader.dataset), {}

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