import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sqlalchemy.orm import Session
from app.db.models import UserLocal, Embedding
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import SimpleClassifier, save_head
import requests

SERVER_URL = "http://server:8080"

class LocalTrainer:
    def __init__(self, db: Session):
        self.db = db
        self.encryptor = EmbeddingEncryptor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_global_label(self, nim): 
        try:
            resp = requests.post(f"{SERVER_URL}/api/training/get_label", json={"nim": nim}, timeout=2)
            if resp.status_code == 200:
                return resp.json()["label"]
        except:
            return None
        return None
    
    def train_local(self, epochs=20, lr=0.01):
        users = self.db.query(UserLocal).all()
        if not users:
            return {"status": "error", "reason": "No users found"}

        user_map = {}
        for u in users:
            lbl = self._fetch_global_label(u.nim)
            if lbl is not None:
                user_map[u.user_id] = lbl
            else:
                print(f"[TRAIN LOCAL] Gagal sync label untuk {u.name}, skip.")
        
        X_train = [] # Data Embedding
        y_train = [] # Label (0, 1, 2...)

        # Decrypt & Prepare Dataset
        embeddings = self.db.query(Embedding).all()
        
        count_used = 0
        for emb in embeddings:
            # Pastikan user_id ada di map
            if emb.user_id not in user_map: 
                continue 
            
            try:
                # Decrypt dari DB
                vec_numpy = self.encryptor.decrypt_embedding(emb.encrypted_embedding, emb.iv)
                
                X_train.append(vec_numpy)
                y_train.append(user_map[emb.user_id])
                count_used += 1
            except Exception as e:
                print(f"[TRAIN] Gagal decrypt embedding {emb.embedding_id}: {e}")
                continue

        if not X_train:
            return {"status": "error", "reason": "No valid embeddings found"}

        # Convert ke Tensor
        X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # Setup Model
        num_users = len(users)
        model = SimpleClassifier(num_classes=100).to(self.device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Training Loop
        print(f"[TRAIN] Start training on {count_used} images for {num_users} users...")
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss {final_loss:.4f}")

        # Save Model Lokal
        save_head(model, path="local_head.pth")
        
        return {
            "status": "success", 
            "users_count": num_users, 
            "samples": count_used,
            "final_loss": final_loss
        }