import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sqlalchemy.orm import Session
from app.db.models import UserLocal, Embedding
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import SimpleClassifier, save_head

class LocalTrainer:
    def __init__(self, db: Session):
        self.db = db
        self.encryptor = EmbeddingEncryptor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_local(self, epochs=20, lr=0.01):
        # 1. Ambil semua user dan embedding mereka dari DB
        users = self.db.query(UserLocal).all()
        if not users:
            return {"status": "error", "reason": "No users found in DB"}

        # PERBAIKAN DI SINI: Ganti u.id menjadi u.user_id
        user_map = {u.user_id: idx for idx, u in enumerate(users)}
        
        X_train = [] # Data Embedding
        y_train = [] # Label (0, 1, 2...)

        # 2. Decrypt & Prepare Dataset
        embeddings = self.db.query(Embedding).all()
        
        count_used = 0
        for emb in embeddings:
            # Pastikan user_id ada di map (cegah error jika ada data yatim piatu)
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

        # 3. Setup Model
        # Kita set output class sesuai jumlah user yang ada saat ini (misal dynamic)
        # Atau fix 10 sesuai prototype
        num_users = len(users)
        # Agar aman, kita pakai max(10, num_users) atau fix 10 jika mau konsisten dengan classifier.py
        model = SimpleClassifier(num_classes=10).to(self.device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # 4. Training Loop
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

        # 5. Save Model Lokal
        save_head(model, path="local_head.pth")
        
        return {
            "status": "success", 
            "users_count": num_users, 
            "samples": count_used,
            "final_loss": final_loss
        }