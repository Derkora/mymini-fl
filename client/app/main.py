import os
import threading
from fastapi import FastAPI, UploadFile, File, Form, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import List
import torch
import numpy as np

from app.db.db import get_db, engine, Base
from app.db.models import UserLocal, Embedding, AttendanceLocal
from app.utils.face_pipeline import face_pipeline
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import load_head
from app.utils.trainer import LocalTrainer

from app.client import start_flower_client, get_global_label, sync_users_from_server
MAX_USERS_CAPACITY = 100

# Init DB Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="FL Edge Client - Local Mode")

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=start_flower_client, daemon=True)
    thread.start()
    sync_thread = threading.Thread(target=sync_users_from_server, daemon=True)
    sync_thread.start()
    
# Mount Static & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

encryptor = EmbeddingEncryptor()
def is_model_ready():
    return os.path.exists("local_head.pth")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user_count = db.query(UserLocal).count()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_count": user_count,
        "model_exists": is_model_ready()
    })

@app.get("/register-page", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/attendance-page", response_class=HTMLResponse)
async def attendance_page(request: Request):
    return templates.TemplateResponse("attendance.html", {"request": request})


# client/app/main.py

@app.post("/register")
async def register_user(
    request: Request,
    name: str = Form(...),
    nim: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    valid_embeddings = []
    errors = []

    for file in files:
        content = await file.read()
        
        embeddings_list, msg = face_pipeline.process_with_augmentation(content)
        
        if embeddings_list:
            valid_embeddings.extend(embeddings_list)
        else:
            print(f"[REGISTER ERROR] File {file.filename}: {msg}") 
            errors.append(f"{file.filename}: {msg}")

    if len(valid_embeddings) == 0:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": "Gagal Registrasi: Wajah tidak terdeteksi di semua foto.",
            "details": {"Error Log": errors}, 
            "next_url": "/register-page",
            "next_label": "Coba Lagi"
        })
    
    lbl = get_global_label(nim, name)
    
    if lbl is None:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": "Gagal menghubungi Server Pusat untuk sinkronisasi Label.",
            "next_url": "/register-page",
            "next_label": "Coba Lagi Nanti"
        })

    existing_user = db.query(UserLocal).filter(UserLocal.nim == nim).first()
    
    if existing_user:
        new_user = existing_user
        print(f"[REGISTER] User {name} sudah ada, menambahkan data wajah baru.")
    else:
        new_user = UserLocal(name=name, nim=nim)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

    for vec in valid_embeddings:
        enc_data, iv, salt = encryptor.encrypt_embedding(vec)
        db_emb = Embedding(
            user_id=new_user.user_id,
            encrypted_embedding=enc_data,
            iv=iv,
            salt=salt
        )
        db.add(db_emb)
            
    db.commit()
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "status": "registered",
        "message": f"Registrasi Berhasil untuk {name}",
        "details": {
            "Global Label ID": lbl,
            "Total Sampel Wajah": len(valid_embeddings), 
            "NIM": nim
        },
        "next_url": "/register-page",
        "next_label": "Daftar Lagi"
    })

@app.post("/train")
async def trigger_training(request: Request, db: Session = Depends(get_db)):
    trainer = LocalTrainer(db)
    result = trainer.train_local()
    
    status = "success" if result.get("status") == "success" else "error"
    msg = "Model Lokal Berhasil Dilatih!" if status == "success" else f"Gagal: {result}"
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "status": status,
        "message": msg,
        "details": result,
        "next_url": "/",
        "next_label": "Kembali ke Dashboard"
    })

@app.post("/attendance")
async def attendance(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_model_ready():
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": "Model belum dilatih! Lakukan training di Dashboard dulu.",
            "next_url": "/",
            "next_label": "Ke Dashboard"
        })

    model = load_head("local_head.pth", num_classes=MAX_USERS_CAPACITY) 
    model.eval()
    
    # Proses Image
    content = await file.read()
    emb_numpy, msg = face_pipeline.process_image(content)
    
    if emb_numpy is None:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": f"Wajah tidak terdeteksi: {msg}",
            "next_url": "/attendance-page",
            "next_label": "Coba Lagi"
        })

    # Inferensi
    emb_tensor = torch.tensor(emb_numpy, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(emb_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
    class_idx = predicted_class.item()
    softmax_score = confidence.item()
    
    # Ambil semua user lokal
    users = db.query(UserLocal).all()
    matched_user = None
    
    for user in users:
        lbl = get_global_label(user.nim) 
        if lbl == class_idx:
            matched_user = user
            break
    
    if not matched_user:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "unknown",
            "message": "Wajah terdeteksi, tapi data user belum tersinkronisasi.",
            "next_url": "/attendance-page",
            "next_label": "Coba Lagi"
        })
    
    stored_embeddings = db.query(Embedding).filter(Embedding.user_id == matched_user.user_id).all()
    
    max_similarity = -1.0
    
    if not stored_embeddings:
        print("[ATTENDANCE] User ditemukan tapi tidak punya sampel wajah untuk verifikasi.")
        final_score = softmax_score
    else:
        for db_emb in stored_embeddings:
            try:
                # Decrypt vektor dari DB
                vec_stored = encryptor.decrypt_embedding(db_emb.encrypted_embedding, db_emb.iv)
                
                # Hitung Cosine Similarity
                # Rumus: A . B
                sim = np.dot(emb_numpy, vec_stored)
                
                if sim > max_similarity:
                    max_similarity = sim
            except Exception as e:
                print(f"[VERIFY ERROR] Gagal decrypt sample: {e}")
                continue
        
        final_score = float(max_similarity)
    
    THRESHOLD = 0.50 
    
    print(f"[ATTENDANCE] Kandidat: {matched_user.name} | Softmax: {softmax_score:.2f} | Cosine Sim: {final_score:.2f}")

    if final_score > THRESHOLD:
        # REKAM LOG
        log = AttendanceLocal(
            user_id=matched_user.user_id,
            confidence=final_score,
            sent_to_server=False
        )
        db.add(log)
        db.commit()
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "success",
            "message": f"Halo, {matched_user.name}!",
            "details": {
                "Metode Verifikasi": "Biometric Match (Cosine)",
                "Skor Kemiripan": f"{final_score:.2f} / 1.0",
                "Waktu": "Baru Saja"
            },
            "next_url": "/attendance-page",
            "next_label": "Absen Lagi"
        })
    else:
        msg = "Wajah tidak terverifikasi."
        if final_score > 0.3:
            msg = "Wajah agak mirip, tapi kurang meyakinkan. Coba lepas kacamata/masker."
            
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "unknown",
            "message": f"Maaf, {msg}",
            "details": {
                "Kandidat Terdekat": matched_user.name,
                "Skor Kemiripan": f"{final_score:.2f} (Butuh > {THRESHOLD})",
            },
            "next_url": "/attendance-page",
            "next_label": "Coba Lagi"
        })