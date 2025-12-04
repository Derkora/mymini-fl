import os
from fastapi import FastAPI, UploadFile, File, Form, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import List
import torch

from app.db.db import get_db, engine, Base
from app.db.models import UserLocal, Embedding, AttendanceLocal
from app.utils.face_pipeline import face_pipeline
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import load_head
from app.utils.trainer import LocalTrainer

# Init DB Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="FL Edge Client - Local Mode")

# 1. Mount Static & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

encryptor = EmbeddingEncryptor()

# --- Helper Check Model ---
def is_model_ready():
    return os.path.exists("local_head.pth")

# --- Routes Tampilan (GET) ---

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


# --- Routes Proses (POST) ---

@app.post("/register")
async def register_user(
    request: Request,
    name: str = Form(...),
    nim: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    # Buat User
    new_user = UserLocal(name=name, nim=nim)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    success_count = 0
    
    for file in files:
        content = await file.read()
        # ML Pipeline
        emb_numpy, msg = face_pipeline.process_image(content)
        
        if emb_numpy is not None:
            # Enkripsi & Simpan
            enc_data, iv, salt = encryptor.encrypt_embedding(emb_numpy)
            db_emb = Embedding(
                user_id=new_user.user_id,
                encrypted_embedding=enc_data,
                iv=iv,
                salt=salt
            )
            db.add(db_emb)
            success_count += 1
            
    db.commit()
    
    # Return ke Halaman Result
    return templates.TemplateResponse("result.html", {
        "request": request,
        "status": "registered",
        "message": f"Registrasi Berhasil untuk {name}",
        "details": {"Foto Valid": success_count, "NIM": nim},
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

    # Load Model
    model = load_head("local_head.pth", num_classes=10)
    model.eval()
    
    # Proses Image
    content = await file.read()
    emb_numpy, msg = face_pipeline.process_image(content)
    
    if emb_numpy is None:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": f"Wajah tidak terdeteksi: {msg}"
        })

    # Inferensi
    emb_tensor = torch.tensor(emb_numpy, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(emb_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
    class_idx = predicted_class.item()
    conf_score = confidence.item()

    # Cari Nama User
    users = db.query(UserLocal).all()
    user_map = {idx: u for idx, u in enumerate(users)}
    matched_user = user_map.get(class_idx)
    
    # Threshold Logic
    if matched_user and conf_score > 0.5: # Ambang batas
        # Simpan Log Absensi
        log = AttendanceLocal(
            user_id=matched_user.user_id,
            confidence=conf_score,
            sent_to_server=False
        )
        db.add(log)
        db.commit()
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "success",
            "message": f"Halo, {matched_user.name}!",
            "details": {
                "Confidence": f"{conf_score:.2%}",
                "Waktu": "Baru Saja"
            },
            "next_url": "/attendance-page",
            "next_label": "Absen Lagi"
        })
    else:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "unknown",
            "message": "Maaf, wajah Anda tidak dikenali.",
            "details": {"Confidence Tertinggi": f"{conf_score:.2%}"},
            "next_url": "/attendance-page",
            "next_label": "Coba Lagi"
        })