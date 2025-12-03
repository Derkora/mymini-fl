from fastapi import FastAPI
import threading
from app.client import start_flower_client

app = FastAPI(title="FL Edge Client")

@app.get("/")
def root():
    return {"message": "Edge client running"}

# Jalankan Flower Client pada thread terpisah saat startup
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=start_flower_client, daemon=True)
    thread.start()
    print("[CLIENT] Flower client started")
