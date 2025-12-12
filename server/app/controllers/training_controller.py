from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional # Import Optional
from app.server_manager_instance import fl_manager
from app.controllers.client_controller import registered_clients

router = APIRouter()

global_user_db = {}

class LabelRequest(BaseModel):
    nim: str
    name: str = ""
    registered_edge_id: Optional[str] = None 

@router.post("/get_label")
def get_or_create_label(req: LabelRequest):
    global global_user_db
    
    if req.nim not in global_user_db:
        new_label = len(global_user_db)
        if new_label >= 100: 
            raise HTTPException(status_code=400, detail="Full")
            
        global_user_db[req.nim] = {
            "label": new_label,
            "name": req.name,
            "nim": req.nim,
            # Simpan Edge ID saat registrasi pertama
            "registered_edge_id": req.registered_edge_id 
        }
        print(f"[REGISTRY] New: {req.name} ({req.nim}) -> Label {new_label} on {req.registered_edge_id}")
        
    else:
        print(f"[REGISTRY] Existing: {req.name} ({req.nim}) -> Label {global_user_db[req.nim]['label']} on {req.registered_edge_id}")
        existing_data = global_user_db[req.nim]
        if existing_data.get("registered_edge_id") and existing_data.get("registered_edge_id") != req.registered_edge_id and req.registered_edge_id:
             raise HTTPException(status_code=403, detail=f"User already registered on {existing_data.get('registered_edge_id')}")

    return global_user_db[req.nim]

@router.get("/global_users")
def get_all_users():
    return list(global_user_db.values())

@router.post("/start")
async def start_training(rounds: int = 10):
    fl_manager.start_training(rounds)
    return {"status": "started", "rounds": rounds}

@router.get("/status")
async def get_status():
    # Ambil status FL 
    fl_status = fl_manager.status()

    clients_list = list(registered_clients.values())
    
    return {
        **fl_status, 
        "clients": clients_list
    }
