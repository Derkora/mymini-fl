from fastapi import APIRouter
from app.server_manager_instance import fl_manager
from app.controllers.client_controller import registered_clients

router = APIRouter()

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
