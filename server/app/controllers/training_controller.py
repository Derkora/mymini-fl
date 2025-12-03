from fastapi import APIRouter
from app.server_manager_instance import fl_manager

router = APIRouter()

@router.post("/start")
async def start_training(rounds: int = 10):
    fl_manager.start_training(rounds)
    return {"status": "started", "rounds": rounds}

@router.post("/stop")
async def stop_training():
    fl_manager.stop()
    return {"status": "stopped"}

@router.get("/status")
async def get_status():
    return fl_manager.status()
