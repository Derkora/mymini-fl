from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class ClientStatus(BaseModel):
    id: str
    ip_address: str
    fl_status: str       
    last_seen: str       


# Simulasi database sementara
registered_clients = {}  

@router.post("/register")
def register_client(client: ClientStatus):
    registered_clients[client.id] = client.dict()
    return {"message": "Client registered", "client": client}

@router.get("/")
def get_all_clients():
    return {"clients": list(registered_clients.values())}

@router.get("/{client_id}")
def get_client(client_id: str):
    if client_id not in registered_clients:
        return {"error": "Client not found"}
    return registered_clients[client_id]

@router.post("/{client_id}/update")
def update_client_status(client_id: str, status: dict):
    if client_id not in registered_clients:
        return {"error": "Client not registered"}
    
    registered_clients[client_id].update(status)
    return {"message": "Status updated", "client": registered_clients[client_id]}
