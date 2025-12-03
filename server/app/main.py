from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.controllers import training_controller, client_controller, attendance_controller

app = FastAPI(title="Federated Learning Server")

# Routers
app.include_router(training_controller.router, prefix="/api/training", tags=["training"])
app.include_router(client_controller.router, prefix="/api/clients", tags=["clients"])
app.include_router(attendance_controller.router, prefix="/api/attendance", tags=["attendance"])

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
