from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Set
import os
import json

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Global storage
initial_system_json = None
latest_trajectory = None
websocket_clients: Set[WebSocket] = set()

# Store the latest data (flexible, for multiple fields)
latest_data_cache = {}

# Data models
class SystemData(BaseModel):
    system_json: dict  # Change from system_json_b64: str to system_json: dict

class RolloutData(BaseModel):
    trajectory: str  # JSON string

class ListenData(BaseModel):
    message: str

@app.post("/system")
def receive_system(data: SystemData):
    global initial_system_json
    initial_system_json = data.system_json
    return {"status": "ok"}

@app.post("/rollout")
async def receive_rollout(data: SystemData):
    # Broadcast the new system JSON to all WebSocket clients
    to_remove = []
    for ws in websocket_clients:
        try:
            print("Sending new system to client.")
            await ws.send_text(json.dumps(data.system_json))
        except Exception as e:
            print(f"Error sending to client: {e}")
            to_remove.append(ws)
    for ws in to_remove:
        websocket_clients.remove(ws)
    return {"status": "ok"}

@app.post("/listen")
async def listen_message(request: Request):
    global latest_data_cache
    data = await request.json()
    if not isinstance(data, dict):
        return {"status": "error", "detail": "Payload must be a JSON object"}
    # Merge new data into cache
    latest_data_cache.update(data)
    to_remove = []
    for ws in websocket_clients:
        try:
            print(f"Sending message to client: {data}")
            # Send as JSON string
            await ws.send_text(json.dumps(data))
        except Exception as e:
            print(f"Error sending to client: {e}")
            to_remove.append(ws)
    for ws in to_remove:
        websocket_clients.remove(ws)
    return {"status": "ok"}

@app.get("/state")
def get_state():
    return latest_data_cache

@app.websocket("/ws/rollout")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    print("Client connected. Total clients:", len(websocket_clients))
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        websocket_clients.discard(websocket)
        print("Remaining clients:", len(websocket_clients))

@app.get("/", response_class=HTMLResponse)
def show_rollout():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
