from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from typing import Set
import uvicorn

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

# WebSocket clients set
websocket_clients: Set[WebSocket] = set()

@app.websocket("/ws/frame")
async def websocket_frame(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    print("Client connected to /ws/frame. Total clients:", len(websocket_clients), "from", websocket.client.host)
    try:
        while True:
            frame = await websocket.receive_text()
            print(f"Received frame from backend: {frame}")  # Debug print
            to_remove = []
            for ws in websocket_clients:
                try:
                    await ws.send_text(frame)
                except Exception as e:
                    print(f"Error sending frame to client: {e}")
                    to_remove.append(ws)
            for ws in to_remove:
                websocket_clients.remove(ws)
    except WebSocketDisconnect:
        print("Client disconnected from /ws/frame.", "from", websocket.client.host)
    except Exception as e:
        print("WebSocket error (/ws/frame):", e)
    finally:
        websocket_clients.discard(websocket)
        print("Remaining clients:", len(websocket_clients))

@app.get("/", response_class=HTMLResponse)
def show_rollout():
    return FileResponse(os.path.join(frontend_dir, "index.html"))




def run_web_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """
    Launches the FastAPI web server for visualization.
    """
    uvicorn.run("backend:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    run_web_server() 