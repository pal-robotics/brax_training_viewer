
import asyncio
import threading
import time
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from etils import epath
from braxviewer.utils.html_utils import render_from_json
import jinja2
import json as std_json
import logging

class ConnectionManager:
    """Manages active WebSocket connections for control signals."""
    def __init__(self, sender):
        self.active_connections: list[WebSocket] = []
        self.sender = sender

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send initial state upon connection
        initial_state_msg = {
            "type": "set_render_state",
            "enabled": self.sender.rendering_enabled
        }
        await websocket.send_text(std_json.dumps(initial_state_msg))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_text(std_json.dumps(message))
            except Exception:
                disconnected_clients.append(connection)

        for client in disconnected_clients:
            self.disconnect(client)

class Server:
    def __init__(self, sender):
        self.sender = sender
        self.config = sender.config
        self.app = FastAPI()
        self.clients_frame = set()
        self.control_manager = ConnectionManager(sender)
        self._setup_cors()
        self._setup_routes()
        
        # Disable logging for frequent status endpoint
        self._disable_status_logging()

    def _disable_status_logging(self):
        """Disable logging for the frequent status endpoint."""
        # Create a custom logging filter to exclude status endpoint logs
        class StatusLogFilter(logging.Filter):
            def filter(self, record):
                # Filter out logs for /api/rendering_status endpoint
                if hasattr(record, 'getMessage'):
                    msg = record.getMessage()
                    if '/api/rendering_status' in msg:
                        return False
                return True
        
        # Apply the filter to uvicorn access logs
        uvicorn_logger = logging.getLogger("uvicorn.access")
        uvicorn_logger.addFilter(StatusLogFilter())

    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
        )

    def _setup_routes(self):
        static_path = epath.resource_path('braxviewer') / 'static'
        self.app.mount("/static", StaticFiles(directory=static_path), name="static")

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            if self.sender.system_json:
                return render_from_json(sys=self.sender.system_json, height=480, colab=True, base_url=None)
            return self._get_waiting_html()

        @self.app.get("/api/rendering_status")
        async def get_rendering_status():
            return {"rendering_enabled": self.sender.rendering_enabled}

        @self.app.post("/api/rendering_status")
        async def set_rendering_status(request: Request):
            """Set rendering status and broadcast to all clients."""
            try:
                data = await request.json()
                is_enabled = bool(data.get("rendering_enabled", True))
                
                # Update the sender's rendering status
                self.sender.rendering_enabled = is_enabled
                self.sender.log(f"Rendering status set to: {is_enabled}")
                
                # Broadcast the change to all control clients
                await self.control_manager.broadcast({
                    "type": "set_render_state",
                    "enabled": is_enabled
                })
                
                return {"status": "success", "rendering_enabled": is_enabled}
            except Exception as e:
                self.sender.log(f"Error setting rendering status: {e}", level="error")
                return {"status": "error", "message": str(e)}

        @self.app.websocket("/ws/frame")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            # If this is the streamer, handle it separately.
            if websocket.query_params.get('is_streamer'):
                self.sender.log("Streamer connected to frame endpoint.")
                try:
                    while True:
                        data = await websocket.receive_text()
                        await self.broadcast_to_frames(data)
                except WebSocketDisconnect:
                    self.sender.log("Streamer disconnected.", level='warning')
            else:
                self.clients_frame.add(websocket)
                self.sender.log(f"New frame client connected. Total: {len(self.clients_frame)}")
                try:
                    # Keep connection open
                    while True:
                        await websocket.receive_text()
                except WebSocketDisconnect:
                    self.clients_frame.remove(websocket)
                    self.sender.log(f"Frame client disconnected. Total: {len(self.clients_frame)}")

        @self.app.websocket("/ws/control")
        async def websocket_control(websocket: WebSocket):
            await self.control_manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        msg = std_json.loads(data)
                        if msg.get('type') == 'toggle_render':
                            is_enabled = bool(msg.get('enabled', False))
                            self.sender.rendering_enabled = is_enabled
                            self.sender.log(f"Real-time rendering toggled to: {self.sender.rendering_enabled}")
                            
                            # Broadcast the change to all clients
                            await self.control_manager.broadcast({
                                "type": "set_render_state",
                                "enabled": is_enabled
                            })
                    except Exception:
                        self.sender.log(f"Invalid control message: {data}", level='warning')
            except WebSocketDisconnect:
                self.control_manager.disconnect(websocket)

    async def broadcast_to_frames(self, data: str):
        if not self.clients_frame:
            return
        
        tasks = [client.send_text(data) for client in self.clients_frame]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle clients that may have disconnected
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client = list(self.clients_frame)[i]
                self.clients_frame.remove(client)
                self.sender.log(f"Removed disconnected frame client. Total: {len(self.clients_frame)}")

    def _get_waiting_html(self):
        # This HTML is largely unchanged
        html_path = epath.resource_path('braxviewer') / 'static' / 'wait.html'
        if not html_path.exists():
            # Fallback in case the file is not there, using the original string
            return HTMLResponse(content="""... waiting page HTML ...""")
        return HTMLResponse(content=html_path.read_text())
    
    async def _broadcast_control(self, message: dict):
        if not self.clients_control: return
        tasks = [client.send_text(std_json.dumps(message)) for client in self.clients_control]
        await asyncio.gather(*tasks, return_exceptions=True)

    def send_control(self, message: dict):
        if hasattr(self, '_loop') and self._loop.is_running():
             asyncio.run_coroutine_threadsafe(self._broadcast_control(message), self._loop)

    def run(self, use_thread: bool = True, wait_for_startup: int = 2):
        def start_server():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Configure uvicorn logger
            level_mapping = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL}
            log_level = level_mapping.get(self.config.server_log_level.lower(), logging.WARNING)
            
            # Silence uvicorn's default loggers
            logging.getLogger("uvicorn").setLevel(log_level)
            logging.getLogger("uvicorn.access").setLevel(log_level)
            
            config = uvicorn.Config(self.app, host=self.config.host, port=self.config.port, log_level=None)
            server = uvicorn.Server(config)
            self._loop.run_until_complete(server.serve())

        if use_thread:
            self._server_thread = threading.Thread(target=start_server, daemon=True)
            self._server_thread.start()
            self.sender.log(f"Brax Viewer starting on http://{self.config.host}:{self.config.port}")
            self.sender.log(f"Server log level: {self.config.server_log_level}")
            time.sleep(wait_for_startup)
            self.sender.start() # Start the viewer's own processes (like the streamer)
            
            # After init, we should have system_json
            if self.sender.system_json:
                self.send_control({"type": "refresh_web"})
        else:
            self.sender.start()
            uvicorn.run(self.app, host=self.config.host, port=self.config.port, log_level=self.config.server_log_level)

    def stop(self):
        self.sender.stop() 