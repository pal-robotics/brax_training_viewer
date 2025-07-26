
import asyncio
import threading
import time
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from etils import epath
from braxviewer.html_utils import render_from_json
import jinja2
import json as std_json
import logging

class Server:
    def __init__(self, viewer):
        self.viewer = viewer
        self.config = viewer.config
        self.app = FastAPI()
        self.clients_frame = set()
        self.clients_control = set()
        self._setup_cors()
        self._setup_routes()

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
            if self.viewer.system_json:
                return render_from_json(sys=self.viewer.system_json, height=480, colab=True, base_url=None)
            else:
                return self._get_waiting_html()

        @self.app.websocket("/ws/frame")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            # If this is the streamer, handle it separately.
            if websocket.query_params.get('is_streamer'):
                self.viewer.log("Streamer connected to frame endpoint.")
                try:
                    while True:
                        data = await websocket.receive_text()
                        await self.broadcast_to_frames(data)
                except WebSocketDisconnect:
                    self.viewer.log("Streamer disconnected.", level='warning')
            else:
                self.clients_frame.add(websocket)
                self.viewer.log(f"New frame client connected. Total: {len(self.clients_frame)}")
                try:
                    # Keep connection open
                    while True:
                        await websocket.receive_text()
                except WebSocketDisconnect:
                    self.clients_frame.remove(websocket)
                    self.viewer.log(f"Frame client disconnected. Total: {len(self.clients_frame)}")

        @self.app.websocket("/ws/control")
        async def websocket_control(websocket: WebSocket):
            await websocket.accept()
            self.clients_control.add(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        msg = std_json.loads(data)
                        if msg.get('type') == 'toggle_render':
                            self.viewer.rendering_enabled = bool(msg.get('enabled', False))
                            self.viewer.log(f"Real-time rendering toggled to: {self.viewer.rendering_enabled}")
                    except Exception:
                        self.viewer.log(f"Invalid control message: {data}", level='warning')
            except WebSocketDisconnect:
                self.clients_control.remove(websocket)

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
                self.viewer.log(f"Removed disconnected frame client. Total: {len(self.clients_frame)}")

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
            self.viewer.log(f"Brax Viewer starting on http://{self.config.host}:{self.config.port}")
            self.viewer.log(f"Server log level: {self.config.server_log_level}")
            time.sleep(wait_for_startup)
            self.viewer.start() # Start the viewer's own processes (like the streamer)
            
            # After init, we should have system_json
            if self.viewer.system_json:
                self.send_control({"type": "refresh_web"})
        else:
            self.viewer.start()
            uvicorn.run(self.app, host=self.config.host, port=self.config.port, log_level=self.config.server_log_level)

    def stop(self):
        self.viewer.stop() 