import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from etils import epath
from braxviewer.html_utils import render_from_json
import uvicorn
import time
import asyncio
from brax.envs.base import PipelineEnv
from typing import Optional
import logging
import jinja2
from brax.io import json
from braxviewer.WebSocketStreamer import WebSocketStreamer
import json as std_json

class WebViewer:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 log_level: str = "error",
                 env: Optional[PipelineEnv] = None,):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("uvicorn.info")
        self.log_level = log_level
        self.streamer = WebSocketStreamer(uri=f"ws://{self.host}:{self.port}/ws/frame")
        self.app = FastAPI()
        
        # Add a state to control rendering. Default to True.
        self.rendering_enabled = True
        self.discard_queue = True

        html_template_path = epath.resource_path('braxviewer') / 'static' / 'index.html'
        self.template = jinja2.Template(html_template_path.read_text())
        
        self.system_json = None
        self.clients_frame = set()
        self.clients_control = set()

        self._setup_cors()
        self._setup_routes()

        if env is not None:
            self.init(env)

    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        static_path = epath.resource_path('braxviewer') / 'static'
        self.app.mount("/static", StaticFiles(directory=static_path), name="static")

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            if self.system_json:
                return render_from_json(sys=self.system_json, height=480, colab=True, base_url=None)
            else:
                return self._get_waiting_html()

        @self.app.websocket("/ws/frame")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.clients_frame.add(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    for client in self.clients_frame:
                        if client != websocket:
                            await client.send_text(data)
            except WebSocketDisconnect:
                self.clients_frame.remove(websocket)

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
                            self.rendering_enabled = bool(msg.get('enabled', False))
                            self.log(f"Real-time rendering toggled to: {self.rendering_enabled}")
                        # Handle other control messages like refresh
                        elif msg.get('type') == 'refresh_web':
                            # This part of the logic is handled by the JS client reloading
                            pass
                    except Exception:
                        self.log(f"Invalid control message: {data}")

            except WebSocketDisconnect:
                self.clients_control.remove(websocket)


    def init(self, env: PipelineEnv):
        self.env = env
        from jax import random as jax_random
        rng = jax_random.PRNGKey(0)
        state = env.reset(rng)
        
        sys_with_dt = env.sys.tree_replace({'opt.timestep': env.dt})
        self.system_json = json.dumps(sys_with_dt, [state.pipeline_state])

        self.send_control({"type": "refresh_web"})
        
    def _get_waiting_html(self) -> HTMLResponse:
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Brax Visualizer - Waiting</title>
            <link rel="shortcut icon" type="image/x-icon" href="/static/favicon.ico">
            <style>
            body {
                margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: #1a1a1a; color: #e0e0e0;
                display: flex; justify-content: center; align-items: center;
                height: 100vh; text-align: center;
            }
            .container {
                background: #2a2a2a; padding: 2rem 3rem; border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            }
            h1 { font-size: 1.8rem; margin-bottom: 1rem; }
            p { font-size: 1rem; color: #b0b0b0; }
            code {
                background-color: #333; border: 1px solid #444;
                padding: 0.2rem 0.4rem; border-radius: 4px;
                font-size: 0.95rem; color: #ff8c8c;
            }
            .status {
                margin-top: 1.5rem; font-size: 0.9rem;
                color: #ffc107; /* Amber */
            }
            .status.connected { color: #81c784; /* Green */}
            .status.error { color: #e57373; /* Red */}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Viewer Not Initialized</h1>
                <p>Waiting for the 3D scene to be generated from Python.</p>
                <p>Please run <code>viewer.init(your_env)</code> in your script.</p>
                <div id="ws-status" class="status">Connecting to server...</div>
            </div>
            <script>
              function connectWebSocket() {
                const statusDiv = document.getElementById('ws-status');
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws/control`;
                const ws = new WebSocket(wsUrl);
                window.control_ws = ws; // Make ws globally accessible

                ws.onopen = () => {
                  console.log('Control WebSocket connected.');
                  statusDiv.textContent = 'Connected. Waiting for init()...';
                  statusDiv.className = 'status connected';
                };

                ws.onmessage = (event) => {
                  try {
                    const msg = JSON.parse(event.data);
                    if (msg && msg.type === 'refresh_web') {
                      console.log('Refresh command received. Reloading page.');
                      statusDiv.textContent = 'Scene ready. Reloading...';
                      setTimeout(() => window.location.reload(), 250);
                    }
                  } catch (e) {
                    console.error('Failed to parse message JSON:', e);
                  }
                };

                ws.onclose = () => {
                  console.log('WebSocket disconnected. Reconnecting in 3s...');
                  statusDiv.textContent = 'Connection lost. Retrying...';
                  statusDiv.className = 'status error';
                  setTimeout(connectWebSocket, 3000);
                };

                ws.onerror = (error) => {
                  console.error('WebSocket error:', error);
                  statusDiv.textContent = 'Connection error. Check console.';
                  statusDiv.className = 'status error';
                  ws.close();
                };
              }
              document.addEventListener('DOMContentLoaded', connectWebSocket);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

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
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level=self.log_level)
            server = uvicorn.Server(config)
            self._loop.run_until_complete(server.serve())

        if use_thread:
            self._server_thread = threading.Thread(target=start_server, daemon=True)
            self._server_thread.start()
            self.log(f"Brax Viewer starting on http://{self.host}:{self.port}")
            time.sleep(wait_for_startup)
            self.streamer.start()
        else:
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    def stop(self):
        self.streamer.stop()
        self.log("Viewer and streamer stopped.")

    def send_frame(self, state):
        self.streamer.send(state)
        if self.discard_queue:
            self.streamer.discard_queue()

    def log(self, message: str):
        self.logger.info(message)