import shutil
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from etils import epath
from braxviewer.html_utils import render
import uvicorn
import time
import asyncio
import jax.random as jax_random
from brax.envs.base import PipelineEnv
from typing import Optional
import json
import logging
from braxviewer.WebSocketStreamer import WebSocketStreamer

class WebViewer:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 log_level: str = "error",
                 env: Optional[PipelineEnv] = None,):
        self.host = host
        self.port = port
        self.web_host_path = epath.resource_path('braxviewer') / '.cache' / 'web_host'
        self.logger = logging.getLogger("uvicorn.info")
        self.log_level = log_level
        self.streamer = WebSocketStreamer(uri=f"ws://{self.host}:{self.port}/ws/frame")

        self.app = FastAPI()
        self._setup_cors()
        self._copy_static_to_cache()
        self._init_html()

        self.clients_frame = set()
        self.clients_control = set()

        self.env = None
        if env is not None:
            self.init(env)

        self._setup_routes()

    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _copy_static_to_cache(self):
        src_static = epath.resource_path('braxviewer') / 'static'
        self.web_host_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_static, self.web_host_path / 'static', dirs_exist_ok=True)

    def _init_html(self):
        index_path = self.web_host_path / 'index.html'
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

                ws.onopen = () => {
                  console.log('Control WebSocket connected.');
                  statusDiv.textContent = 'Connected. Waiting for init()...';
                  statusDiv.className = 'status connected';
                };

                ws.onmessage = (event) => {
                  console.log('Received message:', event.data);
                  try {
                    const msg = JSON.parse(event.data);
                    if (msg && msg.type === 'refresh_web') {
                      console.log('Refresh command received. Reloading page.');
                      statusDiv.textContent = 'Scene ready. Reloading...';
                      // Use a small timeout to ensure the message is visible before reload
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
                  ws.close(); // This will trigger the onclose handler for reconnect
                };
              }
              document.addEventListener('DOMContentLoaded', connectWebSocket);
            </script>
        </body>
        </html>
        """
        with index_path.open('w') as f:
            f.write(html_content)

    def _setup_routes(self):
        @self.app.get("/")
        async def index():
            index_path = self.web_host_path / 'index.html'
            return HTMLResponse(index_path.read_text())

        @self.app.get("/static/{file_path:path}")
        async def static_files(file_path: str):
            full_path = self.web_host_path / 'static' / file_path
            if full_path.exists():
                return FileResponse(full_path)
            return {"error": "File not found"}

        @self.app.websocket("/ws/frame")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.clients_frame.add(websocket)
            # self.log(f"Client connected: {websocket.client}")
            try:
                while True:
                    data = await websocket.receive_text()
                    # self.log(f"Received frame: {data}")
                    for client in self.clients_frame:
                        if client != websocket:
                            await client.send_text(data)
            except WebSocketDisconnect:
                self.clients_frame.remove(websocket)
                # self.log(f"Client disconnected: {websocket.client}")

        @self.app.websocket("/ws/control")
        async def websocket_control(websocket: WebSocket):
            await websocket.accept()
            self.clients_control.add(websocket)
            # self.log(f"Control client connected: {websocket.client}")
            try:
                while True:
                    data = await websocket.receive_text()
                    # self.log(f"Received control message: {data}")
                    # TODO: parse command JSON and act accordingly
            except WebSocketDisconnect:
                self.clients_control.remove(websocket)
                # self.log(f"Control client disconnected: {websocket.client}")


    async def _broadcast_control(self, message: dict):
        if not self.clients_control: return
        tasks = [client.send_text(json.dumps(message)) for client in self.clients_control]
        await asyncio.gather(*tasks, return_exceptions=True)

    def send_control(self, message: dict):
        if self._server_thread and hasattr(self, '_loop'):
             asyncio.run_coroutine_threadsafe(self._broadcast_control(message), self._loop)

    def init(self, env):
        self.env = env
        index_path = self.web_host_path / 'index.html'
        rng = jax_random.PRNGKey(0)
        state = env.reset(rng)
        first_pipeline_state = state.pipeline_state
        sys_with_dt = env.sys.tree_replace({'opt.timestep': env.dt})
        with index_path.open('w') as f:
            f.write(render(sys_with_dt, [first_pipeline_state]))
        time.sleep(0.4) #Without this delay, the page may not load before html ovwerwrite
        self.send_control({"type": "refresh_web"})


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
        """Stops the viewer and its streamer."""
        self.streamer.stop()
        self.log("Viewer and streamer stopped.")

    def send_frame(self, state):
        """Sends a state to the WebSocket stream."""
        self.streamer.send(state)

    def log(self, message: str):
        self.logger.info(message)

if __name__ == "__main__":
    viewer = WebViewer()
    viewer.run()
    viewer.log("Viewer server started")

    import time
    while True:
        time.sleep(1)
        viewer.log("Main thread running...")