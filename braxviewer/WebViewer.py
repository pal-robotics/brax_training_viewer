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
from braxviewer.StateStreamer import StateStreamer
import json as std_json
import jax.numpy as jnp
import jax.tree_util

class WebViewer:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 log_level: str = "info",
                 server_log_level: str = "warning",
                 xml: str = None):
        self.host = host
        self.port = port
        
        # Set up application logger for WebViewer
        self.logger = logging.getLogger("brax.viewer")
        self.log_level = log_level
        
        # Set up server logger level (defaults to warning for less verbose output)
        self.server_log_level = server_log_level
        
        # Level mapping for both loggers
        level_mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        
        # Configure application logger
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            # Prevent propagation to parent loggers to avoid duplicate messages
            self.logger.propagate = False
        
        self.logger.setLevel(level_mapping.get(log_level.lower(), logging.INFO))
        
        # Configure uvicorn logger separately
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        
        # Set uvicorn logger levels
        uvicorn_logger.setLevel(level_mapping.get(self.server_log_level.lower(), logging.WARNING))
        uvicorn_access_logger.setLevel(level_mapping.get(self.server_log_level.lower(), logging.WARNING))
        
        self.streamer = StateStreamer(uri=f"ws://{self.host}:{self.port}/ws/frame")
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

        if xml is not None:
            self.init(xml)

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


    def init(self, xml_string: str, backend: str = 'mjx'):
        """Initialize the viewer with XML string instead of environment object."""
        from braxviewer.brax.brax.io import mjcf
        from braxviewer.brax.brax.envs.base import PipelineEnv
        
        # Create a temporary environment from XML to extract system info
        sys = mjcf.loads(xml_string)
        
        # Create a minimal PipelineEnv-like object with the system info we need
        class TempEnv:
            def __init__(self, sys):
                self.sys = sys
                self.dt = sys.opt.timestep
                
            def reset(self, rng):
                from braxviewer.brax.brax.envs.base import State
                q = jnp.zeros((self.sys.q_size(),))
                qd = jnp.zeros((self.sys.qd_size(),))
                pipeline_state = self.pipeline_init(q, qd)
                return State(pipeline_state=pipeline_state, obs=None, reward=None, done=None)
                
            def pipeline_init(self, q, qd):
                from braxviewer.brax.brax.positional import pipeline as positional_pipeline
                from braxviewer.brax.brax.spring import pipeline as spring_pipeline
                from braxviewer.brax.brax.generalized import pipeline as generalized_pipeline
                
                # Try different pipeline types based on the system
                try:
                    return positional_pipeline.init(self.sys, q, qd)
                except:
                    try:
                        return spring_pipeline.init(self.sys, q, qd)
                    except:
                        return generalized_pipeline.init(self.sys, q, qd)
        
        self.env = TempEnv(sys)
        
        from jax import random as jax_random
        from brax.io import json
        rng = jax_random.PRNGKey(0)
        pipeline_state_to_serialize = None
        try:
            env_state = self.env.reset(rng)
            pipeline_state_to_serialize = env_state.pipeline_state
        except Exception:
            q = jnp.zeros((self.env.sys.q_size(),))
            qd = jnp.zeros((self.env.sys.qd_size(),))
            pipeline_state_to_serialize = self.env.pipeline_init(q, qd)
        
        sys_with_dt = self.env.sys.tree_replace({'opt.timestep': self.env.dt})
        self.system_json = json.dumps(sys_with_dt, [pipeline_state_to_serialize])

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
                <p>Please run <code>viewer.init(your_xml)</code> in your script.</p>
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
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level=self.server_log_level)
            server = uvicorn.Server(config)
            self._loop.run_until_complete(server.serve())

        if use_thread:
            self._server_thread = threading.Thread(target=start_server, daemon=True)
            self._server_thread.start()
            self.log(f"Brax Viewer starting on http://{self.host}:{self.port}")
            self.log(f"Application log level: {self.log_level}")
            self.log(f"Server log level: {self.server_log_level}")
            time.sleep(wait_for_startup)
            self.streamer.start()
        else:
            self.streamer.start()
            uvicorn.run(self.app, host=self.host, port=self.port, log_level=self.server_log_level)

    def stop(self):
        self.streamer.stop()

    def send_frame(self, state):
        if not self.rendering_enabled:
            return
        
        # Unbatch the state if it has a leading dimension of 1
        if hasattr(state, 'pipeline_state') and hasattr(state.pipeline_state, 'q') and state.pipeline_state.q.ndim > 1:
            state_to_render = jax.tree_util.tree_map(lambda x: x[0], state)
        else:
            state_to_render = state

        pipeline_state = state_to_render.pipeline_state
        
        # Subsequent frames are state-only updates.
        message = {
            "x": {
                "pos": pipeline_state.x.pos.tolist(),
                "rot": pipeline_state.x.rot.tolist()
            }
        }
        self.streamer.send(std_json.dumps(message))

    def log(self, message: str, type: str = "info"):
        if type == "info":
            self.logger.info(message)
        elif type == "warning":
            self.logger.warning(message)
        elif type == "error":
            self.logger.error(message)
        elif type == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)