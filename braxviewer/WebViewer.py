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
from braxviewer.config import ViewerConfig
from braxviewer.viewer import Viewer
from braxviewer.server import Server

class WebViewer:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 log_level: str = "info",
                 server_log_level: str = "warning",
                 xml: str = None):
        
        self.config = ViewerConfig(
            host=host,
            port=port,
            log_level=log_level,
            server_log_level=server_log_level,
            xml=xml
        )
        self.viewer = Viewer(self.config)
        self.server = Server(self.viewer)
        
        if self.config.xml is not None:
            self.init(self.config.xml)

    def init(self, xml_string: str):
        self.viewer.init(xml_string)
        # Notify the server to refresh web clients
        self.server.send_control({"type": "refresh_web"})
        
    def run(self, use_thread: bool = True, wait_for_startup: int = 2):
        self.server.run(use_thread=use_thread, wait_for_startup=wait_for_startup)

    def stop(self):
        self.server.stop()

    def send_frame(self, state):
        self.viewer.send_frame(state)
        
    def log(self, message: str, level: str = "info"):
        self.viewer.log(message, level)

    @property
    def rendering_enabled(self):
        return self.viewer.rendering_enabled

    @rendering_enabled.setter
    def rendering_enabled(self, value):
        self.viewer.rendering_enabled = value