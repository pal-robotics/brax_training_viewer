import jax.numpy as jnp
import xml.etree.ElementTree as ET
import json as std_json
import math
from brax.envs.base import State

from braxviewer.core.config import Config
from braxviewer.core.server import Server
from braxviewer.BraxSender import BraxSender

class WebViewer:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 log_level: str = "info",
                 server_log_level: str = "warning",
                 **kwargs):

        self.config = Config(host=host, port=port, log_level=log_level,
                                          server_log_level=server_log_level, **kwargs)
        
        self.sender = BraxSender(self.config)
        self.server = Server(self.sender)
        
        self.sender.init()

    def run(self, use_thread: bool = True, wait_for_startup: int = 2):
        self.server.run(use_thread=use_thread, wait_for_startup=wait_for_startup)

    def stop(self):
        self.server.stop()

    def send_frame(self, state: State, env_id: int = 0):
        self.sender.send_frame(state, env_id)
        
    def stitch_state(self, batched_state: State) -> State:
        return self.sender.stitch_state(batched_state)

    def log(self, message: str, level: str = "info"):
        self.sender.log(message, level)

    @property
    def rendering_enabled(self):
        return self.sender.rendering_enabled

    @rendering_enabled.setter
    def rendering_enabled(self, value):
        self.sender.rendering_enabled = value