
from typing import Optional

class ViewerConfig:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 log_level: str = "info",
                 server_log_level: str = "warning",
                 xml: Optional[str] = None,
                 unbatched_stream: bool = True):
        self.host = host
        self.port = port
        self.log_level = log_level
        self.server_log_level = server_log_level
        self.xml = xml
        self.unbatched_stream = unbatched_stream

class BatchedViewerConfig(ViewerConfig):
    def __init__(self,
                 num_envs: int,
                 grid_dims: Optional[tuple] = None,
                 env_offset: Optional[tuple] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_envs = num_envs
        self.grid_dims = grid_dims
        self.env_offset = env_offset
        self.unbatched_stream = False 