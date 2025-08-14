
from typing import Optional

class Config:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 log_level: str = "warning",
                 server_log_level: str = "warning",
                 xml: Optional[str] = None,
                 unbatched_stream: bool = True,
                 num_envs: Optional[int] = None,
                 grid_dims: Optional[tuple] = None,
                 env_offset: Optional[tuple] = None):
        # Network and server configuration
        self.host = host
        self.port = port
        self.log_level = log_level
        self.server_log_level = server_log_level
        
        # Environment configuration
        self.xml = xml
        self.unbatched_stream = unbatched_stream
        
        # Grid layout configuration (for batched environments)
        self.num_envs = num_envs
        self.grid_dims = grid_dims
        self.env_offset = env_offset
        
        # For backward compatibility, set unbatched_stream to False if num_envs is provided
        if num_envs is not None:
            self.unbatched_stream = False

# Backward compatibility aliases
ViewerConfig = Config
BatchedViewerConfig = Config 