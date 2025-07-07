import jax
import jax.numpy as jnp
import time
from brax.envs.base import PipelineEnv

from braxviewer.WebViewer import WebViewer
from braxviewer.html_utils import render
from braxviewer.utils import stitch_state

class WebViewerBatched(WebViewer):
    def __init__(self,
                 grid_spacing: float = 10.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.grid_spacing = grid_spacing
        self.num_envs = None
        self.streamer.unbatched = False

    def init(self, env_concatenated: PipelineEnv):
        self.env = env_concatenated
        index_path = self.web_host_path / 'index.html'
        sys = env_concatenated.sys
        q_size = sys.q_size()
        qd_size = sys.qd_size()
        q = jnp.zeros(q_size)
        qd = jnp.zeros(qd_size)
        pipeline_state = env_concatenated.pipeline_init(q, qd)
        sys_with_dt = env_concatenated.sys.tree_replace({'opt.timestep': env_concatenated.dt})
        with index_path.open('w') as f:
            f.write(render(sys_with_dt, [pipeline_state]))
        time.sleep(0.4)
        self.send_control({"type": "refresh_web"})

    def send_frame(self, state):
        stitched = stitch_state(state)
        self.streamer.send(stitched)