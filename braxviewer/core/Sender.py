from brax.io import json
from brax.io import json
import jax.numpy as jnp
import jax
from braxviewer.core.StateStreamer import StateStreamer
from braxviewer.core.config import Config
import logging
import json as std_json

class Sender:
    def __init__(self, config: Config):
        self.config = config
        self.system_json = None
        self.rendering_enabled = True
        self.streamer = StateStreamer(
            uri=f"ws://{self.config.host}:{self.config.port}/ws/frame",
            unbatched=self.config.unbatched_stream
        )
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("brax.sender")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        
        level_mapping = {
            "debug": logging.DEBUG, "info": logging.INFO,
            "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL
        }
        logger.setLevel(level_mapping.get(self.config.log_level.lower(), logging.INFO))
        return logger

    def init(self, xml_string: str):
        from brax.io import mjcf
        
        sys = mjcf.loads(xml_string)

        class TempEnv:
            def __init__(self, sys):
                self.sys = sys
                self.dt = sys.opt.timestep
            
            def reset(self, rng):
                from brax.envs.base import State
                q = jnp.zeros((self.sys.q_size(),))
                qd = jnp.zeros((self.sys.qd_size(),))
                pipeline_state = self.pipeline_init(q, qd)
                return State(pipeline_state=pipeline_state, obs=None, reward=None, done=None)
            
            def pipeline_init(self, q, qd):
                # This logic is preserved from the original implementation
                from brax.positional import pipeline as positional_pipeline
                from brax.spring import pipeline as spring_pipeline
                from brax.generalized import pipeline as generalized_pipeline
                try:
                    return positional_pipeline.init(self.sys, q, qd)
                except:
                    try:
                        return spring_pipeline.init(self.sys, q, qd)
                    except:
                        return generalized_pipeline.init(self.sys, q, qd)

        env = TempEnv(sys)
        
        from jax import random as jax_random
        rng = jax_random.PRNGKey(0)
        pipeline_state_to_serialize = None
        try:
            env_state = env.reset(rng)
            pipeline_state_to_serialize = env_state.pipeline_state
        except Exception:
            q = jnp.zeros((env.sys.q_size(),))
            qd = jnp.zeros((env.sys.qd_size(),))
            pipeline_state_to_serialize = env.pipeline_init(q, qd)

        sys_with_dt = env.sys.tree_replace({'opt.timestep': env.dt})
        self.system_json = json.dumps(sys_with_dt, [pipeline_state_to_serialize])

    def send_frame(self, state):
        if not self.rendering_enabled:
            return

        if hasattr(state, 'pipeline_state') and hasattr(state.pipeline_state, 'q') and state.pipeline_state.q.ndim > 1:
            state_to_render = jax.tree_util.tree_map(lambda x: x[0], state)
        else:
            state_to_render = state
            
        pipeline_state = state_to_render.pipeline_state
        
        message = {
            "x": {
                "pos": pipeline_state.x.pos.tolist(),
                "rot": pipeline_state.x.rot.tolist()
            }
        }
        self.streamer.send(std_json.dumps(message))

    def start(self):
        self.streamer.start()
        self.logger.info(f"Brax Sender streamer initialized.")
        self.logger.info(f"Application log level: {self.config.log_level}")

    def stop(self):
        self.streamer.stop()

    def log(self, message: str, level: str = "info"):
        getattr(self.logger, level, self.logger.info)(message) 