from brax.envs import Env, State, Wrapper
from braxviewer.core.Sender import Sender
import jax
import jax.numpy as jnp

class ViewerWrapper(Wrapper):
    """A wrapper that provides rendering functionality for a Brax viewer."""

    def __init__(self, env: Env, sender: Sender):
        """Initializes the ViewerWrapper.

        Args:
            env: The environment to wrap.
            sender: An instance of a sender (e.g., WebViewer) that has a
                `send_frame` method and a `rendering_enabled` property.
        """
        super().__init__(env)
        self.sender = sender

    @property
    def should_render(self) -> jax.Array:
        """Returns a JAX array indicating whether rendering should occur."""
        return jnp.array(self.sender.rendering_enabled, dtype=jnp.bool_)

    def render_fn(self, state: State):
        """The function to be called for rendering a state.

        This function is designed to be used with `jax.experimental.io_callback`.
        When called within the callback, the batched state is a concrete value.
        """
        if not self.sender.rendering_enabled:
            return

        if state.pipeline_state.q.ndim > 1:
            # The state is batched, so we iterate through it in plain Python.
            num_envs = state.pipeline_state.q.shape[0]
            for i in range(num_envs):
                # Extract the state for a single environment.
                single_state = jax.tree_util.tree_map(lambda x: x[i], state)
                self.sender.send_frame(single_state, env_id=i)
        else:
            # The state is not batched, send it directly.
            self.sender.send_frame(state, env_id=0)
