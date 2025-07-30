# JAX Integration (Custom Training)

Use this guide if you are not using the provided PPO and want to integrate the viewer into your own JAX training loop with `jax.experimental.io_callback`.

## Setup

- Start the viewer server (see examples or create one with `WebViewer`).
- Wrap your environment and create a sender:

```python
from braxviewer.BraxSender import BraxSender
from braxviewer.wrapper import ViewerWrapper

sender = BraxSender(host="127.0.0.1", port=8000, xml=xml_string, num_envs=num_envs)
sender.start()

env = ViewerWrapper(env=env, sender=sender)
```

## Core concepts

- `env.render_fn(state)`: Python callback that sends a frame. Safe to call via `io_callback`; handles single or batched `State`.
- `env.should_render`: device boolean (`jax.Array`) reflecting the server toggle. Use it to guard rendering in JIT code (no recompilation).
- In your JAX loop, wrap the callback with `lax.cond(should_render, ...)` to avoid side effects when disabled.

## Example: render hook in a `lax.scan` loop

```python
import jax
import jax.numpy as jnp
from jax.experimental import io_callback

def train_step(env, policy, state, key, num_steps,
               render_fn=None, should_render=jnp.array(False, jnp.bool_)):
  def body(carry, _):
    st, k = carry
    k, k2 = jax.random.split(k)
    action, _ = policy(st.obs, k)
    nst = env.step(st, action)

    def do_render(s):
      if render_fn is not None:
        io_callback(render_fn, None, s)

    jax.lax.cond(should_render, do_render, lambda s: None, nst)
    return (nst, k2), None

  (final_state, _), _ = jax.lax.scan(body, (state, key), None, length=num_steps)
  return final_state
```

### Passing the render signals

```python
should_render = env.should_render  # jax.Array(bool)
render_fn = env.render_fn          # Python callable

state = train_step(env, policy, state, key, num_steps, render_fn, should_render)
```