# ViewerWrapper

A Brax environment wrapper that enables real-time rendering during training.

## Purpose

ViewerWrapper wraps a Brax environment to add real-time rendering capabilities. It intercepts environment steps and sends state updates to the viewer server when rendering is enabled. This allows for seamless integration with existing training loops.

## Constructor

```python
ViewerWrapper(
    env: Env,
    sender: Sender
)
```

## Parameters

- `env`: The Brax environment to wrap ([Brax Env](https://github.com/google/brax/blob/main/brax/envs/base.py))
- `sender`: A [`Sender`](../internal/sender.md) instance (e.g., [`BraxSender`](braxsender.md)) with `send_frame` method and `rendering_enabled` property

## Properties

### `should_render`

Returns a JAX array indicating whether rendering should occur.

**Type:** [jax.Array](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html)

**Access:** Read-only

## Methods

### `render_fn(state: State)`

Function to be called for rendering a state.

**Parameters:**
- `state` ([Brax State](https://github.com/google/brax/blob/main/brax/envs/base.py)): Environment state to render

**Note:** This function is designed to be used with [`jax.experimental.io_callback`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.io_callback.html). 