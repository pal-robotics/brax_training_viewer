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

- `env`: The Brax environment to wrap
- `sender`: A sender instance (e.g., BraxSender) with `send_frame` method and `rendering_enabled` property

## Properties

### should_render

Returns a JAX array indicating whether rendering should occur.

**Type:** jax.Array

**Access:** Read-only

## Methods

### render_fn(state: State)

Function to be called for rendering a state.

**Parameters:**
- `state` (State): Environment state to render

**Note:** This function is designed to be used with `jax.experimental.io_callback`. 