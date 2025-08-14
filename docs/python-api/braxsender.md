# BraxSender

Handles serialization and communication with the web viewer server.

## Purpose

`BraxSender` manages the communication between the training process and the web viewer server. It handles state serialization, frame transmission, and status polling. It also provides automatic reconnection and error handling for network issues.

`BraxSender` extends the base `Sender` class and adds functionality for handling batched environments, including auto-calculating grid layouts and stitching states for visualization.

## Constructor

```python
BraxSender(
    config=None,
    **kwargs
)
```

## Parameters

- `config`: Optional [`Config`](../internal/config.md) object
- `**kwargs`: Configuration parameters if `config` is not provided

## Methods

### `start()`

Starts the sender, initializes the WebSocket connection, and begins a background thread to poll the server for rendering status.

### `stop()`

Stops the sender, closes the WebSocket connection, and terminates the status polling thread.

### `init()`

Initializes the sender with the concatenated XML system configuration for all environments.

### `send_frame(state: State, env_id: int = 0)`

Sends a single frame from a specific environment to the server. It calculates the correct position offset for the environment based on its `env_id` before sending.

**Parameters:**
- `state` ([Brax State](https://github.com/google/brax/blob/main/brax/envs/base.py)): The Brax environment state to send.
- `env_id` ([int](https://docs.python.org/3/c-api/long.html)): The ID of the environment for batched environments (default: 0).

### `stitch_state(batched_state: State) -> State`

Stitches a batched state from multiple environments into a single state that can be rendered as a grid. This involves offsetting the positions of each environment in the batch.

**Parameters:**
- `batched_state` ([Brax State](https://github.com/google/brax/blob/main/brax/envs/base.py)): The batched state from the environment.

**Returns:**
- ([Brax State](https://github.com/google/brax/blob/main/brax/envs/base.py)): A single environment state with all environments stitched together.

### `set_rendering_enabled(enabled: bool)`

Sets the rendering state on the server via an HTTP POST request.

**Parameters:**
- `enabled` ([bool](https://docs.python.org/3/c-api/bool.html)): Whether rendering should be enabled.

**Returns:**
- ([bool](https://docs.python.org/3/c-api/bool.html)): `True` if the request was successful, `False` otherwise.

## Properties

### `rendering_enabled`

Gets or sets the rendering enabled state.

**Type:** [bool](https://docs.python.org/3/c-api/bool.html)

**Access:** Read/Write