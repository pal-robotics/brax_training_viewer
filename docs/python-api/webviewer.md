# WebViewer

The main interface for starting and managing the web viewer server.

## Purpose

WebViewer provides a high-level interface to start, manage, and control the web viewer server. It handles the initialization of both the server and sender components, and provides methods for frame transmission and server lifecycle management.

## Constructor

```python
WebViewer(
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
    server_log_level: str = "warning",
    **kwargs
)
```

## Parameters

- `host` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Server host address (default: "127.0.0.1")
- `port` ([int](https://docs.python.org/3/c-api/long.html)): Server port (default: 8000)
- `log_level` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Logging level for the viewer (default: "info")
- `server_log_level` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Logging level for the server (default: "warning")
- `**kwargs`: Additional configuration parameters passed to Config

## Methods

### run(use_thread: bool = True, wait_for_startup: int = 2)

Start the server and initialize the viewer.

**Parameters:**
- `use_thread` ([bool](https://docs.python.org/3/c-api/bool.html)): Whether to run server in background thread (default: True)
- `wait_for_startup` ([int](https://docs.python.org/3/c-api/long.html)): Seconds to wait for server startup (default: 2)

### stop()

Stop the server and cleanup resources.

### send_frame(state: State, env_id: int = 0)

Send a frame to the viewer.

**Parameters:**
- `state` ([Brax State](https://github.com/google/brax/blob/main/brax/envs/base.py)): Brax environment state to send
- `env_id` ([int](https://docs.python.org/3/c-api/long.html)): Environment ID for batched environments (default: 0)

### stitch_state(batched_state: State) -> State

Convert batched state to single state.

**Parameters:**
- `batched_state` ([Brax State](https://github.com/google/brax/blob/main/brax/envs/base.py)): Batched state from environment

**Returns:**
- [Brax State](https://github.com/google/brax/blob/main/brax/envs/base.py): Single environment state

### log(message: str, level: str = "info")

Log a message with specified level.

**Parameters:**
- `message` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Log message
- `level` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Log level (default: "info")

## Properties

### rendering_enabled

Get or set the rendering enabled state.

**Type:** [bool](https://docs.python.org/3/c-api/bool.html)

**Access:** Read/Write 