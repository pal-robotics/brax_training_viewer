# BraxSender

Handles serialization and communication with the web viewer server.

## Purpose

BraxSender manages the communication between the training process and the web viewer server. It handles state serialization, frame transmission, and status polling. It also provides automatic reconnection and error handling for network issues.

## Constructor

```python
BraxSender(
    config=None,
    **kwargs
)
```

## Parameters

- `config`: Optional Config object
- `**kwargs`: Configuration parameters if config is not provided

## Methods

### start()

Start the sender and status polling thread.

### stop()

Stop the sender and cleanup resources.

### init()

Initialize the sender with system configuration.

### send_frame(state: State, env_id: int = 0)

Send a frame to the server.

**Parameters:**
- `state` (State): Brax environment state to send
- `env_id` (int): Environment ID for batched environments (default: 0)

### stitch_state(batched_state: State) -> State

Convert batched state to single state.

**Parameters:**
- `batched_state` (State): Batched state from environment

**Returns:**
- `State`: Single environment state

### set_rendering_enabled(enabled: bool)

Set rendering state via HTTP API.

**Parameters:**
- `enabled` (bool): Whether rendering should be enabled

**Returns:**
- `bool`: Success status

## Properties

### rendering_enabled

Get or set the rendering enabled state.

**Type:** bool

**Access:** Read/Write 