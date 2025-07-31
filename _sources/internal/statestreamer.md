# StateStreamer

Optimized WebSocket streamer for frontend communication.

## Purpose

StateStreamer provides optimized WebSocket streaming for sending state data to the frontend. It runs in a background thread and handles automatic reconnection, queue management, and efficient state serialization.

## Constructor

```python
StateStreamer(
    uri="ws://localhost:8000/ws/frame",
    unbatched: bool = True
)
```

## Parameters

- `uri` (str): WebSocket URI for frame streaming (default: "ws://localhost:8000/ws/frame")
- `unbatched` (bool): Whether to use unbatched streaming (default: True)

## Methods

### start()

Start the background streaming thread.

### stop()

Stop the streaming thread and cleanup resources.

### send(state, discard_queue: bool = False)

Send a state to the WebSocket stream.

**Parameters:**
- `state`: State to send (can be State object or JSON string)
- `discard_queue` (bool): If True, discard all previous states in queue (default: False)

### discard_queue()

Discard all items in the state queue.

## Properties

### connected

Current connection status.

**Type:** bool

**Access:** Read-only 