# Server

FastAPI application that handles HTTP and WebSocket endpoints.

## Purpose

Server manages the FastAPI web application that provides HTTP and WebSocket endpoints for the viewer. It handles client connections, frame broadcasting, and control message routing. The server runs independently and can handle multiple concurrent clients.

## Constructor

```python
Server(sender)
```

## Parameters

- `sender`: [`BraxSender`](../python-api/braxsender.md) instance for communication

## Methods

### run(use_thread: bool = True, wait_for_startup: int = 2)

Start the server.

**Parameters:**
- `use_thread` ([bool](https://docs.python.org/3/c-api/bool.html)): Whether to run server in background thread (default: True)
- `wait_for_startup` ([int](https://docs.python.org/3/c-api/long.html)): Seconds to wait for server startup (default: 2)

### stop()

Stop the server and cleanup resources.

### broadcast_to_frames(data: str)

Broadcast frame data to all clients.

**Parameters:**
- `data` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Frame data to broadcast 