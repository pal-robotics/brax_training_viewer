# ConnectionManager

Manages active WebSocket connections for control signals.

## Purpose

ConnectionManager handles WebSocket connections for control messages between the server and clients. It maintains a list of active connections and provides methods for broadcasting control messages to all connected clients.

## Constructor

```python
ConnectionManager(sender)
```

## Parameters

- `sender`: [`BraxSender`](../python-api/braxsender.md) instance for communication

## Methods

### connect(websocket: WebSocket)

Accept a new WebSocket connection.

**Parameters:**
- `websocket` ([WebSocket](https://www.starlette.io/websockets/)): WebSocket connection to accept

### disconnect(websocket: WebSocket)

Remove a WebSocket connection.

**Parameters:**
- `websocket` ([WebSocket](https://www.starlette.io/websockets/)): WebSocket connection to remove

### broadcast(message: dict)

Broadcast message to all connected clients.

**Parameters:**
- `message` ([dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)): Message to broadcast to all clients 