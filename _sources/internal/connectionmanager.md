# ConnectionManager

Manages active WebSocket connections for control signals.

## Purpose

ConnectionManager handles WebSocket connections for control messages between the server and clients. It maintains a list of active connections and provides methods for broadcasting control messages to all connected clients.

## Constructor

```python
ConnectionManager(sender)
```

## Parameters

- `sender`: BraxSender instance for communication

## Methods

### connect(websocket: WebSocket)

Accept a new WebSocket connection.

**Parameters:**
- `websocket` (WebSocket): WebSocket connection to accept

### disconnect(websocket: WebSocket)

Remove a WebSocket connection.

**Parameters:**
- `websocket` (WebSocket): WebSocket connection to remove

### broadcast(message: dict)

Broadcast message to all connected clients.

**Parameters:**
- `message` (dict): Message to broadcast to all clients 