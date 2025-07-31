# WebSocket Endpoints

The Brax Web Viewer provides WebSocket endpoints for real-time communication and frame streaming.

## Frame Streaming

### `ws://<host>:<port>/ws/frame`

Frame streaming endpoint for real-time visualization.

**Connection Types**:
- **Streamer**: Sends frame data to all connected viewers
- **Viewer**: Receives real-time frame updates

**Frame Data Format**:
```json
{
  "x": {
    "pos": [[x, y, z], ...],
    "rot": [[w, x, y, z], ...]
  }
}
```

**Query Parameters**:
- `is_streamer=true`: Identifies the connection as a data streamer

**Description**: 
- Handles real-time frame streaming
- Streamers send frame data, viewers receive updates
- Supports multiple concurrent viewers
- Automatic cleanup of disconnected clients

**JavaScript Example (Viewer)**:
```javascript
const ws = new WebSocket("ws://127.0.0.1:8000/ws/frame");

ws.onopen = function() {
    console.log("Connected to frame stream");
};

ws.onmessage = function(event) {
    const frame = JSON.parse(event.data);
    // Update visualization with frame data
    updateViewer(frame);
};

ws.onclose = function() {
    console.log("Disconnected from frame stream");
};
```

**Python Example (Streamer)**:
```python
import websockets
import json

async def send_frames():
    uri = "ws://127.0.0.1:8000/ws/frame?is_streamer=true"
    async with websockets.connect(uri) as websocket:
        while True:
            frame_data = {
                "x": {
                    "pos": [[0, 0, 0], [1, 1, 1]],
                    "rot": [[1, 0, 0, 0], [0, 1, 0, 0]]
                }
            }
            await websocket.send(json.dumps(frame_data))
            await asyncio.sleep(0.016)  # ~60 FPS
```

## Control Communication

### `ws://<host>:<port>/ws/control`

Bi-directional control channel for toggling rendering and receiving state changes.

**Incoming Messages**:
```json
{
  "type": "toggle_render",
  "enabled": true
}
```

**Outgoing Messages**:
```json
{
  "type": "set_render_state",
  "enabled": true
}
```

**Description**: 
- Manages control messages for rendering state
- Clients can send toggle commands and receive state updates
- Broadcasts state changes to all connected clients
- Maintains connection state for all control clients

**JavaScript Example**:
```javascript
const controlWs = new WebSocket("ws://127.0.0.1:8000/ws/control");

controlWs.onopen = function() {
    console.log("Connected to control channel");
};

controlWs.onmessage = function(event) {
    const message = JSON.parse(event.data);
    if (message.type === "set_render_state") {
        updateRenderingUI(message.enabled);
    }
};

// Send control message
function toggleRendering(enabled) {
    controlWs.send(JSON.stringify({
        type: "toggle_render",
        enabled: enabled
    }));
}
```

**Python Example**:
```python
import websockets
import json

async def control_client():
    uri = "ws://127.0.0.1:8000/ws/control"
    async with websockets.connect(uri) as websocket:
        # Send toggle command
        await websocket.send(json.dumps({
            "type": "toggle_render",
            "enabled": False
        }))
        
        # Listen for state updates
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "set_render_state":
                print(f"Rendering enabled: {data['enabled']}")
```

## Connection Management

### Automatic Reconnection

The server handles connection management automatically:
- **Client Disconnection**: Automatic cleanup of disconnected clients
- **Invalid Messages**: Logged and ignored gracefully
- **Connection Limits**: No hard limits on concurrent connections

### Error Handling

**WebSocket Errors**:
- **Connection Disconnect**: Automatic cleanup of disconnected clients
- **Invalid Messages**: Logged and ignored
- **Serialization Errors**: Graceful handling of malformed data

**Common Error Scenarios**:
- Network interruptions
- Invalid JSON messages
- Unexpected message formats
- Client timeouts

## Performance Considerations

### Frame Streaming
- **Efficient Broadcasting**: Frame data broadcast to all connected viewers
- **No Backpressure**: Streamers can send at any rate
- **Automatic Cleanup**: Disconnected viewers automatically removed

### Control Communication
- **Lightweight Messages**: Control messages are small and efficient
- **Bidirectional**: Real-time state synchronization
- **Broadcast Updates**: State changes sent to all connected clients

### Concurrent Connections
- **Multiple Viewers**: Supports unlimited concurrent viewers
- **Multiple Streamers**: Supports multiple data sources
- **Resource Management**: Automatic cleanup prevents memory leaks 