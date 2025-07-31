# Integration Guide

This guide covers how to integrate the BraxViewer server with your training workflows.

## Server API

The Brax Web Viewer server provides a real-time visualization interface for Brax environments via a FastAPI backend. It is designed for seamless integration with distributed or batched RL training, supporting both custom and built-in Brax environments.

## Architecture Overview

The server architecture consists of several key components:

- **WebViewer**: High-level interface to start/stop the server and manage communication
- **Server**: FastAPI application exposing HTTP and WebSocket endpoints for control and frame streaming
- **BraxSender**: Handles serialization and transmission of environment state to the viewer
- **ViewerWrapper**: Brax environment wrapper that triggers frame sending during training

## Endpoints

### HTTP Endpoints

#### `GET /`
Returns the main HTML viewer interface.

**Response**: HTML page with the 3D visualization interface

#### `GET /api/rendering_status`
Returns the current rendering status.

**Response**:
```json
{
  "rendering_enabled": true
}
```

#### `POST /api/rendering_status`
Sets the rendering state and broadcasts to all connected clients.

**Request Body**:
```json
{
  "rendering_enabled": true
}
```

**Response**:
```json
{
  "status": "success",
  "rendering_enabled": true
}
```

### WebSocket Endpoints

#### `ws://<host>:<port>/ws/frame`
Frame streaming endpoint for real-time visualization.

**Streamer Connection**: Sends frame data (JSON) to all connected viewers
**Viewer Connection**: Receives real-time frame updates

**Frame Data Format**:
```json
{
  "x": {
    "pos": [[x, y, z], ...],
    "rot": [[w, x, y, z], ...]
  }
}
```

#### `ws://<host>:<port>/ws/control`
Bi-directional control channel for toggling rendering and receiving state changes.

**Control Messages**:
```json
{
  "type": "toggle_render",
  "enabled": true
}
```

**State Updates**:
```json
{
  "type": "set_render_state",
  "enabled": true
}
```

## Usage Patterns

### 1. Start the Viewer Server

Run the viewer in a separate terminal. Example for a custom Cartpole:

```bash
python examples/brax/cartpole/viewer.py
```

Or for built-in Brax environments:

```bash
python examples/brax/brax_envs/viewer.py
```

### 2. Run Training

In another terminal, start your training script. The environment must be wrapped with `ViewerWrapper` and use a `BraxSender` configured to match the viewer server.

```bash
python examples/brax/cartpole/train.py
# or
python examples/brax/brax_envs/train.py
```

### 3. View in Browser

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to see real-time training visualization.

## Example Integration

### Custom Environment (Cartpole)

See `examples/brax/cartpole/train.py` and `viewer.py` for a complete example.

The training script creates a `BraxSender`, wraps the environment with `ViewerWrapper`, and sends frames to the running viewer server.

### Built-in Brax Environments

See `examples/brax/brax_envs/train.py` and `viewer.py` for a complete example.

The viewer loads the correct XML asset and number of environments based on configuration.

## Configuration

### Network Configuration

```python
HOST = "127.0.0.1"
PORT = 8000
```

### Environment Configuration

For custom environments:
```python
XML_MODEL = """
<mujoco model="inverted pendulum">
    <!-- XML model definition -->
</mujoco>
"""
```

For built-in environments:
```python
ENV_NAME = 'humanoid_simplfied'  # Available: ant, halfcheetah, hopper, humanoid, etc.
BACKEND = 'mjx'  # Available: generalized, positional, spring, mjx
```

### Training Configuration

Hyperparameters are typically defined in `config.py` files within the examples. See the example directories for complete configurations.

## Minimal Example

```python
# Start viewer server
from braxviewer.WebViewer import WebViewer

viewer = WebViewer(
    host="127.0.0.1", 
    port=8000, 
    xml=xml_string, 
    num_envs=8
)
viewer.run()

# In training script
from braxviewer.BraxSender import BraxSender
from braxviewer.wrapper import ViewerWrapper

sender = BraxSender(
    host="127.0.0.1", 
    port=8000, 
    xml=xml_string, 
    num_envs=8
)
sender.start()

env = ViewerWrapper(env=env, sender=sender)
```

## Important Notes

- The viewer and training script must use consistent network and environment configuration
- The server supports multiple clients and can be run in a background thread
- Rendering can be toggled at runtime via the API or control WebSocket
- The server automatically handles client disconnections and reconnections
- Frame streaming is optimized to minimize performance impact on training

## Error Handling

The server includes robust error handling for:
- Client disconnections
- Invalid message formats
- Network connectivity issues
- Rendering state synchronization

All errors are logged with appropriate levels and the server continues operating normally. 