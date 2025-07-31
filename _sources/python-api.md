# Python API

This section provides detailed documentation for all Python classes and components in the BraxViewer system.

## Core Classes

- [WebViewer](api/webviewer): Main interface for starting and managing the web viewer server
- [BraxSender](api/braxsender): Handles serialization and communication with the web viewer server
- [ViewerWrapper](api/viewerwrapper): Brax environment wrapper that enables real-time rendering during training
- [Config](api/config): Configuration class for the viewer system

## Server Components

- [Server](api/server): FastAPI application that handles HTTP and WebSocket endpoints
- [ConnectionManager](api/connectionmanager): Manages active WebSocket connections for control signals

## Streaming Components

- [StateStreamer](api/statestreamer): Optimized WebSocket streamer for frontend communication
- [Sender](api/sender): Base sender class for communication with the viewer

## Utility Components

- [HTML Utils](api/html-utils): Utilities for HTML rendering and visualization
- [Utils](api/utils): Utility functions for state conversion and processing

## Quick Reference

### Basic Usage

```python
from braxviewer.WebViewer import WebViewer
from braxviewer.BraxSender import BraxSender
from braxviewer.wrapper import ViewerWrapper

# Start viewer
viewer = WebViewer(host="127.0.0.1", port=8000, xml=xml_string, num_envs=8)
viewer.run()

# In training script
sender = BraxSender(host="127.0.0.1", port=8000, xml=xml_string, num_envs=8)
sender.start()

env = ViewerWrapper(env=env, sender=sender)
```

### Configuration

```python
from braxviewer.config import Config

config = Config(
    host="127.0.0.1",
    port=8000,
    log_level="info",
    xml=xml_string,
    num_envs=8
)
```

For detailed documentation of each component, click on the links above. 