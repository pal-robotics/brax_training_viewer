# Web API

The Brax Web Viewer provides HTTP and WebSocket endpoints for real-time visualization and control.

## Overview

The web API is built with FastAPI and provides two main types of endpoints:
- **HTTP endpoints** for server control and status
- **WebSocket endpoints** for real-time frame streaming and control communication

## Documentation Sections

- [HTTP Endpoints](web-api/http-endpoints): Detailed HTTP API documentation with examples
- [WebSocket Endpoints](web-api/websocket-endpoints): Real-time communication protocols
- [Error Handling](web-api/error-handling): Error handling patterns and troubleshooting

## Quick Reference

### HTTP Endpoints

- `GET /` - Main viewer interface
- `GET /api/rendering_status` - Get rendering status
- `POST /api/rendering_status` - Set rendering status

### WebSocket Endpoints

- `ws://<host>:<port>/ws/frame` - Frame streaming
- `ws://<host>:<port>/ws/control` - Control communication

### Basic Usage

**Start the server**:
```bash
python examples/brax/cartpole/viewer.py
```

**Access the viewer**:
Open `http://127.0.0.1:8000` in your browser

**Check status**:
```bash
curl http://127.0.0.1:8000/api/rendering_status
```

**Toggle rendering**:
```bash
curl -X POST http://127.0.0.1:8000/api/rendering_status \
  -H "Content-Type: application/json" \
  -d '{"rendering_enabled": false}'
```

## Architecture Features

- **CORS Support**: Cross-origin requests enabled
- **Concurrent Clients**: Multiple viewers supported
- **Automatic Cleanup**: Disconnected clients removed
- **Error Handling**: Comprehensive error management
- **Performance Optimized**: Efficient frame broadcasting

For detailed documentation, click on the links above.
