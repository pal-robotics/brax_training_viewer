# HTTP Endpoints

The Brax Web Viewer provides HTTP endpoints for server control and status management.

## Main Interface

### `GET /`

Returns the main HTML viewer interface.

**Response**: HTML page with the 3D visualization interface

**Description**: 
- Serves the main viewer page
- If system JSON is available, renders the viewer with the system configuration
- Otherwise, shows a waiting page with instructions

**Example Usage**:
```bash
curl http://127.0.0.1:8000/
```

## Rendering Status API

### `GET /api/rendering_status`

Returns the current rendering status.

**Response**:
```json
{
  "rendering_enabled": true
}
```

**Description**: Returns the current rendering enabled state from the server.

**Example Usage**:
```bash
curl http://127.0.0.1:8000/api/rendering_status
```

**Python Example**:
```python
import requests

response = requests.get("http://127.0.0.1:8000/api/rendering_status")
status = response.json()
print(f"Rendering enabled: {status['rendering_enabled']}")
```

### `POST /api/rendering_status`

Sets the rendering state and broadcasts to all connected clients.

**Request Body**:
```json
{
  "rendering_enabled": true
}
```

**Success Response**:
```json
{
  "status": "success",
  "rendering_enabled": true
}
```

**Error Response**:
```json
{
  "status": "error",
  "message": "Error description"
}
```

**Description**: 
- Updates the rendering status on the server
- Broadcasts the change to all connected control clients
- Returns the updated status

**Example Usage**:
```bash
curl -X POST http://127.0.0.1:8000/api/rendering_status \
  -H "Content-Type: application/json" \
  -d '{"rendering_enabled": false}'
```

**Python Example**:
```python
import requests

# Enable rendering
response = requests.post(
    "http://127.0.0.1:8000/api/rendering_status",
    json={"rendering_enabled": True}
)
result = response.json()

# Disable rendering
response = requests.post(
    "http://127.0.0.1:8000/api/rendering_status",
    json={"rendering_enabled": False}
)
result = response.json()
```

## Error Handling

### HTTP Status Codes

- **200 OK**: Successful request
- **400 Bad Request**: Invalid request format or missing required fields
- **500 Internal Server Error**: Server processing error

### Error Response Format

```json
{
  "status": "error",
  "message": "Detailed error description"
}
```

## Rate Limiting

The server includes optimized logging for frequent status requests:
- Status endpoint logging is disabled to reduce noise
- No rate limiting is applied to status endpoints
- All requests are processed immediately

## CORS Support

All HTTP endpoints support CORS with the following configuration:
- **Allow Origins**: `*` (all origins)
- **Allow Methods**: `GET`, `POST`, `OPTIONS`
- **Allow Headers**: `*` (all headers)

This enables cross-origin requests from web applications. 