# Error Handling

The Brax Web Viewer includes comprehensive error handling for both HTTP and WebSocket endpoints.

## HTTP Error Handling

### Status Codes

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

### Common HTTP Errors

**Invalid JSON in POST Request**:
```json
{
  "status": "error",
  "message": "Invalid JSON format"
}
```

**Missing Required Fields**:
```json
{
  "status": "error",
  "message": "Missing required field: rendering_enabled"
}
```

**Server Processing Error**:
```json
{
  "status": "error",
  "message": "Internal server error occurred"
}
```

## WebSocket Error Handling

### Connection Errors

**Automatic Reconnection**: The server handles connection interruptions gracefully:
- Disconnected clients are automatically removed
- No manual reconnection required
- New connections are accepted immediately

**Connection Timeout**: WebSocket connections may timeout due to:
- Network interruptions
- Client inactivity
- Server restarts

### Message Errors

**Invalid JSON**: Malformed messages are logged and ignored:
```javascript
// This will be ignored
ws.send("invalid json");
```

**Unexpected Message Format**: Messages with wrong structure are handled gracefully:
```javascript
// This will be ignored
ws.send(JSON.stringify({unknown: "field"}));
```

**Missing Required Fields**: Control messages without required fields are ignored:
```javascript
// This will be ignored
ws.send(JSON.stringify({type: "toggle_render"})); // missing 'enabled'
```

## Logging

### Server Logs

The server includes comprehensive logging:
- **Connection Events**: Client connect/disconnect events
- **Error Events**: All errors logged with appropriate levels
- **Status Requests**: Optimized logging for frequent status checks

### Log Levels

- **INFO**: Normal operation events
- **WARNING**: Non-critical issues (disconnections, invalid messages)
- **ERROR**: Critical errors (server failures, processing errors)

### Log Filtering

Status endpoint logging is disabled to reduce noise:
- Frequent status requests don't clutter logs
- Other endpoints maintain full logging
- Error conditions are always logged

## Troubleshooting

### Common Issues

**Connection Refused**:
- Check if server is running
- Verify host and port configuration
- Ensure firewall allows connections

**WebSocket Connection Fails**:
- Verify WebSocket URL format
- Check for proxy/firewall issues
- Ensure client supports WebSocket protocol

**Frame Streaming Issues**:
- Verify streamer connection with `is_streamer=true`
- Check frame data format
- Ensure viewers are connected before streaming

**Control Messages Not Working**:
- Verify message format and required fields
- Check WebSocket connection status
- Ensure control channel is connected

### Debugging

**Enable Debug Logging**:
```python
# In server configuration
log_level = "debug"
```

**Check Connection Status**:
```javascript
// Browser console
console.log("WebSocket readyState:", ws.readyState);
```

**Monitor Network Traffic**:
- Use browser developer tools
- Check Network tab for WebSocket frames
- Monitor console for error messages

## Performance Considerations

### Error Recovery

- **Automatic Cleanup**: Disconnected clients automatically removed
- **Graceful Degradation**: Server continues operating with partial failures
- **Resource Management**: Memory leaks prevented through proper cleanup

### Error Prevention

- **Input Validation**: All inputs validated before processing
- **Type Checking**: JSON message structure verified
- **Bounds Checking**: Array indices and values validated

### Monitoring

- **Connection Count**: Track active connections
- **Error Rates**: Monitor error frequency
- **Performance Metrics**: Track response times and throughput 