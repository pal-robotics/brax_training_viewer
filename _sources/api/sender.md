# Sender

Base sender class for communication with the viewer.

## Purpose

Sender provides the base functionality for communicating with the web viewer. It handles state serialization, WebSocket communication, and basic lifecycle management. It serves as the foundation for more specialized sender classes like BraxSender.

## Constructor

```python
Sender(config: Config)
```

## Parameters

- `config`: Configuration object

## Methods

### init(xml_string: str)

Initialize with XML system configuration.

**Parameters:**
- `xml_string` (str): XML model string for the environment

### start()

Start the sender and WebSocket connection.

### stop()

Stop the sender and cleanup resources.

### send_frame(state)

Send a frame to the viewer.

**Parameters:**
- `state`: State to send to the viewer

### log(message: str, level: str = "info")

Log a message with specified level.

**Parameters:**
- `message` (str): Log message
- `level` (str): Log level (default: "info") 