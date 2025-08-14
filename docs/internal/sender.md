# Sender

Base sender class for communication with the viewer.

## Purpose

Sender provides the base functionality for communicating with the web viewer. It handles state serialization, WebSocket communication, and basic lifecycle management. It serves as the foundation for more specialized sender classes like BraxSender.

## Constructor

```python
Sender(config: Config)
```

## Parameters

- `config` ([Config](config.md)): Configuration object.

## Attributes

- `config`: The configuration object for the sender.
- `system_json`: JSON representation of the system configuration.
- `rendering_enabled`: A boolean flag to enable or disable rendering.
- `streamer`: A [`StateStreamer`](statestreamer.md) instance for handling WebSocket communication.
- `logger`: A configured logger for logging messages.

## Methods

### init(xml_string: str)

Initialize with XML system configuration.

**Parameters:**
- `xml_string` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): XML model string for the environment.

### start()

Start the sender and WebSocket connection.

### stop()

Stop the sender and cleanup resources.

### send_frame(state)

Send a frame to the viewer. This method handles multi-dimensional states and extracts the position and rotation data for rendering.

**Parameters:**
- `state`: State to send to the viewer.

### log(message: str, level: str = "info")

Log a message with specified level.

**Parameters:**
- `message` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Log message.
- `level` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Log level (default: "info"). 