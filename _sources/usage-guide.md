# Usage Guide

This guide covers detailed usage patterns and customization options for BraxViewer.

## Basic Usage

The typical workflow involves running the viewer server separately from your training process:

1. **Start the viewer server** in one terminal
2. **Run your training script** in another terminal
3. **View the visualization** in your web browser

## Viewer Server

The viewer server can be started using the `WebViewer` class or the provided example scripts:

```python
from braxviewer.WebViewer import WebViewer

viewer = WebViewer(
    host="127.0.0.1",
    port=8000,
    xml=xml_string,
    num_envs=8
)
viewer.run()
```

## Training Integration

To integrate with your training process, wrap your environment with `ViewerWrapper`:

```python
from braxviewer.BraxSender import BraxSender
from braxviewer.wrapper import ViewerWrapper

# Create sender to communicate with viewer
sender = BraxSender(
    host="127.0.0.1",
    port=8000,
    xml=xml_string,
    num_envs=8
)
sender.start()

# Wrap your environment
env = ViewerWrapper(env=env, sender=sender)
```

## Configuration

### Network Settings

Configure the network settings in your config file:

```python
HOST = "127.0.0.1"
PORT = 8000
```

### Environment Settings

For custom environments, provide the XML model:

```python
XML_MODEL = """
<mujoco model="your_model">
    <!-- Your XML definition -->
</mujoco>
"""
```

For built-in Brax environments:

```python
ENV_NAME = 'humanoid_simplfied'
BACKEND = 'mjx'
```

## Examples

See the `examples/` directory for complete working examples:

- `examples/brax/cartpole/`: Custom Cartpole environment
- `examples/brax/brax_envs/`: Built-in Brax environments

## Advanced Usage

For detailed API documentation and advanced integration patterns, see the [Python API](python-api) documentation.

## Troubleshooting

### Common Issues

1. **Viewer not connecting**: Ensure the server is running and the host/port match
2. **No visualization**: Check that rendering is enabled and frames are being sent
3. **Performance issues**: Consider reducing the number of environments or disabling rendering during heavy computation

### Debugging

Enable debug logging:

```python
viewer = WebViewer(
    host="127.0.0.1",
    port=8000,
    log_level="debug"
)
```

## Performance Considerations

- The viewer has minimal impact on training performance
- Rendering can be toggled on/off during training
- Multiple clients can connect to the same server
- Frame streaming is optimized for real-time visualization
