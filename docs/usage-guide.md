# Usage Guide

This guide covers detailed usage patterns and customization options for BraxViewer.

## Basic Usage

The typical workflow involves running the viewer server separately from your training process:

1. **Start the viewer server** in one terminal
2. **Run your training script** in another terminal
3. **View the visualization** in your web browser

## Viewer Server

The viewer server can be started using the `WebViewer` class:
`host`: the IP to host server
`port`: the port to host server
`xml_string`: your robot XML
`num_envs`: the number of parallel environments in your viewer

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

To integrate with your training process, wrap your environment using `ViewerWrapper` with a `BraxSender` instance:
`host`: IP of the host server you want to send to
`port`: port of the host server you want to send to
`xml_string`: your robot XML
`num_envs`: the number of parallel environments in your viewer


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

# Your Brax training codes
...
```

### PPO integration

```python
from braxviewer.brax.training.agents.ppo import train as ppo
```

- When `env` is wrapped by `ViewerWrapper`, `ppo.train(...)` automatically uses `env.render_fn` and dynamically reads `env.sender.rendering_enabled` to toggle rendering at runtime.
- No extra callback wiring is required. modified PPO algorithm already has callback integrated.

:::note
Make sure the IP and port are both available.
:::

## Examples

See the [Quick Start Guide](quick-start) for simple examples:
- `examples/brax/cartpole/`: Custom Cartpole environment
- `examples/brax/brax_envs/`: Built-in Brax environments

## Advanced Usage

For detailed API documentation and advanced integration patterns, see the [Python API](python-api) documentation.
