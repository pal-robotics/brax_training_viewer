# Overview
<div style="display: flex; align-items: center; justify-content: 
center; margin: 20px 0;">
  <img alt="Logo" src="_static/logo_fox.png" width="150" 
  style="margin-right: 30px;"/>
  <img alt="JAX SimViewer" src="_static/name.webp" width="500" 
  style=""/>
</div>

BraxViewer is a real-time, interactive web viewer for monitoring reinforcement learning (RL) policies during training on Brax, offering a convenient tool for debugging and analyzing the behavior of parallel environments in real-time, while seamlessly integrating into your current workflow.

Architected for JAX's Just-in-time (JIT) compilation paradigm, the viewer visualizes states from JIT-compiled functions and vectorized (vmap) rollouts without breaking the computation flow. It streams data from the Python backend to a web frontend, which is perfect for monitoring remote, headless training sessions with only a web browser. A toggle allows you to pause state synchronization on the fly, minimizing performance impact.

## Key Features

- **Real-time visualization**: Stream Brax environment states to a web browser
- **JAX-compatible**: Designed to work seamlessly with JAX's JIT compilation
- **Lightweight**: Minimal impact on training performance
- **Web-based**: Access visualization from any device with a web browser
- **Interactive controls**: Enable/disable rendering through the web interface
- **Distributed architecture**: Viewer server runs independently from training process

### About the Brax Submodule
This project includes a custom Brax fork as a git submodule that contains additional features needed for the viewer.

:::{important}
This viewer requires the **modified Brax submodule** which contains additional features needed for real-time visualization. **DO NOT** install the official Brax package from PyPI as it is not compatible.
:::

## Documentation

- [Quick Start](quick-start): Get up and running in minutes
- [Usage Guide](usage-guide): Detailed usage patterns and customization
- [Web API](web-api): HTTP and WebSocket endpoint documentation
- [Python API](python-api): Python class and component documentation

## Requirements

- Python 3.10+
- JAX
- Web browser
- Git (for submodule management)