<h1>
  <a href="#"><img alt="BRAX Viewer" src="_static/banner.png" width="100%"/></a>
</h1>

BraxViewer is a real-time, interactive web viewer for monitoring reinforcement learning (RL) policies during training on Brax, offering a convenient tool for debugging and analyzing the behavior of parallel environments in real-time, while seamlessly integrating into your current workflow.

Architected for JAX's Just-in-time (JIT) compilation paradigm, the viewer visualizes states from JIT-compiled functions and vectorized (vmap) rollouts without breaking the computation flow. It streams data from the Python backend to a web frontend, which is perfect for monitoring remote, headless training sessions with only a web browser. A toggle allows you to pause state synchronization on the fly, minimizing performance impact.

## Key Features

- **Real-time visualization**: Stream Brax environment states in real-time
- **Web-based**: Access visualization from any device with a web browser, perfect for remote headless training
- **JAX-compatible**: Designed to work seamlessly with JAX's JIT compilation with minimal impact on training performance
- **Lightweight**: Depends only on widely used standard packages
- **Interactive controls**: Enable/disable rendering through the web interface to save compute resource on the fly
- **Distributed architecture**: Viewer server runs independently from training process