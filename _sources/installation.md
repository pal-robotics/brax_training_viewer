# Installation

This guide covers installing BraxViewer and its dependencies.

## Prerequisites

- **Python 3.10 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Install Git](https://git-scm.com/downloads)
- **A package manager** (pip or uv) - [Install pip](https://pip.pypa.io/en/stable/installation/) or [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation Methods

Choose your preferred installation method:

- **[Pip Installation](installation-pip)**: Traditional Python package installation using pip
- **[UV Installation](installation-uv)**: Fast, modern Python package installation using uv

## Dependencies

BraxViewer automatically installs:

- **FastAPI** (≥0.115.13): Web framework
- **Uvicorn** (≥0.34.3): ASGI server
- **WebSockets** (≥15.0.1): Real-time communication
- **asyncio** (≥3.4.3): Async support

### Optional Dependencies

- **JAX**: For numerical computations - [Install JAX](https://github.com/google/jax#installation)
- **Brax**: For reinforcement learning environments (included as submodule)

:::{important}
This viewer requires the **modified Brax submodule** which contains additional features needed for real-time visualization. **DO NOT** install the official Brax package from PyPI as it is not compatible.
:::

## Verification

After installation, verify that BraxViewer is working:

```python
from braxviewer import WebViewer
```

## Next Steps

After successful installation, proceed to the [Quick Start Guide](quick-start) to begin using BraxViewer.

For additional help, please [open an issue](https://github.com/pal-robotics/brax_training_viewer/issues) on GitHub. 