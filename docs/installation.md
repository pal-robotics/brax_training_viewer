# Installation

This guide covers installing BraxViewer and its dependencies.

## Prerequisites

- **Python 3.10 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Install Git](https://git-scm.com/downloads)
- **A package manager** (pip) - [Install pip](https://pip.pypa.io/en/stable/installation/)

## Steps

### (Optional) Using Conda Virtual Environment

1. Install [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)
2. Create a virtual environment: 
    ```bash
    conda create -n virtual_env_name python=3.10
    ```
3. Activate the environment: 
    ```bash
    conda activate virtual_env_name
    ```

### Repository Setup

1. Download the repository
    ```bash
    git clone https://github.com/pal-robotics/brax_training_viewer.git
    ```
2. Change work directory
    ```bash
    cd brax_training_viewer
    ```

3. Choose your preferred method to install dependencies:
    - **[Pip Installation](installation/installation-pip)**: Traditional Python package installation using pip


## Dependencies

- **FastAPI** (≥0.115.13): Web framework
- **Uvicorn** (≥0.34.3): ASGI server
- **WebSockets** (≥15.0.1): Real-time communication
- **asyncio** (≥3.4.3): Async support
- **JAX**: For numerical computations - [Install JAX](https://github.com/google/jax#installation)

    :::{note}
    Install the version compatible for your hardware to accelerate codes.
    :::
- **Brax**: For reinforcement learning environments

## Next Steps

After successful installation, proceed to the [Quick Start Guide](quick-start) to begin using BraxViewer.

For additional help, please [open an issue](https://github.com/pal-robotics/brax_training_viewer/issues) on GitHub. 