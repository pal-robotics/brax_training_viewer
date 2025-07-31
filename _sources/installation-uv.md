# UV Installation

This guide covers installing BraxViewer using uv.

## Quick Start

```bash
uv add braxviewer
```

## What is UV?

UV is a fast Python package installer and resolver, written in Rust. It offers:

- **Speed**: 10-100x faster than pip
- **Reliability**: Deterministic dependency resolution
- **Modern**: Built for Python 3.8+ with modern tooling
- **Compatibility**: Works with existing pip workflows

## Prerequisites

### Install UV

[UV](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver, written in Rust.

```bash
pip install uv
```

### Verify UV Installation
```bash
uv --version
```

## Detailed Installation

### 1. Verify Python Version
```bash
python --version
```

### 2. Clone Repository and Initialize Submodules

```bash
# Clone the repository
git clone https://github.com/pal-robotics/brax_training_viewer.git
cd brax_training_viewer

# Initialize the Brax submodule (required)
git submodule update --init --recursive
```

### 3. Install Dependencies

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install the local Brax submodule (required)
uv pip install -e brax/

# Install the project
uv pip install .
```

### 4. Optional: Install JAX with Hardware Acceleration

```bash
# For CUDA 12
uv pip install -U "jax[cuda12]"

# For CUDA 11
uv pip install -U "jax[cuda11]"

# For TPU
uv pip install -U "jax[tpu]"
```

### 5. Verify Installation
```python
from braxviewer import WebViewer
```

## Project-Based Installation

### 1. Initialize a New Project
```bash
mkdir my-braxviewer-project
cd my-braxviewer-project
uv init
```

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/pal-robotics/brax_training_viewer.git
cd brax_training_viewer

# Initialize submodules
git submodule update --init --recursive

# Add dependencies
uv add -r requirements.txt
uv pip install -e brax/
uv add .
```

### 3. Run with UV
```bash
uv run python -c "from braxviewer import WebViewer; print('Success!')"
```

## Development Installation

```bash
# Clone the repository
git clone https://github.com/pal-robotics/brax_training_viewer.git
cd brax_training_viewer

# Initialize submodules
git submodule update --init --recursive

# Initialize UV project
uv init

# Install in development mode
uv pip install -e brax/
uv pip install -e .
uv pip install -r requirements.txt
```

## Virtual Environment Management

UV automatically manages virtual environments:

```bash
# UV automatically creates and manages virtual environments
uv venv

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -e brax/
uv pip install .
```

## Troubleshooting

### Common Issues

#### ModuleNotFoundError: No module named 'brax'
```bash
# Run submodule update
git submodule update --init --recursive

# Install the local Brax submodule
uv pip install -e brax/
```

#### Conflicts with Official Brax
```bash
# Uninstall official brax
uv pip uninstall brax

# Install the modified version
uv pip install -e brax/
```

:::{important}
If you installed the official Brax package from PyPI, uninstall it first before installing the modified version.
:::

## Uninstallation

```bash
# Remove from project
uv remove braxviewer

# Or uninstall globally
uv pip uninstall braxviewer
```

## Next Steps

After successful installation, proceed to the [Quick Start Guide](quick-start) to begin using BraxViewer.

## Getting Help

If you continue to experience issues:

1. Check the [Pip Installation Guide](installation-pip) as an alternative
2. Review the [Troubleshooting section](#troubleshooting) above
3. [Open an issue](https://github.com/pal-robotics/brax_training_viewer/issues) on GitHub
4. Check the [UV documentation](https://docs.astral.sh/uv/) for more information 