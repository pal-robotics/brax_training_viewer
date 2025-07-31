# Pip Installation

This guide covers installing BraxViewer using pip.

## Quick Start

```bash
pip install braxviewer
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
# Install the local Brax submodule (required)
pip install -e brax/

# Install braxviewer
pip install .

# Install additional requirements
pip install -r requirements.txt
```

### 4. Optional: Install JAX with Hardware Acceleration

```bash
# For CUDA 12
pip install -U "jax[cuda12]"

# For CUDA 11
pip install -U "jax[cuda11]"

# For TPU
pip install -U "jax[tpu]"
```

### 5. Verify Installation

```python
from braxviewer import WebViewer
print("BraxViewer installed successfully!")
```

## Development Installation

```bash
# Clone the repository
git clone https://github.com/pal-robotics/brax_training_viewer.git
cd brax_training_viewer

# Initialize submodules
git submodule update --init --recursive

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e brax/
pip install -e .
pip install -r requirements.txt
```

## Virtual Environment Setup

```bash
# Create virtual environment
python -m venv braxviewer-env
source braxviewer-env/bin/activate  # On Windows: braxviewer-env\Scripts\activate

# Clone and setup
git clone https://github.com/pal-robotics/brax_training_viewer.git
cd brax_training_viewer
git submodule update --init --recursive

# Install BraxViewer
pip install -e brax/
pip install .
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues


#### ./brax foler is empty
```bash
# Run submodule update
git submodule update --init --recursive

# Install the local Brax submodule
pip install -e brax/
```

#### Conflicts with Official Brax
```bash
# Uninstall official brax
pip uninstall brax

# Install the modified version
pip install -e brax/
```

:::{important}
If you installed the official Brax package from PyPI, uninstall it first before installing the modified version.
:::

#### Port Conflicts

`ERROR:    [Errno 48] error while attempting to bind on address ('x.x.x.x', xxxx): [errno 48] address already in use`

Check availability of a port. A port may be occpuied due to unexpected quit of a thread that uses the port.

## Uninstallation

```bash
pip uninstall braxviewer
```

## Next Steps

After successful installation, proceed to the [Quick Start Guide](quick-start) to begin using BraxViewer.

## Getting Help

If you continue to experience issues:

1. Check the [UV Installation Guide](installation-uv) as an alternative
2. Review the [Troubleshooting section](#troubleshooting) above
3. [Open an issue](https://github.com/pal-robotics/brax_training_viewer/issues) on GitHub 