# Pip Installation

This guide covers installing dependencies using pip.

### Method 1: Install from Git (recommended)

- Single command:

```bash
pip install git+https://github.com/pal-robotics/brax_training_viewer.git
```

### Method 2: Install from source (development)

1. (Optional) create and activate a virtual environment
   - Using conda: `conda create -n your_env python=3.10 && conda activate your_env`
2. Clone the repository

```bash
git clone https://github.com/pal-robotics/brax_training_viewer.git
cd brax_training_viewer
```

3. Install the package in editable mode and its requirements

```bash
pip install -e .
pip install -r requirements.txt
```

## Optional: Hardware Acceleration

Install JAX with hardware acceleration for improved performance:

```bash
# For CUDA 12
pip install -U "jax[cuda12]"

# For CUDA 11
pip install -U "jax[cuda11]"

# For TPU
pip install -U "jax[tpu]"
```

## Getting Help

If you continue to experience issues:

- Review the [Troubleshooting Guide](troubleshooting) for common solutions
- [Open an issue](https://github.com/pal-robotics/brax_training_viewer/issues) on GitHub 