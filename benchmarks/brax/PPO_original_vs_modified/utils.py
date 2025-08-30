
import platform
import jax
import subprocess

def get_hardware_identifier():
    """
    Tries to get a descriptive hardware name, prioritizing NVIDIA GPU names.
    Falls back to JAX device kind, and then to CPU info.
    """
    try:
        # Try nvidia-smi for specific NVIDIA GPU names
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return result.stdout.strip().replace(' ', '_')
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fallback for non-NVIDIA GPUs or if nvidia-smi is not installed
        try:
            device = jax.devices()[0]
            return device.device_kind.replace(' ', '_')
        except Exception:
            # Generic fallback
            return platform.processor().replace(' ', '_') or "unknown_cpu"
