# Brax Environments Example

This example demonstrates how to train Brax environments with the separated web viewer architecture.

## Architecture

This example shows how to use pre-defined Brax environments with the web viewer:

- `config.py`: Shared configuration for network settings, environment parameters, and training hyperparameters
- `train.py`: Training script that loads Brax environments and connects to the viewer
- `viewer.py`: Independent viewer server that can be run separately

### PPO import

```python
from braxviewer.brax.training.agents.ppo import train as ppo
```

## Usage

### 1. Start the Viewer Server

First, start the viewer server in a separate terminal:

```bash
python examples/brax/brax_envs/viewer.py
```

This will start the WebViewer server on `http://127.0.0.1:8000`. You should see output like:

```
Viewer server started. Visit http://127.0.0.1:8000
Environment: humanoid_simplfied (Brax: humanoid)
Press Ctrl+C to stop.
```

### 2. Run Training

In another terminal, run the training script:

```bash
python examples/brax/brax_envs/train.py
```

### 3. View Training in Browser

Open your web browser and navigate to `http://127.0.0.1:8000` to see the training visualization in real-time.

## Configuration

You can modify the environment and training parameters in `config.py`:

- `ENV_NAME`: Choose from available Brax environments (default: 'humanoid_simplfied')
- `BACKEND`: Choose the physics backend (default: 'mjx')
- `HOST` and `PORT`: Network configuration for the viewer (default: 127.0.0.1:8000)
- Training hyperparameters are automatically selected based on the environment

## Available Environments

The example supports all Brax built-in environments:

- `ant`
- `halfcheetah`
- `hopper`
- `humanoid` / `humanoid_simplfied`
- `humanoidstandup`
- `inverted_pendulum`
- `inverted_double_pendulum`
- `pusher`
- `reacher`
- `walker2d`