# Cartpole Example

This example demonstrates how to train a custom Cartpole environment with the separated web viewer architecture.

## Architecture

This example shows how to create a custom Brax environment and integrate it with the web viewer:

- `config.py`: Shared configuration for network settings, environment parameters, and training hyperparameters
- `train.py`: Training script that creates a custom CartPole environment and connects to the viewer
- `viewer.py`: Independent viewer server that can be run separately

### PPO import

```python
from braxviewer.brax.training.agents.ppo import train as ppo
```

## Usage

### 1. Start the Viewer Server

First, start the viewer server in a separate terminal:

```bash
python examples/brax/cartpole/viewer.py
```

This will start the WebViewer server on `http://127.0.0.1:8000`.

### 2. Run Training

In another terminal, run the training script:

```bash
python examples/brax/cartpole/train.py
```

### 3. View Training in Browser

Open your web browser and navigate to `http://127.0.0.1:8000` to see the training visualization in real-time.