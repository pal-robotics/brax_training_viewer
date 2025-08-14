# CartPole

Viewer Performance Impact Benchmark

## Abstract

This benchmark evaluates the performance impact of Brax Viewer on Brax PPO training using the CartPole environment. We measure training throughput under three scenarios: pure training, viewer backend with rendering disabled, and full real-time visualization.

## Methods

### Training Settings

- Environment: Custom `CartPole` with `mjx` backend
- Training steps: 2,000
- Episode length: 300
- Number of parallel environments: 8
- Learning rate: 3e-4
- Entropy cost: 1e-2

## Implementation

The benchmark is implemented in `benchmarks/brax/cartpole/time_test_small.py`. Running the script prints timing results for three scenarios to the console.

## Results

| Hardware | No Viewer | Viewer OFF | Viewer ON |
|----------|-----------|------------|-----------|
| Apple M1 Max (32-core GPU) | 20.63s | 18.17s | 51.54s |
| NVIDIA RTX 5090 | 51.81s | 33.53s | 63.57s |
| NVIDIA RTX A5000 | 61.21s | 47.11s | 106.81s |

:::{note}
The "No Viewer" results are slower than "Viewer OFF" because the benchmark script runs "No Viewer" first, and the initial run takes longer may due to JIT (just-in-time) compilation. 
:::