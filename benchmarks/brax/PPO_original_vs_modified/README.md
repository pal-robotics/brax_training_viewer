# PPO Performance Benchmark

Measures performance overhead of the modified `braxviewer.brax` PPO vs. the original `brax` PPO.

## Run

1.  **Collect data:**
    ```bash
    python benchmarks/brax/PPO_original_vs_modified/train_viewer_ppo.py
    python benchmarks/brax/PPO_original_vs_modified/train_original_ppo.py
    ```
    This creates hardware-specific JSON files (e.g., `..._results_NVIDIA_RTX_4090.json`).

2.  **Generate plots:**
    ```bash
    python benchmarks/brax/PPO_original_vs_modified/plot_results.py
    ```
    This creates a hardware-specific PNG for each machine benchmarked (e.g., `ppo_comparison_NVIDIA_RTX_4090.png`).
