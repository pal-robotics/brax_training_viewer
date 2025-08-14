# === 1. Imports and System Path Setup ===

import time
import functools
import sys
import os
import csv

# This allows the script to find custom modules (like braxviewer)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
from brax import envs
from braxviewer.WebViewer import WebViewer
from braxviewer.brax.training.agents.ppo import train as ppo
from braxviewer.brax.training.agents.sac import train as sac


# === 2. Experiment Configurations ===

# Define the different workloads you want to test.
# We will vary `num_envs` as a proxy for workload size.
# Each dictionary represents one full experiment (a "with" and "without" run).
EXPERIMENT_CONFIGS = [
    {
        'name': 'Small Workload',
        'num_envs': 512,
        'batch_size': 256,
        'num_minibatches': 2,
    },
    {
        'name': 'Medium Workload',
        'num_envs': 1024,
        'batch_size': 512,
        'num_minibatches': 2,
    },
    {
        'name': 'Large Workload',
        'num_envs': 2048,
        'batch_size': 1024,
        'num_minibatches': 2,
    },
    {
        'name': 'Extra Large Workload',
        'num_envs': 4096,
        'batch_size': 2048,
        'num_minibatches': 2,
    }
]


# === 3. Environment and Helper Functions ===

# Environment setup (remains the same)
env_name = 'humanoid'
backend = 'positional'
env = envs.get_environment(env_name=env_name, backend=backend)

# Progress function (remains the same)
def progress(num_steps, metrics):
  """A simple callback to show that training is making progress."""
  print(f'    ...at num_steps: {num_steps}')


# === 4. Main Experiment Execution Logic ===

def run_all_experiments():
    """
    Iterates through all defined experiment configurations, runs the
    viewer-on/off comparison for each, and saves the results.
    """
    output_filename = 'training_times_multiple.csv'
    print(f"Starting {len(EXPERIMENT_CONFIGS)} experiments. Results will be saved to '{output_filename}'...")

    # Check if the file already exists to decide whether to write the header
    file_exists = os.path.exists(output_filename)
    
    # Open the file in append mode ONCE, and pass the writer object down.
    with open(output_filename, mode='a', newline='') as csv_file:
        # Define the new, more detailed fieldnames for the CSV
        fieldnames = ['experiment_name', 'num_envs', 'batch_size', 'condition', 'time_seconds']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # If the file is new, write the header row
        if not file_exists:
            writer.writeheader()

        # Loop through each configuration
        for i, config in enumerate(EXPERIMENT_CONFIGS):
            print("\n" + "="*80)
            print(f"EXPERIMENT {i+1}/{len(EXPERIMENT_CONFIGS)}: {config['name']}")
            print(f"PARAMETERS: num_envs={config['num_envs']}, batch_size={config['batch_size']}")
            print("="*80)

            # --- Dynamically create the training function for THIS experiment ---
            # This is the key change: we configure the training function inside the loop
            # using the parameters from the current 'config' dictionary.
            train_fn = functools.partial(
                ppo.train,
                num_timesteps=50_000_000,
                num_evals=10,
                reward_scaling=0.1,
                episode_length=1000,
                normalize_observations=True,
                action_repeat=1,
                unroll_length=10,
                num_updates_per_batch=8,
                discounting=0.97,
                learning_rate=3e-4,
                entropy_cost=1e-3,
                seed=1,
                # Parameters from our config dictionary
                num_envs=config['num_envs'],
                batch_size=config['batch_size'],
                num_minibatches=config['num_minibatches']
            )

            # -- Run 1: Train WITH the viewer --
            print("\n--- Starting training WITH viewer ---")
            viewer = WebViewer()
            viewer.run()
            viewer.init(env)
            start_time_with_viewer = time.time()
            train_fn(environment=env, progress_fn=progress, viewer=viewer)
            end_time_with_viewer = time.time()
            viewer.stop()
            elapsed_time_with_viewer = end_time_with_viewer - start_time_with_viewer
            print(f"--- Finished training WITH viewer in {elapsed_time_with_viewer:.2f} seconds ---\n")

            # -- Run 2: Train WITHOUT the viewer --
            print("--- Starting training WITHOUT viewer ---")
            start_time_no_viewer = time.time()
            train_fn(environment=env, progress_fn=progress)
            end_time_no_viewer = time.time()
            elapsed_time_no_viewer = end_time_no_viewer - start_time_no_viewer
            print(f"--- Finished training WITHOUT viewer in {elapsed_time_no_viewer:.2f} seconds ---")

            # -- Save results for the current experiment --
            print("\n--- Saving results for this experiment ---")
            writer.writerow({
                'experiment_name': config['name'],
                'num_envs': config['num_envs'],
                'batch_size': config['batch_size'],
                'condition': 'with_viewer',
                'time_seconds': f'{elapsed_time_with_viewer:.4f}'
            })
            writer.writerow({
                'experiment_name': config['name'],
                'num_envs': config['num_envs'],
                'batch_size': config['batch_size'],
                'condition': 'without_viewer',
                'time_seconds': f'{elapsed_time_no_viewer:.4f}'
            })
            print("--- Results saved. ---")
            
    print("\n" + "="*80)
    print("All experiments complete.")
    print("="*80)

# === 5. Script Entry Point ===
if __name__ == "__main__":
    run_all_experiments()