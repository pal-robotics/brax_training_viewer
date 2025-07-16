"""
Brax Training Speed Analysis Script
==================================

This script benchmarks the training speed of Brax PPO under different rendering conditions and hyperparameter settings.
It measures the time for three scenarios:
  1. Viewer created and rendering ON
  2. Viewer created but rendering OFF
  3. No viewer (pure Brax training)

It supports large-scale batched environments and produces CSV, TXT, and figure outputs for performance analysis.

---

## Usage

1. **Install Python dependencies**

   This script requires Python 3.8+ and the following packages:

   - jax
   - numpy
   - pandas
   - matplotlib
   - etils
   - fastapi
   - uvicorn
   - (and their dependencies)

   You can install them via pip:

   ```bash
   pip install jax numpy pandas matplotlib etils fastapi uvicorn
   ```

2. **Install Brax and BraxViewer**

   - If you use the official Brax:
     ```bash
     pip install brax
     ```
   - If you use a custom/forked Brax (with rendering callback support), install your local version:
     ```bash
     pip install -e braxviewer/brax
     ```
   - For BraxViewer (this repo):
     ```bash
     pip install -e .
     ```
     (run this in the root of your brax_training_viewer repo)

3. **Run the script**

   ```bash
   python demo/analysis_speed.py
   ```

   The script will run multiple experiments and save results to:
   - `training_times_analysis.csv` (raw data)
   - `training_times_analysis.txt` (human-readable summary)
   - `training_times_analysis.png`/`.pdf` (figures)

4. **View the Web Viewer**

   During training, open your browser to the port shown in the console (default: 8000, 8001, etc.) to view the 3D rendering.


"""
# === 1. Imports and System Path Setup ===

import time
import functools
import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This allows the script to find custom modules (like braxviewer)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from brax import envs
from braxviewer.WebViewerBatched import WebViewerBatched
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.envs.wrappers.training import VmapWrapper, EpisodeWrapper, AutoResetWrapper
from brax.training.agents.ppo import networks as ppo_networks


# === 2. Experiment Configurations ===

# Define the different workloads you want to test.
# We will vary multiple hyperparameters to test different scenarios.
# For testing, you can use a smaller subset by uncommenting the TEST_CONFIGS below
EXPERIMENT_CONFIGS = [
    # Small workloads
    {
        'name': 'Small Workload - Low Batch',
        'num_envs': 512,
        'batch_size': 256,
        'num_minibatches': 2,
        'unroll_length': 10,
        'num_updates_per_batch': 8,
    },
    {
        'name': 'Small Workload - High Batch',
        'num_envs': 512,
        'batch_size': 512,
        'num_minibatches': 4,
        'unroll_length': 15,
        'num_updates_per_batch': 10,
    },
    # Medium workloads
    {
        'name': 'Medium Workload - Low Batch',
        'num_envs': 1024,
        'batch_size': 512,
        'num_minibatches': 2,
        'unroll_length': 10,
        'num_updates_per_batch': 8,
    },
    {
        'name': 'Medium Workload - High Batch',
        'num_envs': 1024,
        'batch_size': 1024,
        'num_minibatches': 4,
        'unroll_length': 15,
        'num_updates_per_batch': 10,
    },
    # Large workloads
    {
        'name': 'Large Workload - Low Batch',
        'num_envs': 2048,
        'batch_size': 1024,
        'num_minibatches': 2,
        'unroll_length': 10,
        'num_updates_per_batch': 8,
    },
    {
        'name': 'Large Workload - High Batch',
        'num_envs': 2048,
        'batch_size': 2048,
        'num_minibatches': 4,
        'unroll_length': 15,
        'num_updates_per_batch': 10,
    },
    # Extra large workloads
    {
        'name': 'Extra Large Workload - Low Batch',
        'num_envs': 4096,
        'batch_size': 2048,
        'num_minibatches': 2,
        'unroll_length': 10,
        'num_updates_per_batch': 8,
    },
    {
        'name': 'Extra Large Workload - High Batch',
        'num_envs': 4096,
        'batch_size': 4096,
        'num_minibatches': 4,
        'unroll_length': 15,
        'num_updates_per_batch': 10,
    },
    # Extreme workloads
    {
        'name': 'Extreme Workload - Low Batch',
        'num_envs': 8192,
        'batch_size': 4096,
        'num_minibatches': 2,
        'unroll_length': 10,
        'num_updates_per_batch': 8,
    },
    {
        'name': 'Extreme Workload - High Batch',
        'num_envs': 8192,
        'batch_size': 8192,
        'num_minibatches': 4,
        'unroll_length': 15,
        'num_updates_per_batch': 10,
    }
]

# Uncomment the line below for quick testing with just one small experiment
# EXPERIMENT_CONFIGS = [EXPERIMENT_CONFIGS[0]]  # Only run the first experiment

# For debugging: uncomment to test with minimal parameters
# EXPERIMENT_CONFIGS = [{
#     'name': 'Debug Test',
#     'num_envs': 64,  # Very small for testing
#     'batch_size': 32,
#     'num_minibatches': 2,
#     'unroll_length': 5,
#     'num_updates_per_batch': 4,
# }]


# === 3. Environment and Helper Functions ===

# Environment setup (remains the same)
env_name = 'humanoid'
backend = 'mjx'
env = envs.get_environment(env_name=env_name, backend=backend)

# Get XML asset path for the environment
from etils import epath
xml_asset_paths = {
    'ant': 'brax/envs/assets/ant.xml',
    'halfcheetah': 'brax/envs/assets/half_cheetah.xml',
    'hopper': 'brax/envs/assets/hopper.xml',
    'humanoid': 'brax/envs/assets/humanoid.xml',
    'humanoidstandup': 'brax/envs/assets/humanoidstandup.xml',
    'inverted_double_pendulum': 'brax/envs/assets/inverted_double_pendulum.xml',
    'inverted_pendulum': 'brax/envs/assets/inverted_pendulum.xml',
    'pusher': 'brax/envs/assets/pusher.xml',
    'reacher': 'brax/envs/assets/reacher.xml',
    'swimmer': 'brax/envs/assets/swimmer.xml',
    'walker2d': 'brax/envs/assets/walker2d.xml',
}
xml_path = xml_asset_paths[env_name]
xml_string = epath.resource_path('brax') / f'envs/assets/{env_name}.xml'
xml_string = xml_string.read_text()

def make_render_fn(viewer, num_envs):
    def render_fn(state):
        if not viewer.rendering_enabled:
            return
        # Batched: send each env's state
        for i in range(num_envs):
            single_state = jax.tree_util.tree_map(lambda x: x[i], state)
            viewer.send_frame(single_state)
    return render_fn

# Progress function (remains the same)
def progress(num_steps, metrics):
  """A simple callback to show that training is making progress."""
  print(f'    ...at num_steps: {num_steps}')

def progress_fn(current_step, metrics):
    if current_step > 0:
        print(f'Training Step: {current_step} \t Eval Reward: {metrics.get("eval/episode_reward", 0):.3f}')

make_networks_factory = ppo_networks.make_ppo_networks

def custom_wrap_env(env, episode_length=1000, action_repeat=1, randomization_fn=None):
    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env

def save_text_results(results_data, output_filename):
    """Save detailed results to a text file."""
    with open(output_filename, 'w') as f:
        f.write("BRAX TRAINING SPEED ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Group by experiment
        for experiment_name in sorted(set([r['experiment_name'] for r in results_data])):
            f.write(f"EXPERIMENT: {experiment_name}\n")
            f.write("-" * 30 + "\n")
            
            exp_results = [r for r in results_data if r['experiment_name'] == experiment_name]
            exp_results.sort(key=lambda x: x['scenario'])
            
            for result in exp_results:
                f.write(f"  {result['scenario']}: {result['time_seconds']}s\n")
            
            # Calculate overheads
            times = {r['scenario']: float(r['time_seconds']) for r in exp_results}
            if len(times) == 3:
                rendering_overhead = times['viewer_rendering_on'] - times['viewer_rendering_off']
                viewer_overhead = times['viewer_rendering_off'] - times['no_viewer']
                total_overhead = times['viewer_rendering_on'] - times['no_viewer']
                
                f.write(f"  Rendering overhead: {rendering_overhead:.2f}s\n")
                f.write(f"  Viewer overhead: {viewer_overhead:.2f}s\n")
                f.write(f"  Total overhead: {total_overhead:.2f}s\n")
                f.write(f"  Rendering overhead %: {(rendering_overhead/times['no_viewer']*100):.1f}%\n")
                f.write(f"  Total overhead %: {(total_overhead/times['no_viewer']*100):.1f}%\n")
            
            f.write("\n")
        
        # Summary statistics
        f.write("\nSUMMARY STATISTICS\n")
        f.write("=" * 20 + "\n")
        
        # Calculate averages across all experiments
        all_times = {scenario: [] for scenario in ['viewer_rendering_on', 'viewer_rendering_off', 'no_viewer']}
        for result in results_data:
            all_times[result['scenario']].append(float(result['time_seconds']))
        
        for scenario, times in all_times.items():
            if times:
                f.write(f"{scenario} - Avg: {np.mean(times):.2f}s, Min: {np.min(times):.2f}s, Max: {np.max(times):.2f}s\n")
        
        # Calculate average overheads
        avg_rendering_overhead = np.mean([float(r['time_seconds']) for r in results_data if r['scenario'] == 'viewer_rendering_on']) - \
                               np.mean([float(r['time_seconds']) for r in results_data if r['scenario'] == 'viewer_rendering_off'])
        avg_total_overhead = np.mean([float(r['time_seconds']) for r in results_data if r['scenario'] == 'viewer_rendering_on']) - \
                           np.mean([float(r['time_seconds']) for r in results_data if r['scenario'] == 'no_viewer'])
        
        f.write(f"\nAverage rendering overhead: {avg_rendering_overhead:.2f}s\n")
        f.write(f"Average total overhead: {avg_total_overhead:.2f}s\n")


def generate_figures(results_data, output_prefix):
    """Generate figures showing performance comparisons."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results_data)
    df['time_seconds'] = df['time_seconds'].astype(float)
    df['num_envs'] = df['num_envs'].astype(int)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Brax Training Speed Analysis: Rendering Impact', fontsize=16, fontweight='bold')
    
    # 1. Training time by scenario across all experiments
    ax1 = axes[0, 0]
    scenarios = ['no_viewer', 'viewer_rendering_off', 'viewer_rendering_on']
    colors = ['green', 'orange', 'red']
    
    for i, scenario in enumerate(scenarios):
        scenario_data = df[df['scenario'] == scenario]
        ax1.scatter(scenario_data['num_envs'], scenario_data['time_seconds'], 
                   label=scenario.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.7, s=50)
    
    ax1.set_xlabel('Number of Environments')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time vs Number of Environments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Overhead comparison
    ax2 = axes[0, 1]
    experiments = df['experiment_name'].unique()
    rendering_overheads = []
    viewer_overheads = []
    total_overheads = []
    
    for exp in experiments:
        exp_data = df[df['experiment_name'] == exp]
        if len(exp_data) == 3:
            times = {row['scenario']: row['time_seconds'] for _, row in exp_data.iterrows()}
            rendering_overheads.append(times['viewer_rendering_on'] - times['viewer_rendering_off'])
            viewer_overheads.append(times['viewer_rendering_off'] - times['no_viewer'])
            total_overheads.append(times['viewer_rendering_on'] - times['no_viewer'])
    
    x_pos = np.arange(len(experiments))
    width = 0.25
    
    ax2.bar(x_pos - width, viewer_overheads, width, label='Viewer Overhead', color='orange', alpha=0.8)
    ax2.bar(x_pos, rendering_overheads, width, label='Rendering Overhead', color='red', alpha=0.8)
    ax2.bar(x_pos + width, total_overheads, width, label='Total Overhead', color='purple', alpha=0.8)
    
    ax2.set_xlabel('Experiments')
    ax2.set_ylabel('Overhead (seconds)')
    ax2.set_title('Performance Overhead Breakdown')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([exp.split(' - ')[0] for exp in experiments], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Overhead percentage
    ax3 = axes[1, 0]
    overhead_percentages = []
    for exp in experiments:
        exp_data = df[df['experiment_name'] == exp]
        if len(exp_data) == 3:
            times = {row['scenario']: row['time_seconds'] for _, row in exp_data.iterrows()}
            total_overhead_pct = (times['viewer_rendering_on'] - times['no_viewer']) / times['no_viewer'] * 100
            overhead_percentages.append(total_overhead_pct)
    
    ax3.bar(range(len(experiments)), overhead_percentages, color='purple', alpha=0.8)
    ax3.set_xlabel('Experiments')
    ax3.set_ylabel('Total Overhead (%)')
    ax3.set_title('Total Overhead as Percentage of Baseline')
    ax3.set_xticks(range(len(experiments)))
    ax3.set_xticklabels([exp.split(' - ')[0] for exp in experiments], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance vs batch size
    ax4 = axes[1, 1]
    batch_sizes = df['batch_size'].unique()
    avg_times_by_batch = {}
    
    for scenario in scenarios:
        avg_times_by_batch[scenario] = []
        for batch_size in sorted(batch_sizes):
            scenario_batch_data = df[(df['scenario'] == scenario) & (df['batch_size'] == batch_size)]
            if len(scenario_batch_data) > 0:
                avg_times_by_batch[scenario].append(scenario_batch_data['time_seconds'].mean())
            else:
                avg_times_by_batch[scenario].append(0)
    
    for i, scenario in enumerate(scenarios):
        if any(avg_times_by_batch[scenario]):
            ax4.plot(sorted(batch_sizes), avg_times_by_batch[scenario], 
                    marker='o', label=scenario.replace('_', ' ').title(), 
                    color=colors[i], linewidth=2, markersize=6)
    
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Average Training Time (seconds)')
    ax4.set_title('Training Time vs Batch Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Figures saved as {output_prefix}_analysis.png and {output_prefix}_analysis.pdf")


# === 4. Main Experiment Execution Logic ===

def run_all_experiments():
    """
    Iterates through all defined experiment configurations, runs the
    three rendering scenarios for each, and saves the results.
    """
    output_filename = 'training_times_analysis.csv'
    output_txt_filename = 'training_times_analysis.txt'
    output_prefix = 'training_times_analysis'
    
    print(f"Starting {len(EXPERIMENT_CONFIGS)} experiments with 3 scenarios each.")
    print(f"Results will be saved to '{output_filename}', '{output_txt_filename}', and figures.")

    # Check if the file already exists to decide whether to write the header
    file_exists = os.path.exists(output_filename)
    
    # Collect all results for text file and figures
    all_results = []
    
    # Open the file in append mode ONCE, and pass the writer object down.
    with open(output_filename, mode='a', newline='') as csv_file:
        # Define the new, more detailed fieldnames for the CSV
        fieldnames = [
            'experiment_name', 'num_envs', 'batch_size', 'num_minibatches', 
            'unroll_length', 'num_updates_per_batch', 'scenario', 'time_seconds'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # If the file is new, write the header row
        if not file_exists:
            writer.writeheader()

        # Loop through each configuration
        for i, config in enumerate(EXPERIMENT_CONFIGS):
            print("\n" + "="*80)
            print(f"EXPERIMENT {i+1}/{len(EXPERIMENT_CONFIGS)}: {config['name']}")
            print(f"PARAMETERS: num_envs={config['num_envs']}, batch_size={config['batch_size']}")
            print(f"           num_minibatches={config['num_minibatches']}, unroll_length={config['unroll_length']}")
            print(f"           num_updates_per_batch={config['num_updates_per_batch']}")
            print("="*80)

            # -- Scenario 1: Train WITH viewer and rendering ON --
            print("\n--- Scenario 1: Training WITH viewer (rendering ON) ---")
            viewer = WebViewerBatched(num_envs=config['num_envs'], xml=xml_string, port=8000 + i*3)
            viewer.run(wait_for_startup=1)  # Reduce wait time
            print(f"Viewer started on port {8000 + i*3}")
            start_time_with_viewer = time.time()
            ppo.train(
                environment=env,
                eval_env=env,
                num_timesteps=50_000_000,
                num_evals=10,
                episode_length=1000,
                num_envs=config['num_envs'],
                num_eval_envs=config['num_envs'],
                batch_size=config['batch_size'],
                num_minibatches=config['num_minibatches'],
                unroll_length=config['unroll_length'],
                num_updates_per_batch=config['num_updates_per_batch'],
                normalize_observations=True,
                discounting=0.97,
                learning_rate=3e-4,
                entropy_cost=1e-3,
                network_factory=make_networks_factory,
                seed=1,
                wrap_env=True,
                wrap_env_fn=custom_wrap_env,
                render_fn=make_render_fn(viewer, config['num_envs']),
                should_render=jnp.array(True, dtype=jnp.bool_),
                progress_fn=progress_fn,
            )
            end_time_with_viewer = time.time()
            viewer.stop()
            elapsed_time_with_viewer = end_time_with_viewer - start_time_with_viewer
            print(f"--- Finished training WITH viewer in {elapsed_time_with_viewer:.2f} seconds ---")

            # -- Scenario 2: Train WITH viewer but rendering OFF --
            print("\n--- Scenario 2: Training WITH viewer (rendering OFF) ---")
            viewer_no_render = WebViewerBatched(num_envs=config['num_envs'], xml=xml_string, port=8001 + i*3)
            viewer_no_render.run(wait_for_startup=1)  # Reduce wait time
            print(f"Viewer started on port {8001 + i*3}")
            viewer_no_render.rendering_enabled = False  # Disable rendering
            start_time_with_viewer_no_render = time.time()
            ppo.train(
                environment=env,
                eval_env=env,
                num_timesteps=50_000_000,
                num_evals=10,
                episode_length=1000,
                num_envs=config['num_envs'],
                num_eval_envs=config['num_envs'],
                batch_size=config['batch_size'],
                num_minibatches=config['num_minibatches'],
                unroll_length=config['unroll_length'],
                num_updates_per_batch=config['num_updates_per_batch'],
                normalize_observations=True,
                discounting=0.97,
                learning_rate=3e-4,
                entropy_cost=1e-3,
                network_factory=make_networks_factory,
                seed=1,
                wrap_env=True,
                wrap_env_fn=custom_wrap_env,
                render_fn=make_render_fn(viewer_no_render, config['num_envs']),
                should_render=jnp.array(False, dtype=jnp.bool_),
                progress_fn=progress_fn,
            )
            end_time_with_viewer_no_render = time.time()
            viewer_no_render.stop()
            elapsed_time_with_viewer_no_render = end_time_with_viewer_no_render - start_time_with_viewer_no_render
            print(f"--- Finished training WITH viewer (no render) in {elapsed_time_with_viewer_no_render:.2f} seconds ---")

            # -- Scenario 3: Train WITHOUT viewer (normal Brax training) --
            print("\n--- Scenario 3: Training WITHOUT viewer (normal Brax) ---")
            start_time_no_viewer = time.time()
            ppo.train(
                environment=env,
                eval_env=env,
                num_timesteps=50_000_000,
                num_evals=10,
                episode_length=1000,
                num_envs=config['num_envs'],
                num_eval_envs=config['num_envs'],
                batch_size=config['batch_size'],
                num_minibatches=config['num_minibatches'],
                unroll_length=config['unroll_length'],
                num_updates_per_batch=config['num_updates_per_batch'],
                normalize_observations=True,
                discounting=0.97,
                learning_rate=3e-4,
                entropy_cost=1e-3,
                network_factory=make_networks_factory,
                seed=1,
                wrap_env=True,
                wrap_env_fn=custom_wrap_env,
                render_fn=None,
                should_render=jnp.array(False, dtype=jnp.bool_),
                progress_fn=progress_fn,
            )
            end_time_no_viewer = time.time()
            elapsed_time_no_viewer = end_time_no_viewer - start_time_no_viewer
            print(f"--- Finished training WITHOUT viewer in {elapsed_time_no_viewer:.2f} seconds ---")

            # -- Save results for the current experiment --
            print("\n--- Saving results for this experiment ---")
            
            # Prepare result data
            result_data = {
                'experiment_name': config['name'],
                'num_envs': config['num_envs'],
                'batch_size': config['batch_size'],
                'num_minibatches': config['num_minibatches'],
                'unroll_length': config['unroll_length'],
                'num_updates_per_batch': config['num_updates_per_batch'],
            }
            
            # Save scenario 1 results
            scenario1_result = result_data.copy()
            scenario1_result.update({
                'scenario': 'viewer_rendering_on',
                'time_seconds': f'{elapsed_time_with_viewer:.4f}'
            })
            writer.writerow(scenario1_result)
            all_results.append(scenario1_result)
            
            # Save scenario 2 results
            scenario2_result = result_data.copy()
            scenario2_result.update({
                'scenario': 'viewer_rendering_off',
                'time_seconds': f'{elapsed_time_with_viewer_no_render:.4f}'
            })
            writer.writerow(scenario2_result)
            all_results.append(scenario2_result)
            
            # Save scenario 3 results
            scenario3_result = result_data.copy()
            scenario3_result.update({
                'scenario': 'no_viewer',
                'time_seconds': f'{elapsed_time_no_viewer:.4f}'
            })
            writer.writerow(scenario3_result)
            all_results.append(scenario3_result)
            
            print("--- Results saved. ---")
            
            # Print summary for this experiment
            print(f"\n--- Summary for {config['name']} ---")
            print(f"  Viewer + Rendering ON:  {elapsed_time_with_viewer:.2f}s")
            print(f"  Viewer + Rendering OFF: {elapsed_time_with_viewer_no_render:.2f}s")
            print(f"  No Viewer:               {elapsed_time_no_viewer:.2f}s")
            print(f"  Rendering overhead:     {elapsed_time_with_viewer - elapsed_time_with_viewer_no_render:.2f}s")
            print(f"  Viewer overhead:        {elapsed_time_with_viewer_no_render - elapsed_time_no_viewer:.2f}s")
            print(f"  Total overhead:         {elapsed_time_with_viewer - elapsed_time_no_viewer:.2f}s")
    
    # Save text results and generate figures
    print("\n" + "="*80)
    print("Generating analysis files and figures...")
    print("="*80)
    
    # Save detailed text results
    save_text_results(all_results, output_txt_filename)
    print(f"Text results saved to {output_txt_filename}")
    
    # Generate figures
    generate_figures(all_results, output_prefix)
    
    print("\n" + "="*80)
    print("All experiments complete.")
    print("="*80)

# === 5. Script Entry Point ===
if __name__ == "__main__":
    run_all_experiments()