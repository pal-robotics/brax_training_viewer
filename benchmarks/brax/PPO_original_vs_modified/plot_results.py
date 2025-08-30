
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import glob
import os
from collections import defaultdict

PPO_COLORS = {'original': '#1f77b4', 'viewer': '#ff7f0e'}

def plot_combined_results(results_by_hardware):
    """Plots a consolidated view of all hardware results."""
    all_labels = set()
    for __, results in results_by_hardware.items():
        for _, data in results.items():
            for d in data:
                all_labels.add(d['params'])
    labels = sorted(list(all_labels))
    hardware_names = sorted(list(results_by_hardware.keys()))

    if not labels or not hardware_names:
        print("Not enough data for consolidated plot.")
        return

    plot_data = defaultdict(lambda: defaultdict(list))
    plot_stds = defaultdict(lambda: defaultdict(list))

    for hardware in hardware_names:
        for ppo_type in ['original', 'viewer']:
            data = results_by_hardware[hardware].get(ppo_type, [])
            times_by_label = {d['params']: d['times'] for d in data}
            
            means = [np.mean(times_by_label.get(label, [0])) for label in labels]
            stds = [np.std(times_by_label.get(label, [0])) for label in labels]
            
            plot_data[hardware][ppo_type] = means
            plot_stds[hardware][ppo_type] = stds
    
    x = np.arange(len(labels))
    n_hardware = len(hardware_names)
    n_ppo_types = 2  # original, viewer
    
    total_width = 0.8
    bar_width = total_width / (n_hardware * n_ppo_types)
    
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, hardware in enumerate(hardware_names):
        for j, ppo_type in enumerate(['original', 'viewer']):
            offset = (i * n_ppo_types + j) - (n_hardware * n_ppo_types) / 2.0
            position = x + (offset + 0.5) * bar_width
            
            means = plot_data[hardware][ppo_type]
            stds = plot_stds[hardware][ppo_type]
            
            rects = ax.bar(position, means, bar_width, yerr=stds, capsize=3, 
                           color=PPO_COLORS[ppo_type], hatch='///')
            ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    num_runs = 0
    for hardware in hardware_names:
        for ppo_type in ['original', 'viewer']:
            data = results_by_hardware[hardware].get(ppo_type, [])
            if data and data[0].get('times'):
                num_runs = len(data[0]['times'])
                break
        if num_runs > 0:
            break

    ax.set_ylabel('Time (s)')
    ax.set_title(f'PPO Performance Comparison (Avg of {num_runs} runs)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    legend_handles = []
    for hardware in hardware_names:
        legend_handles.append(mpatches.Patch(color=PPO_COLORS['original'], label=hardware.replace("_", " ")))
    
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Viewer PPO (OFF)'))
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', label='Original PPO'))
    
    ax.legend(handles=legend_handles, fontsize='small')
    
    fig.tight_layout()
    output_filename = 'benchmarks/brax/PPO_original_vs_modified/ppo_comparison_consolidated.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_filename}")


def plot_consolidated_results(results_by_hardware):
    """Plots a consolidated view of all hardware results."""
    all_labels = set()
    for __, results in results_by_hardware.items():
        for _, data in results.items():
            for d in data:
                all_labels.add(d['params'])
    labels = sorted(list(all_labels))
    hardware_names = sorted(list(results_by_hardware.keys()))

    if not labels or not hardware_names:
        print("Not enough data for consolidated plot.")
        return

    hardware_colors = {name: f'C{i}' for i, name in enumerate(hardware_names)}
    ppo_hatches = {'original': None, 'viewer': '///'}
    
    plot_data = defaultdict(lambda: defaultdict(list))
    plot_stds = defaultdict(lambda: defaultdict(list))

    for hardware in hardware_names:
        for ppo_type in ['original', 'viewer']:
            data = results_by_hardware[hardware].get(ppo_type, [])
            times_by_label = {d['params']: d['times'] for d in data}
            
            means = [np.mean(times_by_label.get(label, [0])) for label in labels]
            stds = [np.std(times_by_label.get(label, [0])) for label in labels]
            
            plot_data[hardware][ppo_type] = means
            plot_stds[hardware][ppo_type] = stds
    
    x = np.arange(len(labels))
    n_hardware = len(hardware_names)
    n_ppo_types = 2  # original, viewer
    
    total_width = 0.8
    bar_width = total_width / (n_hardware * n_ppo_types)
    
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, hardware in enumerate(hardware_names):
        for j, ppo_type in enumerate(['original', 'viewer']):
            offset = (i * n_ppo_types + j) - (n_hardware * n_ppo_types) / 2.0
            position = x + (offset + 0.5) * bar_width
            
            means = plot_data[hardware][ppo_type]
            stds = plot_stds[hardware][ppo_type]
            
            rects = ax.bar(position, means, bar_width, yerr=stds, capsize=3, 
                           color=hardware_colors[hardware], hatch=ppo_hatches[ppo_type])
            ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    num_runs = 0
    for hardware in hardware_names:
        for ppo_type in ['original', 'viewer']:
            data = results_by_hardware[hardware].get(ppo_type, [])
            if data and data[0].get('times'):
                num_runs = len(data[0]['times'])
                break
        if num_runs > 0:
            break

    ax.set_ylabel('Time (s)')
    ax.set_title(f'PPO Performance Comparison (Avg of {num_runs} runs)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    legend_handles = []
    for hardware in hardware_names:
        legend_handles.append(mpatches.Patch(color=hardware_colors[hardware], label=hardware.replace("_", " ")))
    
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch=ppo_hatches['viewer'], label='Viewer PPO (OFF)'))
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', label='Original PPO'))
    
    ax.legend(handles=legend_handles, fontsize='small')
    
    fig.tight_layout()
    output_filename = 'benchmarks/brax/PPO_original_vs_modified/ppo_comparison_consolidated.png'
    plt.savefig(output_filename)
    print(f"\nConsolidated plot saved to {output_filename}")


def plot_results():
    search_path = 'benchmarks/brax/PPO_original_vs_modified/*_ppo_results_*.json'
    result_files = glob.glob(search_path)
    
    if not result_files:
        print(f"No result files found at '{search_path}'. Run the training scripts first.")
        return

    results_by_hardware = defaultdict(dict)
    
    for f in result_files:
        basename = os.path.basename(f)
        parts = basename.replace('.json', '').split('_ppo_results_')
        if len(parts) != 2:
            continue
            
        ppo_type, hardware_name = parts
        
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        results_by_hardware[hardware_name][ppo_type] = data

    for hardware, results in results_by_hardware.items():
        print(f"Plotting results for hardware: {hardware}")
        
        viewer_data = results.get('viewer', [])
        original_data = results.get('original', [])

        if not viewer_data or not original_data:
            print(f"  Skipping {hardware}, missing viewer or original data.")
            continue

        labels = sorted(list(set([d['params'] for d in viewer_data] + [d['params'] for d in original_data])))
        
        viewer_times = {d['params']: d['times'] for d in viewer_data}
        original_times = {d['params']: d['times'] for d in original_data}
        
        viewer_means = [np.mean(viewer_times.get(label, [0])) for label in labels]
        original_means = [np.mean(original_times.get(label, [0])) for label in labels]
        
        viewer_stds = [np.std(viewer_times.get(label, [0])) for label in labels]
        original_stds = [np.std(original_times.get(label, [0])) for label in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, original_means, width, yerr=original_stds, capsize=5, label='Original PPO', color=PPO_COLORS['original'])
        rects2 = ax.bar(x + width/2, viewer_means, width, yerr=viewer_stds, capsize=5, label='Viewer PPO (OFF)', color=PPO_COLORS['viewer'])

        ax.set_ylabel('Time (s)')
        
        num_runs = len(viewer_times.get(labels[0], []))
        ax.set_title(f'PPO Performance on {hardware.replace("_", " ")} (Avg of {num_runs} runs)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')

        fig.tight_layout()
        output_filename = f'benchmarks/brax/PPO_original_vs_modified/ppo_comparison_{hardware}.png'
        plt.savefig(output_filename)
        print(f"  Plot saved to {output_filename}")
    
    # Create combined plot
    print("Creating combined plot...")
    plot_combined_results(results_by_hardware)

    if results_by_hardware:
        plot_consolidated_results(results_by_hardware)

if __name__ == '__main__':
    plot_results()
