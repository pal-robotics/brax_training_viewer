
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import defaultdict

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
        rects1 = ax.bar(x - width/2, original_means, width, yerr=original_stds, capsize=5, label='Original PPO')
        rects2 = ax.bar(x + width/2, viewer_means, width, yerr=viewer_stds, capsize=5, label='Viewer PPO (OFF)')

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

if __name__ == '__main__':
    plot_results()
