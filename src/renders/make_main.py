import logging
import numpy as np
from pathlib import Path
from utils import loadj
from .plot_functions import output_dir, ckpt_dir, rainbow_bar_plot_with_error, make_legend
from arguments import nft_project_names, plot_colors, Breeding_Types, Baseline_Methods


def print_main():
    """Create main plots from multi-run experiments with error bars."""
    out_dir = output_dir / 'main_exp'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = ckpt_dir / 'main_exp'
    
    taglist = ['revenue', 'buyer_utility', 'runtime']
    yaxis_list = ['Revenue', 'Avg. Utility', 'Runtime (s)']
    
    for nft_project_name in nft_project_names:
        project_dir = result_dir / nft_project_name
        if not project_dir.exists():
            logging.info(f'Skipping {nft_project_name}, no result directory...')
            continue

        for tag, ylabel in zip(taglist, yaxis_list):
            filename = f'{tag}_{nft_project_name}_multi.jpg'
            filepath = out_dir / filename

            if filepath.exists():
                logging.info(f'Overwriting existing {filepath}...')
            else:
                logging.info(f'Rendering main plot to {filepath}...')

            # Load results with means and errors
            results_mean, results_error = load_results_for_project(project_dir, tag)
            
            if not any(results_mean):  # Skip if no data
                logging.warning(f'No data for {tag} in {nft_project_name}')
                continue
                
            # Calculate y-axis limits accounting for error bars
            all_values_with_errors = []
            for means, errors in zip(results_mean, results_error):
                for mean, error in zip(means, errors):
                    if mean > 0:  # Only consider non-zero values
                        all_values_with_errors.extend([mean + error, mean - error])
            
            if not all_values_with_errors:
                continue
                
            y_axis_max = max(all_values_with_errors) * 1.1
            y_axis_min = max(0, min(all_values_with_errors) * 0.9)

            infos = {
                'log': False,
                'ylabel': ylabel,
                'y_axis_max': y_axis_max,
                'y_axis_min': y_axis_min,
                'colors': plot_colors,
                'xticks': Breeding_Types[:-1],
            }

            rainbow_bar_plot_with_error(results_mean, results_error, infos, filepath)

    # Create legend
    legend_file = out_dir / 'zlegend.jpg'
    make_legend(Baseline_Methods, legend_file, 'bar', plot_colors)


def load_results_for_project(project_dir, tag):
    """Load results for a project and calculate means and standard errors."""
    results_mean = []
    results_error = []
    
    for breeding_type in Breeding_Types[:-1]:  # Exclude 'None'
        breeding_means = []
        breeding_errors = []
        
        for method in Baseline_Methods:
            method_dir = project_dir / method
            result_file = method_dir / f'{breeding_type}_results.json'
            
            if result_file.exists():
                runs = loadj(result_file)
                if runs:
                    # Extract values from all runs
                    if tag == 'revenue':
                        values = [run['seller_revenue'] for run in runs]
                    elif tag == 'buyer_utility':
                        values = [run['avg_buyer_utility'] for run in runs]
                    elif tag == 'runtime':
                        values = [run['runtime'] for run in runs]
                        if method == 'Auction':
                            values = [v * 2 for v in values]
                        # Add preparation time
                        prep_time = 62 if breeding_type == 'Heterogeneous' else 102
                        values = [v + prep_time for v in values]
                    
                    # Calculate mean and standard error
                    if values:
                        mean_val = np.mean(values)
                        std_err = np.std(values) / np.sqrt(len(values))
                        breeding_means.append(mean_val)
                        breeding_errors.append(std_err)
                    else:
                        breeding_means.append(0)
                        breeding_errors.append(0)
                else:
                    breeding_means.append(0)
                    breeding_errors.append(0)
            else:
                breeding_means.append(0)
                breeding_errors.append(0)
        
        results_mean.append(breeding_means)
        results_error.append(breeding_errors)
    
    return results_mean, results_error


def load_scale_results_with_errors(result_dir, tag, nft_project_name, breeding_type):
    """Load scaling results and calculate statistics for line plots."""
    results_mean = []
    results_error = []
    
    methods = ['Greedy', 'Group', 'NCF', 'LightGCN', 'HetRecSys', 'BANTER']
    
    for method in methods:
        method_means = []
        method_errors = []
        
        for scale in range(1, 11):
            result_file = result_dir / f'{method}_{nft_project_name}_{breeding_type}_scale{scale}_results.json'
            
            if result_file.exists():
                runs = loadj(result_file)
                if runs:
                    # Extract values from all runs
                    if tag == 'revenue':
                        values = [run['revenue'] for run in runs]
                    elif tag == 'utility':
                        values = [run['utility'] for run in runs]
                    elif tag == 'runtime':
                        values = [run['runtime'] for run in runs]
                        # Add preparation time
                        prep_time = 100
                        values = [v + prep_time * scale for v in values]
                    
                    # Calculate statistics
                    if values:
                        mean_val = np.mean(values)
                        std_err = np.std(values) / np.sqrt(len(values))
                        method_means.append(mean_val)
                        method_errors.append(std_err)
                    else:
                        method_means.append(0)
                        method_errors.append(0)
                else:
                    method_means.append(0)
                    method_errors.append(0)
            else:
                method_means.append(0)
                method_errors.append(0)
        
        results_mean.append(method_means)
        results_error.append(method_errors)
    
    return results_mean, results_error