import time
import random
import logging
import numpy as np
from utils import loadj, dumpj
from pathlib import Path
from .plot_functions import output_dir, ckpt_dir, line_plot_with_error, make_legend, rainbow_bar_plot_with_error
from arguments import Breeding_Types, Baseline_Methods
from debug import *

plot_colors = ['#FFD92F', '#FF7F0E', '#1770af', '#ADD8E6', '#BCBD22', '#D62728']
themarkers = ['X', '^', 'o', 'P', 's', '*']
Baseline_Methods = ['Greedy', 'Group', 'NCF', 'LightGCN', 'HetRecSys', 'BANTER']
taglist = ['revenue', 'buyer_utility', 'runtime']
yaxis_list = ['Revenue', 'Avg. Utility', 'Runtime (s)']


def print_scale():
    print_yelp()
    print_large()

def print_yelp():
    out_dir = output_dir/'scale/yelp'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = ckpt_dir/'scale/yelp'
    
    for nft_project_name in ['yelp']:
        if not any(nft_project_name in file.name for file in result_dir.iterdir()):
            logging.info(f'skipping {nft_project_name}, no result files...')
            continue

        for tag, ylabel in zip(taglist, yaxis_list):
            filename = f'{tag}_{nft_project_name}_multi.jpg'
            filepath = out_dir/filename

            if filepath.exists():
                logging.info(f'overwriting existing {filepath}...')
            else:
                logging.info(f'rendering main plot to {filepath}...')

            results_mean, results_error = read_info_with_errors(tag, nft_project_name, result_dir)
            
            # Calculate y-axis limits accounting for error bars
            all_values_with_errors = []
            for means, errors in zip(results_mean, results_error):
                for mean, error in zip(means, errors):
                    if mean > 0:
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
                'xticks': ['Heter', 'Homo', 'Child'],
                'xtick_size': 50
            }

            rainbow_bar_plot_with_error(results_mean, results_error, infos, filepath, True)

    legend_file = out_dir/'zlegend.jpg'
    make_legend(Baseline_Methods, legend_file, 'bar', plot_colors)

def read_info_with_errors(tag, nft_project_name, result_dir):
    """Read multi-run results and calculate statistics."""
    results_mean = []
    results_error = []
    
    for breeding_type in Breeding_Types[:-1]:
        breeding_means = []
        breeding_errors = []
        
        for method in Baseline_Methods:
            result_json = result_dir/f'{nft_project_name}_{method}_{breeding_type}_results.json'
            
            if result_json.exists():
                runs = loadj(result_json)
                if runs and len(runs) > 0:
                    # Extract values from all runs
                    if tag == 'revenue':
                        values = [run['seller_revenue'] for run in runs if 'seller_revenue' in run]
                    elif tag == 'buyer_utility':
                        values = [run['avg_buyer_utility'] for run in runs if 'avg_buyer_utility' in run]
                    elif tag == 'runtime':
                        prep_time = 128 if breeding_type == 'Heterogeneous' else 204
                        values = [run['runtime'] + prep_time for run in runs if 'runtime' in run]
                    
                    # Calculate statistics
                    if values and len(values) > 0:
                        mean_val = np.mean(values)
                        std_err = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
                        breeding_means.append(mean_val)
                        breeding_errors.append(std_err)
                    else:
                        breeding_means.append(-2 + random.random())
                        breeding_errors.append(0)
                else:
                    breeding_means.append(-2 + random.random())
                    breeding_errors.append(0)
            else:
                breeding_means.append(-2 + random.random())
                breeding_errors.append(0)
        
        results_mean.append(breeding_means)
        results_error.append(breeding_errors)

    return results_mean, results_error


def print_large():
    out_dir = output_dir/'scale/large'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = ckpt_dir/'scale/large'
        
    nft_project_name = 'fatapeclub'

    for breeding_type in ['Heterogeneous']:
        for tag, ylabel in zip(taglist, yaxis_list):
            if tag == 'buyer_utility':
                continue

            filename = f'{tag}_{nft_project_name}_{breeding_type}_multi.jpg'
            filepath = out_dir/filename

            if filepath.exists():
                logging.info(f'overwriting existing {filepath}...')
            else:
                logging.info(f'rendering main plot to {filepath}...')

            results_mean, results_error = load_large_results_with_errors(
                result_dir, tag, nft_project_name, breeding_type
            )

            infos = {
                'log': False,
                'xlabel': 'Number of Buyers',
                'ylabel': ylabel,
                'colors': plot_colors,
                'markers': themarkers,
            }
            
            X = [scale*10000 for scale in range(1, 11)]
            line_plot_with_error(X, results_mean, results_error, infos, filepath)

    legend_file = out_dir/'zlegend.jpg'
    make_legend(Baseline_Methods, legend_file, 'line', plot_colors, themarkers)


def load_large_results_with_errors(result_dir, tag, nft_project_name, breeding_type):
    """Load large-scale results and calculate statistics."""
    results_mean = []
    results_error = []
    
    for method in ['Greedy', 'Group', 'NCF', 'LightGCN', 'HetRecSys', 'BANTER']:
        method_means = []
        method_errors = []
        
        for scale in range(1, 11):
            result_json = result_dir / f'{nft_project_name}_{breeding_type}_{method}_scale{scale}_results.json'
            
            if result_json.exists():
                runs = loadj(result_json)
                if runs:
                    # Extract values from all runs
                    if tag == 'revenue':
                        values = [run['seller_revenue'] for run in runs]
                    elif tag == 'buyer_utility':
                        values = [run['avg_buyer_utility'] for run in runs]
                    elif tag == 'runtime':
                        prep_time = 100
                        values = [run['runtime'] + prep_time*scale for run in runs]
                    
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