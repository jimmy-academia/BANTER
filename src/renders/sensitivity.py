import random
import logging
import numpy as np
from arguments import Breeding_Types
from utils import loadj

from .plot_functions import output_dir, ckpt_dir, line_plot_with_error, make_legend


Compare_methods = ['Greedy', 'Group', 'NCF', 'LightGCN', 'HetRecSys', 'BANTER']
plot_colors = ['#FFD92F', '#FF7F0E', '#1770af', '#ADD8E6', '#BCBD22', '#D62728']
themarkers = ['X', '^', 'o', 'P', 's', '*']


def print_sensitivity():
    """Print sensitivity analysis with error bars from multi-run data."""
    out_dir = output_dir/'sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = ckpt_dir/'sensitivity'
        
    nft_project_name = 'fatapeclub'
    N_M_infos = loadj('ckpt/N_M_infos.json')
    N = N_M_infos.get(nft_project_name).get('N')
    M = N_M_infos.get(nft_project_name).get('M')

    item_X = [int(M/10 * scale) for scale in range(1, 11)]
    num_X = [int(N/10 * scale) for scale in range(1, 11)]
    bud_X = [scale * 10 for scale in range(1, 11)]

    for tag, xlabel, X in zip(['item', 'num', 'bud'], ['Number of NFTs',  'Number of Buyers', 'Buyer Budgets (%)'], [item_X, num_X, bud_X]):
        for breeding_type in ['ChildProject']:
            results_mean, results_error = load_sensitivity_results_with_errors(
                result_dir / tag, nft_project_name, breeding_type
            )
            
            for type_, ylabel in zip(['rev', 'utils', 'time'], ['Revenue', 'Avg. Utility', 'Runtime (s)']):
                filename = f'{tag}_{nft_project_name}_{breeding_type}_{type_}_multi.jpg'
                filepath = out_dir/filename

                logging.info(f'Creating sensitivity plot: {filepath}')

                # Extract data for this metric
                if type_ == 'rev':
                    plot_means = [method_data['revenue_mean'] for method_data in results_mean]
                    plot_errors = [method_data['revenue_error'] for method_data in results_error]
                elif type_ == 'utils':
                    plot_means = [method_data['utility_mean'] for method_data in results_mean]
                    plot_errors = [method_data['utility_error'] for method_data in results_error]
                elif type_ == 'time':
                    plot_means = [method_data['runtime_mean'] for method_data in results_mean]
                    plot_errors = [method_data['runtime_error'] for method_data in results_error]

                infos = {
                    'log': False,
                    'xlabel': xlabel,
                    'ylabel': ylabel,
                    'colors': plot_colors,
                    'markers': themarkers,
                    'xmin': 0
                }
                line_plot_with_error(X, plot_means, plot_errors, infos, filepath)

    legend_file = out_dir/'zlegend.jpg'
    make_legend(Compare_methods, legend_file, 'line', plot_colors, themarkers)


def load_sensitivity_results_with_errors(result_dir, nft_project_name, breeding_type):
    """Load sensitivity results and calculate statistics for each method and scale."""
    results_mean = []
    results_error = []
    
    for method in Compare_methods:
        method_revenue_means = []
        method_revenue_errors = []
        method_utility_means = []
        method_utility_errors = []
        method_runtime_means = []
        method_runtime_errors = []
        
        for scale in range(1, 11):
            result_file = result_dir / f'{method}_{nft_project_name}_{breeding_type}_scale{scale}_results.json'
            
            if result_file.exists():
                runs = loadj(result_file)
                if runs and len(runs) > 0:
                    # Extract values from all runs
                    revenue_values = [run['revenue'] for run in runs if 'revenue' in run]
                    utility_values = [run['utility'] for run in runs if 'utility' in run]
                    runtime_values = [run['runtime'] for run in runs if 'runtime' in run]
                    
                    # Calculate statistics for each metric
                    if revenue_values:
                        method_revenue_means.append(np.mean(revenue_values))
                        method_revenue_errors.append(np.std(revenue_values) / np.sqrt(len(revenue_values)) if len(revenue_values) > 1 else 0)
                    else:
                        method_revenue_means.append(0)
                        method_revenue_errors.append(0)
                    
                    if utility_values:
                        method_utility_means.append(np.mean(utility_values))
                        method_utility_errors.append(np.std(utility_values) / np.sqrt(len(utility_values)) if len(utility_values) > 1 else 0)
                    else:
                        method_utility_means.append(0)
                        method_utility_errors.append(0)
                    
                    if runtime_values:
                        method_runtime_means.append(np.mean(runtime_values))
                        method_runtime_errors.append(np.std(runtime_values) / np.sqrt(len(runtime_values)) if len(runtime_values) > 1 else 0)
                    else:
                        method_runtime_means.append(0)
                        method_runtime_errors.append(0)
                else:
                    # No runs available
                    method_revenue_means.append(0)
                    method_revenue_errors.append(0)
                    method_utility_means.append(0)
                    method_utility_errors.append(0)
                    method_runtime_means.append(0)
                    method_runtime_errors.append(0)
            else:
                # File doesn't exist
                method_revenue_means.append(0)
                method_revenue_errors.append(0)
                method_utility_means.append(0)
                method_utility_errors.append(0)
                method_runtime_means.append(0)
                method_runtime_errors.append(0)
        
        # Store organized data for this method
        results_mean.append({
            'revenue_mean': method_revenue_means,
            'utility_mean': method_utility_means,
            'runtime_mean': method_runtime_means
        })
        
        results_error.append({
            'revenue_error': method_revenue_errors,
            'utility_error': method_utility_errors,
            'runtime_error': method_runtime_errors
        })
    
    return results_mean, results_error