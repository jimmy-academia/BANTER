
import time
import random
import logging
from utils import loadj
from pathlib import Path
from .plot_functions import output_dir, ckpt_dir, line_plot, make_legend, line_plot
from arguments import Breeding_Types, Baseline_Methods


import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "font.size": 50,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
})

out_dir = output_dir/'ablation'
out_dir.mkdir(parents=True, exist_ok=True)
result_dir = ckpt_dir/'ablation'

X = list(range(1,18,2))
use_colors = ['#D62728', '#2CA02C', '#1F77B4']
use_markers = ['*', '^', 'P']

legend = {'optimization': ['BANTER', 'BANTER (no init)', 'INIT'], 
    'schedule': ['BANTER', 'BANTER (fixed)', 'BANTER (none)'], 
    'module': ['BANTER', 'BANTER (objective)', 'BANTER (random)']}
def print_ablation():
    do_optimization_schedule()
    do_module()


def do_optimization_schedule():

    for taskname in ['optimization', 'schedule']:
        
        filepath = out_dir/f'{taskname}.jpg'

        if filepath.exists():
            logging.info(f'overwriting existing {filepath}...')
        else:
            logging.info(f'rendering main plot to {filepath}...')

        results = []
        for i in range(3):
            result_json = result_dir/f'fatapeclub_ChildProject_{taskname}{i}.json'
            rev_list = loadj(result_json).get('revenue_list')
            rev_list = [(rev_list[i]+rev_list[i-1])/2 - rev_list[1] + rev_list[0] for i in X]
            results.append(rev_list)

        infos = {
            'log': taskname == 'optimization',
            'ylabel': 'Revenue',
            'xlabel': 'iteration step',
            'colors': use_colors,
            'markers': use_markers,
            'xticks': X,
        }
        line_plot(X, results, infos, filepath)
        make_legend(legend[taskname], out_dir/f'legend_{taskname}.jpg', 'line', use_colors, markers=use_markers)


def do_module():
    nft_project_name = 'fatapeclub'

    results = []
    
    filepath = out_dir/f'module.jpg'

    if filepath.exists():
        logging.info(f'overwriting existing {filepath}...')
    else:
        logging.info(f'rendering main plot to {filepath}...')

    for _breeding in Breeding_Types[:-1]:
        values = []
        for i in range(3):
            result_json = result_dir/f'fatapeclub_{_breeding}_module{i}.json'
            value = loadj(result_json).get('avg_buyer_utility')
            values.append(value)
        results.append(values)

    y_axis_max = max(max(breeding_values) for breeding_values in results) * 1.1
    infos = {
        'figsize': (13, 4),
        'log': False,
        'ylabel': 'Avg. Utility',
        'y_axis_max': y_axis_max,
        'y_axis_min': 0,
        'colors': use_colors,
        'xticks': Breeding_Types[:-1],
    }
    rainbow_bar_plot(results, infos, filepath)

    make_legend(legend['module'], out_dir/'legend_module.jpg', 'bar', use_colors)

def rainbow_bar_plot(project_revenues, infos, filepath):
    """Creates a rainbow bar plot and saves it to a file.

    Args:
        project_revenues (list): List of project revenues.
        infos (dict): Information for the plot (e.g., colors, xticks, labels).
        filepath (str or Path): Path to save the plot.
    """
    figsize = infos['figsize'] if 'figsize' in infos else (13, 6)
    plt.figure(figsize=figsize, dpi=200)
    plt.ylabel(infos['ylabel'], fontsize=35, fontweight='bold', y=0.3)
    plt.ylim(infos['y_axis_min'], infos['y_axis_max'])

    bar_width = 1
    set_width = 1.2*len(infos['colors']) + 0.6
    indexes = range(len(project_revenues))
    for index, rev_rainbow in zip(indexes, project_revenues):
        for k, (rev, color) in enumerate(zip(rev_rainbow, infos['colors'])):
            plt.bar(index*set_width+k*(bar_width+0.2), rev, bar_width, color=color)
        if index != len(project_revenues) - 1:
            plt.axvline(x=index*set_width+(k+1)*(bar_width+0.2)-0.2, color='black', linestyle=':', linewidth=3)
    
    if infos['xticks'] is not None:
        tick_positions = [index * set_width + (len(infos['colors']) - 1) / 2 * (bar_width + 0.2) for index in range(len(project_revenues))]
        plt.xticks(tick_positions, infos['xticks'], fontsize=30)
    else:
        plt.xticks([])

    if infos['log']:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()

def line_plot(X, project_values, infos, filepath):
    figsize = infos['figsize'] if 'figsize' in infos else (13, 4)
    plt.figure(figsize=figsize, dpi=200)
    plt.ylabel(infos['ylabel'], fontsize=40, fontweight='bold', y=0.3)
    plt.xlabel(infos['xlabel'], fontsize=40, fontweight='bold')
    plt.tick_params(axis='y', labelsize=20)
    for values, color, marker in zip(project_values, infos['colors'], infos['markers']):
        plt.plot(X[:len(values)], values, color=color, marker=marker, markersize=18, linewidth=3.5)
    if 'legends' in infos:
        plt.legend(infos['legends'], loc='upper left', fontsize=30, markerscale=1.8)
    if 'no_xtic' in infos and infos['no_xtic']:
        plt.xticks([])
    if 'xticks' in infos:
        plt.xticks(infos['xticks'], fontsize=20)

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

