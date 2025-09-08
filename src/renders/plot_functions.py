import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import loadj

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "font.size": 50,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
})

output_dir = Path('ckpt/output')
output_dir.mkdir(exist_ok=True)
ckpt_dir = Path('ckpt')

def calculate_stats(values):
    """Calculate mean and standard error for a list of values."""
    if not values:
        return 0, 0
    values = np.array(values)
    mean = np.mean(values)
    std_err = np.std(values) / np.sqrt(len(values))  # Standard error
    return mean, std_err

def rainbow_bar_plot_with_error(project_revenues, project_errors, infos, filepath, scalar=False):
    """Creates a rainbow bar plot with error bars and saves it to a file.

    Args:
        project_revenues (list): List of mean project revenues.
        project_errors (list): List of standard errors for project revenues.
        infos (dict): Information for the plot (e.g., colors, xticks, labels).
        filepath (str or Path): Path to save the plot.
        scalar (bool): Whether to use scientific notation.
    """
    figsize = infos['figsize'] if 'figsize' in infos else (13, 6)
    plt.figure(figsize=figsize, dpi=200)
    plt.ylabel(infos['ylabel'], fontweight='bold')
    plt.ylim(infos['y_axis_min'], infos['y_axis_max'])

    bar_width = 1
    set_width = 1.2*len(infos['colors']) + 0.6
    indexes = range(len(project_revenues))
    
    for index, (rev_rainbow, err_rainbow) in zip(indexes, zip(project_revenues, project_errors)):
        for k, (rev, err, color) in enumerate(zip(rev_rainbow, err_rainbow, infos['colors'])):
            x_pos = index*set_width+k*(bar_width+0.2)
            plt.bar(x_pos, rev, bar_width, color=color, yerr=err, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
        if index != len(project_revenues) - 1:
            plt.axvline(x=index*set_width+(k+1)*(bar_width+0.2)-0.2, color='black', linestyle=':', linewidth=3)
    
    if infos['xticks'] is not None:
        tick_positions = [index * set_width + (len(infos['colors']) - 1) / 2 * (bar_width + 0.2) for index in range(len(project_revenues))]
        plt.xticks(tick_positions, infos['xticks'], fontsize=infos.get('xtick_size', 30))
    else:
        plt.xticks([])

    if scalar:
        ax = plt.gca() 
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)  

    if infos['log']:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def line_plot_with_error(X, project_values, project_errors, infos, filepath):
    """Line plot with error bars."""
    figsize = infos['figsize'] if 'figsize' in infos else (12, 6)
    plt.figure(figsize=figsize, dpi=200)
    plt.ylabel(infos['ylabel'])
    plt.xlabel(infos['xlabel'])
    
    for values, errors, color, marker in zip(project_values, project_errors, infos['colors'], infos['markers']):
        x_vals = X[:len(values)]
        plt.errorbar(x_vals, values, yerr=errors, color=color, marker=marker, 
                    markersize=18, linewidth=3.5, capsize=5, capthick=2)
    
    if 'legends' in infos:
        plt.legend(infos['legends'], loc='upper left', fontsize=30, markerscale=1.8)
    if 'no_xtic' in infos and infos['no_xtic']:
        plt.xticks([])
    if 'xticks' in infos:
        plt.xticks(infos['xticks'])
    if 'xmin' in infos:
        plt.xlim(infos['xmin'], None)
    if infos['log']:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

# Keep backward compatibility with original functions
def rainbow_bar_plot(project_revenues, infos, filepath, scalar=False):
    """Original rainbow bar plot without error bars."""
    # If error data is not provided, create zero errors
    project_errors = [[0] * len(rev_row) for rev_row in project_revenues]
    rainbow_bar_plot_with_error(project_revenues, project_errors, infos, filepath, scalar)

def line_plot(X, project_values, infos, filepath):
    """Original line plot without error bars."""
    # If error data is not provided, create zero errors
    project_errors = [[0] * len(values) for values in project_values]
    line_plot_with_error(X, project_values, project_errors, infos, filepath)

def make_legend(legends, filepath, tag, colors, markers=None):
    """Creates a legend plot and saves it to a file."""
    fig, ax = plt.subplots()
    if tag == 'bar':
        [ax.bar(0, 0, color=colors[i], label=legends[i]) for i in range(len(legends))]
    elif tag == 'line':
        [ax.plot(0, 0, color=colors[i], label=legends[i], marker=markers[i], markersize=30, linewidth=12)  for i in range(len(legends))]
    else:
        raise ValueError(f"Unsupported tag: {tag}")

    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)

    legend_fig_width = len(legends) * 0.5  # inches per entry, adjust as needed
    fig_legend = plt.figure(figsize=(legend_fig_width, 1), dpi=300)  
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=len(legends), frameon=False, 
        fontsize=50, handlelength=0.8, handletextpad=0.2, columnspacing=0.75, markerscale=1.2)
    fig_legend.savefig(filepath, bbox_inches='tight')
    plt.close(fig_legend)