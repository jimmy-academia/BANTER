from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 50
plt.rcParams["font.weight"] = "bold"
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40

def line_plot(X, project_values, infos, filepath):
    figsize = infos['figsize'] if 'figsize' in infos else (12, 6)
    plt.figure(figsize=figsize, dpi=200)
    plt.ylabel(infos['ylabel'])
    plt.xlabel(infos['xlabel'])
    for values, color, marker in zip(project_values, infos['colors'], infos['markers']):
        plt.plot(X, values, color=color, marker=marker, markersize=10, linewidth=3.5)
    if 'legends' in infos:
        plt.legend(infos['legends'], loc='upper left', fontsize=30)
    if 'no_xtic' in infos and infos['no_xtic']:
        plt.xticks([])
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def bar_plot(values, infos, filepath):
    plt.figure(figsize=(8, 6), dpi=200)
    plt.ylabel(infos['ylabel'])
    y_min = infos['y_axis_min'] if 'y_axis_min' in infos else 0
    plt.ylim(y_min, infos['y_axis_lim'])
    for x, val, color in zip(range(len(values)), values, infos['colors']):
        plt.bar(x, val, color=color)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()

def rainbow_bar_plot(project_revenues, infos, filepath):
    plt.figure(figsize=(13, 6), dpi=200)
    plt.ylabel(infos['ylabel'])
    plt.ylim(infos['y_axis_min'], infos['y_axis_lim'])

    bar_width = 1
    set_width = 1.2*len(infos['colors']) + 0.6
    indexes = range(len(project_revenues))
    for index, rev_rainbow in zip(indexes, project_revenues):
        for k, (rev, color) in enumerate(zip(rev_rainbow, infos['colors'])):
            plt.bar(index*set_width+k*(bar_width+0.2), rev, bar_width, color=color)
        if index != len(project_revenues) - 1:
            plt.axvline(x=index*set_width+(k+1)*(bar_width+0.2)-0.2, color='black', linestyle=':', linewidth=3)
    
    if infos['xticks'] is None:
        plt.xticks([])
    else:
        plt.xticks([index*set_width+ (len(infos['colors'])-1)/2 *(bar_width+0.2) for index in range(len(project_revenues))], infos['xticks'], fontsize=50)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()

def tripple_bar_plot(project_revenues, infos, filepath):
    plt.figure(figsize=(13, 6), dpi=200)
    plt.ylabel(infos['ylabel'])
    plt.ylim(infos['y_axis_min'], infos['y_axis_lim'])

    bar_width = 1
    set_width = 1.2*len(infos['patterns'])
    indexes = range(len(project_revenues))
    for index, rev_tripple, color in zip(indexes, project_revenues, infos['colors']):
        for k, (rev, pat) in enumerate(zip(rev_tripple, infos['patterns'])):
            plt.bar(index*set_width+k*(bar_width+0.2), rev, bar_width, color=color, edgecolor='black', hatch=pat*2)

    plt.xticks([])
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()


def make_legend(legends, filepath, tag, colors, patterns=None, markers=None):

    # tag = 'bar', 'tripple', 'line'
    fig, ax = plt.subplots()
    if tag == 'bar':
        bars = [ax.bar(0, 0, color=colors[i], label=legends[i]) for i in range(len(legends))]
    elif tag == 'line':
        [ax.plot(0, 0, color=colors[i], label=legends[i], marker=markers[i], markersize=30, linewidth=12)  for i in range(len(legends))]
    elif tag == 'tripple':
        num_breed = 4
        bars = [ax.bar(0,0, color=colors[i], label=legends[i]) for i in range(len(legends) - num_breed)]
        bars += [ax.bar(0,0, color='white', edgecolor='black', hatch=patterns[i], label=legends[i-num_breed]) for i in range(num_breed)]
    else:
        raise Exception(f'{tag} not found')

    # Extract the handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)

    legend_fig_width = len(legends) * 0.5  # inches per entry, adjust as needed
    fig_legend = plt.figure(figsize=(legend_fig_width, 1), dpi=300)  # High DPI for quality
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=len(legends), frameon=False, fontsize=50, handlelength=0.8, handletextpad=0.2, columnspacing=0.75)
    fig_legend.savefig(filepath, bbox_inches='tight')
    plt.close(fig_legend)
