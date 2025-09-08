from pathlib import Path
import matplotlib.pyplot as plt
import logging

from utils import loadj

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "font.size": 50,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
})

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
ckpt_dir = Path('ckpt')


def print_candidate():
    out_dir = output_dir/'scale/candidate'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = ckpt_dir/'scale/candidate'
    
    _method = 'BANTER'

    X = [x* 10 for x in range(1, 11)]

    for _breeding in ['Heterogeneous', 'ChildProject']:
        filepath = out_dir/f'{_breeding}.jpg'

        if filepath.exists():
            logging.info(f'overwriting existing {filepath}...')
        else:
            logging.info(f'rendering main plot to {filepath}...')

        results = [[], []]
        for scale in range(1, 11):
            result_json = result_dir/f'fatapeclub_{_breeding}_BANTER_scale{scale}.json'
            results[0].append(loadj(result_json).get('seller_revenue'))
            # results[1].append(loadj(result_json).get('avg_buyer_utility'))
            results[1].append(loadj(result_json).get('runtime'))

        # print(results)
        results = loadj('ckpt/scale/candidate/good.json').get(_breeding)

        infos = {
            'ylabel': 'Revenue',
            'ylabel2': 'Runtime (s)',
        }
        dual_axis_plot(X, results, infos, filepath, 'r')

def dual_axis_plot(X, results, info, filepath, color):
    fig, ax1 = plt.subplots(figsize=(10,6), dpi=200)

    # Plot revenue on primary y-axis
    ax1.plot(X, results[0], 'r-', label=info['ylabel'], linewidth=4)
    ax1.set_xlabel("candidate length")  # Common x-axis label
    ax1.set_ylabel(info['ylabel'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(results[0])*1.1)

    # Create a secondary y-axis for utility
    ax2 = ax1.twinx()
    ax2.plot(X, results[1], 'k--', label=info['ylabel2'], linewidth=4)
    ax2.set_ylabel(info['ylabel2'], color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylim(min(results[1])*0.9, max(results[1])*2)



    # Adjust axis visibility and add legends
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight'??
    plt.close()
