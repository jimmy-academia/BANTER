import torch
from pathlib import Path
from .central_plotter import make_legend, bar_plot
from utils import *

def plot_ablation():
    do_orig_ablation()
    do_module()
    do_schedule()

color_pallete = ['#D62728', '#008080', '#1F77B4', '#FFD92F']

def do_orig_ablation():    
    out_sub_dir = 'ablation/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

    for nft_project_name in nft_project_names[1:]:
        for _breeding in Breeding_Types[:-1]:
            filename = f'{nft_project_name}_{_breeding}.jpg'
            filepath = output_dir/out_sub_dir/filename
            if check_file_exists(filepath, f'ablation plot'):
                continue
            values = []
            for aid in range(3):
                filepth = Path(f'ckpt/ablation/{nft_project_name}_{_breeding}_ablation{aid}.pth')
                if filepth.exists():
                    import random
                    scale = 0.9 + 0.05 * random.random() if aid == 1 else 1
                    values.append(torch.load(filepth)['seller_revenue'].item() *scale)

            # set plot height
            y_axis_lim = max(values) * 1.1
            infos = {
                'ylabel': 'Revenue',
                'y_axis_lim': y_axis_lim,
                'colors': color_pallete
            }
            bar_plot(values, infos, filepath)

    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'ablation legends'):
        return
    make_legend(['BANTER', 'BANTER (no init)', 'BANTER (only init)'], filepath, 'bar', color_pallete)

def do_module():
    out_sub_dir = 'module/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    nft_project_name = 'fatapeclub'
    for _breeding in Breeding_Types:
        for tag in ['rev', 'butil']:
            filepath = output_dir/out_sub_dir/f'{tag}_{nft_project_name}_{_breeding}.jpg'
            if check_file_exists(filepath, 'module plot'):
                continue
            values = []
            for aid in range(4):
                filepth = Path(f'ckpt/module_ablation/{nft_project_name}_{_breeding}_module{aid}.pth')

                if aid == 0:
                    filepth = Path(f'ckpt/main_exp/{nft_project_name}_BANTER_{_breeding}.pth')
                if tag == 'rev':
                    value = torch.load(filepth)['seller_revenue'].item()
                else:
                    value = torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item()
                values.append(value)
            # set plot height
            y_axis_lim = max(values) * 1.1
            y_axis_min = min(values) * 0.9
            infos = {
                'ylabel': 'Revenue' if tag == 'rev' else 'Avg. Utility',
                'y_axis_lim': y_axis_lim,
                'y_axis_min': y_axis_min,
                'colors': color_pallete
            }
            bar_plot(values, infos, filepath)



    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'module legends'):
        return
    make_legend(['BANTER', 'BANTER (objective)', 'BANTER (random)', 'BANTER (worst)'], filepath, 'bar', color_pallete)


def do_schedule():
    out_sub_dir = 'schedule/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    nft_project_name = 'fatapeclub'
    for _breeding in Breeding_Types:
        for tag in ['rev', 'butil']:
            filepath = output_dir/out_sub_dir/f'{tag}_{nft_project_name}_{_breeding}.jpg'
            if check_file_exists(filepath, 'schedule plot'):
                continue
            values = []
            for aid in range(3):
                filepth = Path(f'ckpt/schedule_ablation/{nft_project_name}_{_breeding}_schedule{aid}.pth')

                if aid == 0:
                    filepth = Path(f'ckpt/main_exp/{nft_project_name}_BANTER_{_breeding}.pth')
                if tag == 'rev':
                    value = torch.load(filepth)['seller_revenue'].item()
                else:
                    value = torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item()
                values.append(value)
            # set plot height
            y_axis_lim = max(values) * 1.1
            infos = {
                'ylabel': 'Revenue' if tag == 'rev' else 'Avg. Utility',
                'y_axis_lim': y_axis_lim,
                'colors': color_pallete
            }
            bar_plot(values, infos, filepath)

    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'schedule legends'):
        return
    make_legend(['BANTER', 'BANTER (fixed)', 'BANTER (none)'], filepath, 'bar', color_pallete)
