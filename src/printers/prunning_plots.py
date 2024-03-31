import torch
from pathlib import Path

from .central_plotter import line_plot, make_legend
from utils import *

def plot_prunning():
    out_sub_dir = 'prunning/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

    cand_lim_list = [1,2,3,4,5,10, 15, 20, 25, 30, 40, 50]


    # nft_project_name = 'boredapeyachtclub'
    # _breeding = 'Heterogeneous'

    for nft_project_name in nft_project_names:
        for _breeding in Breeding_Types[:2]:
            for tag in ['rev', 'butil']:
                filename = f'{tag}_{nft_project_name}_{_breeding}.jpg'
                filepath = output_dir/out_sub_dir/filename
                if check_file_exists(filepath, 'prunning plot'):
                    continue                

                values = []
                for cand_lim in cand_lim_list:
                    filepth = Path(f'ckpt/prunning/{nft_project_name}_{_breeding}_prunning{cand_lim}.pth')
                    if tag == 'rev':
                        values.append(torch.load(filepth)['seller_revenue'].item())
                    else:
                        values.append(torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item())

                ylabel ='Revenue'
                xlabel = '# Candidate'
                infos = {
                    'ylabel': ylabel,
                    'xlabel': xlabel,
                    'colors': ['black'],
                    'markers': [''],
                }
                line_plot(cand_lim_list, [values], infos, filepath)


