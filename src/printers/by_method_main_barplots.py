import torch
from pathlib import Path

from .central_plotter import make_legend, tripple_bar_plot
from utils import *

def plot_main_exp():
    out_sub_dir = 'main_exp/' 
    (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)
    for tag, ylabel in zip(['revenue', 'buyer_utility', 'runtime'], ['Revenue', 'Avg. Utility', 'Runtime (s)']):
        for nft_project_name in nft_project_names:
            filename = f'{tag}_{nft_project_name}.jpg'
            filepath = output_dir/out_sub_dir/filename
            if check_file_exists(filepath, f'main_exp {tag} plot'):
                continue

            project_values = []
            for _method in Baseline_Methods:
                method_vals = []
                for _breeding in Breeding_Types:
                    filepth = Path(f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth')
                    if filepth.exists():
                        if tag == 'revenue':
                            method_vals.append(torch.load(filepth)['seller_revenue'].item())
                        elif tag == 'buyer_utility':
                            if _method == 'BANTER':
                                method_vals.append(torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item())  ## average buyer utility
                            else:
                                method_vals.append(torch.load(filepth)['buyer_utilities'][:, :2].sum(1).mean().item())  ## no breeding recommendation
                        elif tag == 'runtime':
                            method_vals.append(torch.load(filepth)['runtime'])
                    else:
                        method_vals.append(0)
                project_values.append(method_vals)

            # set plot height
            y_axis_lim = max([max(method_vals) for method_vals in project_values])
            # _increase =  if tag == 'runtime' else 0.1
            y_axis_lim = y_axis_lim + 0.1 * y_axis_lim

            y_axis_min = min([min(method_vals) for method_vals in project_values]) if tag == 'runtime' else 0
            y_axis_min = y_axis_min - 0.1 * y_axis_min

            infos = {
                'log': tag == 'runtime',
                'ylabel': ylabel,
                'y_axis_lim': y_axis_lim,
                'y_axis_min': y_axis_min,
                'colors': thecolors,
                'patterns': thepatterns,
            }
            tripple_bar_plot(project_values, infos, filepath)

    filepath = output_dir/out_sub_dir/'legend.jpg'
    if check_file_exists(filepath, 'main_exp legends'):
        return
    make_legend(Baseline_Methods + Breeding_Types, filepath, 'tripple', thecolors, thepatterns)



# def plot_total_revenue():
#     out_sub_dir = 'total_revenue/' 
#     (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

#     for nft_project_name in nft_project_names:
#         filename = f'{nft_project_name}.jpg'
#         filepath = output_dir/out_sub_dir/filename
#         if check_file_exists(filepath, 'revenue plot'):
#             continue

#         project_revenues = []
#         for _method in Baseline_Methods:
#             revenues = []
#             for _breeding in Breeding_Types:
#                 filepth = Path(f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth')
#                 if filepth.exists():
#                     revenues.append(torch.load(filepth)['seller_revenue'].item())
#                 else:
#                     revenues.append(0)
#             project_revenues.append(revenues)

#         # set plot height
#         y_axis_lim = max([max(revenues) for revenues in project_revenues])
#         y_axis_lim = y_axis_lim + 0.1 * y_axis_lim

#         infos = {
#             'ylabel': 'Revenue',
#             'y_axis_lim': y_axis_lim
#         }
#         tripple_bar_plot(project_revenues, infos, filepath)

#     filepath = output_dir/out_sub_dir/'legend.jpg'
#     if check_file_exists(filepath, 'revenue legends'):
#         return
#     make_legend(Baseline_Methods + Breeding_Types, filepath, 'tripple')


# def plot_buyer_utilities():
#     out_sub_dir = 'buyer_utils/' 
#     (output_dir/out_sub_dir).mkdir(parents=True, exist_ok=True)

#     for project_name in nft_project_names:
#         filename = f'{project_name}.jpg'
#         filepath = output_dir/out_sub_dir/filename
#         if check_file_exists(filepath, 'utility plot'):
#             continue

#         project_butils = []
#         for _method in Baseline_Methods:
#             butilities = []
#             for _breeding in Breeding_Types:
#                 filepth = Path(f'ckpt/main_exp/{project_name}_{_method}_{_breeding}.pth')
#                 if filepth.exists():
#                     if _method == 'BANTER':
#                         butilities.append(torch.load(filepth)['buyer_utilities'][:, :3].sum(1).mean().item())  ## average buyer utility
#                     else:
#                         butilities.append(torch.load(filepth)['buyer_utilities'][:, :2].sum(1).mean().item())  ## no breeding recommendation
#                 else:
#                     butilities.append(0)
#             project_butils.append(butilities)

#         # set plot height
#         y_axis_lim = max([max(butilities) for butilities in project_butils])
#         y_axis_lim = y_axis_lim + 0.1 * y_axis_lim

#         infos = {
#             'ylabel': 'Avg. Utility',
#             'y_axis_lim': y_axis_lim
#         }
#         tripple_bar_plot(project_butils, infos, filepath)

# def quick_check():
#     ## quick check
#     for nft_project_name in nft_project_names:
#         for _breeding in Breeding_Types:
#             revenues = []
#             method_list = []
#             for _method in Baseline_Methods:
#                 filepath = f'ckpt/main_exp/{nft_project_name}_{_method}_{_breeding}.pth'
#                 try:
#                     revenues.append(torch.load(filepath)['seller_revenue'])
#                     method_list.append(_method)
#                 except:
#                     pass

#             if len(revenues) > 0:
#                 print(f'[{nft_project_name} {_breeding}]')
#                 print(method_list)
#                 print(f'rev: {revenues}')

