from solver.base import BaseSolver
from utils import *
from arguments import default_args, nft_project_names, NFT_Projects
import logging
def print_stats():
    args = default_args()
    args.setN = None
    args.setM = None
    args.breeding_type = 'None'
    print()
    print("\tNFT project \t N \t M\t#asset\t # attr. \t # trade")

    N_M_infos = {}
    for nft_project_name, Project in zip(nft_project_names, NFT_Projects):
        args.nft_project_name = nft_project_name
        Solver = BaseSolver(args)
        infos = [Project, Solver.nftP.N, Solver.nftP.M, Solver.nft_counts.sum().item(), Solver.nft_attributes.shape[-1], Solver.nftP.num_trades]
        print('    {}\t&${}$\t&${}$\t&${}$\t&${}$\t&${}$\\\\'.format(*infos))
        N_M_infos[f'{nft_project_name}'] = {'N': Solver.nftP.N, 'M': Solver.nftP.M}

    info_file = 'ckpt/N_M_infos.json'
    dumpj(N_M_infos, info_file)
    logging.info(f'saved {info_file}')