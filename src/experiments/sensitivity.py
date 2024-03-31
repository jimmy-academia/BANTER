import shutil
from solver import get_solver
from utils import *
import time

def run_sensitivity_tests():

    N_M_infos = loadj('ckpt/N_M_infos.json')  ## prepare with python visualize -c stats

    args = default_args()
    # args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'sensitivity'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print('>>> (sensitivity.py) Sensitivity tests \n')
    msg = f'''
         {nft_project_names[3:]}
        x {Breeding_Types}
        x {Baseline_Methods}
        x num or budget
        x 10 scales
        '''
    print(msg)
    for nft_project_name in nft_project_names[3:]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types:
            args.breeding_type = _breeding
            for tag in ['num', 'bud']:
                compact_results = {'revenue':[], 'utility':[], 'runtime':[]}
                new_result_path = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_{tag}.pth'
                if new_result_path.exists():
                    print(f'{new_result_path} exists, experiment is completed.')
                    continue

                for _method in Baseline_Methods:
                    method_revenue = []
                    method_utility = []
                    method_runtime = []
                    for scale in range(1, 11):
                        print(f'running [{nft_project_name}_{_method}_{_breeding}_{tag}{scale}] experiment...')

                        args.setN = None if tag == 'bud' else int(N_M_infos[nft_project_name]['N']/10 * scale)
                        Solver = get_solver(args, _method)
                        if tag == 'bud':
                            Solver.buyer_budgets *= (scale/10)
                        
                        start = time.time()
                        Solver.solve()  
                        runtime = time.time() - start
                        Solver.evaluate() # evaluate buyer utility, seller revenue

                        method_revenue.append(Solver.seller_revenue.item())
                        if _method == 'BANTER':
                            method_utility.append(Solver.buyer_utilities[:, :3].sum(1).mean().item())  ## average buyer utility
                        else:
                            method_utility.append(Solver.buyer_utilities[:, :2].sum(1).mean().item())  ## no breeding 
                        method_runtime.append(runtime)
                compact_results['revenue'].append(method_revenue)
                compact_results['utility'].append(method_utility)
                compact_results['runtime'].append(method_runtime)

                torch.save(compact_results, new_result_path)
                print('____________________________________ experiment done.')

