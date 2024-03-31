import shutil
import time
from solver import BANTERSolver
from utils import *

def run_ablation_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'ablation'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('>>> (ablation.py) Ablation tests')
    msg = f'''
         {nft_project_names[1:]}
        x {Breeding_Types} x BANTER
        x 3 ablation (0: original, 1:no init, 2:only init)
        '''
    print(msg)
    for nft_project_name in nft_project_names[1:]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types:
            args.breeding_type = _breeding
            for ablation_id in range(3):
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_ablation{ablation_id}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                    continue
                if ablation_id == 0:
                    # original version, just copy results
                    shutil.copy(args.ckpt_dir/'main_exp'/f'{nft_project_name}_BANTER_{_breeding}.pth', result_file)
                    continue

                print(f'running [{nft_project_name}, {_breeding}, ablation{ablation_id}] experiment...')

                args.ablation_id = ablation_id
                Solver = BANTERSolver(args)
                Solver.solve()    
                Solver.evaluate()
                Result = {
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                    }
                torch.save(deep_to_cpu(Result), result_file)
                print('______________________________ experiment done.')



def run_module_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'module_ablation'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('>>> (ablation.py) Module tests')
    msg = f'''
         {nft_project_names[3:4]}
        x {Breeding_Types} x BANTER
        x 3 Module (0: original, 1:objective, 2:random, 3: worst)
        '''
    print(msg)
    for nft_project_name in nft_project_names[3:4]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types:
            args.breeding_type = _breeding
            for module_id in range(4):
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_module{module_id}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                    continue
                # if module_id == 0:
                    # original version, just copy results
                    # shutil.copy(args.ckpt_dir/'main_exp'/f'{nft_project_name}_BANTER_{_breeding}.pth', result_file)
                    # continue

                print(f'running [{nft_project_name}, {_breeding}, module{module_id}] experiment...')

                args.module_id = module_id
                Solver = BANTERSolver(args)
                Solver.solve()    
                Solver.evaluate()
                Result = {
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                    }
                torch.save(deep_to_cpu(Result), result_file)
                print('______________________________ experiment done.')


def run_schedule_tests():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'schedule_ablation'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print()
    print('>>> (ablation.py) Schedule tests')
    msg = f'''
         {nft_project_names[3:4]}
        x {Breeding_Types} x BANTER
        x 3 schedule (0: original, 1:fixed, 2:none)
        '''
    print(msg)
    for nft_project_name in nft_project_names[3:4]:
        args.nft_project_name = nft_project_name
        for _breeding in Breeding_Types:
            args.breeding_type = _breeding
            for schedule_id in range(3):
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_breeding}_schedule{schedule_id}.pth'
                if result_file.exists():
                    print(f'{result_file} exists, experiment is completed.')
                    continue
                # if schedule_id == 0:
                    # original version, just copy results
                    # shutil.copy(args.ckpt_dir/'main_exp'/f'{nft_project_name}_BANTER_{_breeding}.pth', result_file)
                    # continue

                print(f'running [{nft_project_name}, {_breeding}, schedule{schedule_id}] experiment...')

                args.schedule_id = schedule_id
                Solver = BANTERSolver(args)
                Solver.solve()    
                Solver.evaluate()
                Result = {
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                    }
                torch.save(deep_to_cpu(Result), result_file)
                print('______________________________ experiment done.')

