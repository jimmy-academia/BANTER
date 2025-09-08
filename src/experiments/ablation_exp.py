import time
import torch
import shutil
import logging
import random
from pathlib import Path
from tqdm import tqdm

from solver import BANTERSolver
from arguments import default_args
from utils import dumpj, loadj, deep_to_pylist, set_seeds

def check_completion_status():
    """Check completion status and return detailed information."""
    base_dir = Path('ckpt/ablation')
    
    if not base_dir.exists():
        print("No ablation experiments have been run yet.")
        return {}
    
    print("\n===== Ablation Experiment Status =====")
    print("(Balanced round-based execution)")
    
    status_info = {}
    total_runs_all = 0
    
    # Task 1: Optimization Ablation
    print("\nOptimization Ablation:")
    for ablation_id in range(3):
        result_file = base_dir / f'fatapeclub_ChildProject_optimization{ablation_id}_results.json'
        if result_file.exists():
            results = loadj(result_file)
            num_runs = len(results)
            config_key = ('optimization', ablation_id)
            status_info[config_key] = num_runs
            total_runs_all += num_runs
            
            ablation_name = ["BANTER", "BANTER (no init)", "INIT"][ablation_id]
            print(f"  {ablation_name:20}: {num_runs} runs")
    
    # Task 2: Scheduling Ablation
    print("\nScheduling Ablation:")
    for schedule_id in range(3):
        result_file = base_dir / f'fatapeclub_ChildProject_schedule{schedule_id}_results.json'
        if result_file.exists():
            results = loadj(result_file)
            num_runs = len(results)
            config_key = ('schedule', schedule_id)
            status_info[config_key] = num_runs
            total_runs_all += num_runs
            
            schedule_name = ["BANTER", "BANTER (fixed)", "BANTER (none)"][schedule_id]
            print(f"  {schedule_name:20}: {num_runs} runs")
    
    # Task 3: Module Ablation
    print("\nModule Sampling Ablation:")
    breeding_types = ['Heterogeneous', 'Homogeneous', 'ChildProject']
    for breed_type in breeding_types:
        for module_id in range(3):
            result_file = base_dir / f'fatapeclub_{breed_type}_module{module_id}_results.json'
            if result_file.exists():
                results = loadj(result_file)
                num_runs = len(results)
                config_key = ('module', breed_type, module_id)
                status_info[config_key] = num_runs
                total_runs_all += num_runs
                
                module_name = ["BANTER", "BANTER (objective)", "BANTER (random)"][module_id]
                print(f"  {breed_type:15} / {module_name:20}: {num_runs} runs")
    
    if status_info:
        min_runs = min(status_info.values())
        max_runs = max(status_info.values())
        avg_runs = total_runs_all / len(status_info)
        print(f"\nTotal: {total_runs_all} runs across {len(status_info)} configurations")
        print(f"Range: {min_runs} to {max_runs} runs per configuration")
        print(f"Average: {avg_runs:.1f} runs per configuration")
    
    return status_info


def run_ablation_exp(num_runs=5, persist=False):
    """
    Run ablation experiments with balanced distribution across configurations.
    
    Args:
        num_runs: Target number of runs per configuration (default: 5)
        persist: If True, continue running indefinitely. If False, stop when all configs reach num_runs.
    """
    args = default_args()
    
    # Create organized directory structure
    base_dir = args.ckpt_dir / 'ablation'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Show current status first
    print("\n" + "="*60)
    print("ABLATION EXPERIMENT - BALANCED EXECUTION")
    print("="*60)
    print(f"Target runs per configuration: {num_runs}")
    print(f"Persist mode: {'ON (continue indefinitely)' if persist else 'OFF (stop at target)'}")
    print("\nChecking current status...")
    
    status_info = check_completion_status()
    
    print("\nStarting experiments...")
    print("="*60 + "\n")
    
    # Create all configurations
    all_configurations = []
    
    # Task 1: Optimization Ablation
    for ablation_id in range(3):
        all_configurations.append(('optimization', ablation_id))
    
    # Task 2: Scheduling Ablation
    for schedule_id in range(3):
        all_configurations.append(('schedule', schedule_id))
    
    # Task 3: Module Ablation
    breeding_types = ['Heterogeneous', 'Homogeneous', 'ChildProject']
    for breed_type in breeding_types:
        for module_id in range(3):
            all_configurations.append(('module', breed_type, module_id))
    
    # Check initial completion status
    if not persist:
        min_runs = min(status_info.values()) if status_info else 0
        if min_runs >= num_runs:
            print(f"All configurations have reached {num_runs} runs. Stopping.")
            print("Use --persist flag to continue running experiments.")
            return
    
    completed_experiments = 0
    round_num = 0
    
    # Continue until stopping condition is met
    while True:
        round_num += 1
        
        # Determine which configurations to run this round
        if persist:
            # In persist mode, always run all configurations
            # But prioritize those with fewer runs
            configurations_to_run = sorted(all_configurations, 
                                        key=lambda x: status_info.get(x, 0))
        else:
            # In non-persist mode, only run configurations that haven't reached target
            configurations_to_run = []
            for config in all_configurations:
                current_runs = status_info.get(config, 0)
                if current_runs < num_runs:
                    configurations_to_run.append(config)
            
            # Sort by number of runs (least first) to balance
            configurations_to_run.sort(key=lambda x: status_info.get(x, 0))
        
        # If no configurations need more runs and we're not in persist mode, stop
        if not configurations_to_run and not persist:
            break
        
        # If in persist mode but somehow no configurations, use all
        if not configurations_to_run and persist:
            configurations_to_run = all_configurations
        
        # For balanced execution, only run configurations with the minimum number of runs
        if configurations_to_run:
            min_runs_in_batch = min(status_info.get(config, 0) for config in configurations_to_run)
            round_configurations = [config for config in configurations_to_run 
                                  if status_info.get(config, 0) == min_runs_in_batch]
        else:
            round_configurations = []
        
        if not round_configurations:
            break
        
        logging.info(f"\n=== Round {round_num} (Running configs with {min_runs_in_batch} runs) ===")
        
        # Progress bar for this round
        round_pbar = tqdm(
            round_configurations, 
            desc=f"Round {round_num}", 
            position=0,
            leave=True
        )
        
        for config in round_pbar:
            task_type = config[0]  # 'optimization', 'schedule', or 'module'
            
            # Prepare base arguments
            run_args = default_args()
            run_args.checkpoint_dir = base_dir
            run_args.nft_project_name = 'fatapeclub'
            run_args.breeding_type = 'ChildProject'  # Default, may be overridden
            
            if task_type == 'optimization':
                ablation_id = config[1]
                config_key = config
                
                # Set up file paths
                result_file = base_dir / f'fatapeclub_ChildProject_optimization{ablation_id}_results.json'
                
                # Set task-specific args
                run_args.read_initial_steps = True
                run_args.ablation_id = ablation_id
                
                task_name = ["BANTER", "BANTER (no init)", "INIT"][ablation_id]
                round_pbar.set_postfix({
                    'task': 'optimization',
                    'variant': task_name
                })
                
            elif task_type == 'schedule':
                schedule_id = config[1]
                config_key = config
                
                # Set up file paths
                result_file = base_dir / f'fatapeclub_ChildProject_schedule{schedule_id}_results.json'
                
                # Set task-specific args
                run_args.read_initial_steps = True
                run_args.schedule_id = schedule_id
                
                task_name = ["BANTER", "BANTER (fixed)", "BANTER (none)"][schedule_id]
                round_pbar.set_postfix({
                    'task': 'scheduling',
                    'variant': task_name
                })
                
            elif task_type == 'module':
                breed_type, module_id = config[1], config[2]
                config_key = config
                
                # Set up file paths
                result_file = base_dir / f'fatapeclub_{breed_type}_module{module_id}_results.json'
                
                # Set task-specific args
                run_args.breeding_type = breed_type
                run_args.module_id = module_id
                
                task_name = ["BANTER", "BANTER (objective)", "BANTER (random)"][module_id]
                round_pbar.set_postfix({
                    'task': 'module',
                    'breeding': breed_type,
                    'variant': task_name
                })
                
                # For the first round of module_id 0, try copying from main experiment if available
                if module_id == 0 and not result_file.exists():
                    main_file = args.ckpt_dir/'main_exp'/f'fatapeclub/BANTER/{breed_type}_results.json'
                    if main_file.exists():
                        try:
                            shutil.copy(main_file, result_file)
                            # Update status info and skip to next config
                            results = loadj(result_file)
                            status_info[config_key] = len(results)
                            round_pbar.set_postfix({
                                'status': 'copied from main',
                                'runs': len(results)
                            })
                            continue
                        except Exception as e:
                            logging.warning(f"Failed to copy from main experiment: {e}")
            
            # Load existing results
            if result_file.exists():
                results = loadj(result_file)
            else:
                results = []
            
            current_runs = len(results)
            
            # Generate unique seed for this run
            existing_seeds = {r['seed'] for r in results} if results else set()
            seed = random.randint(0, 2**32 - 1)
            while seed in existing_seeds:
                seed = random.randint(0, 2**32 - 1)
            
            # Set seed and run
            set_seeds(seed)
            
            # Run the experiment
            try:
                start = time.time()
                solver = BANTERSolver(run_args)
                solver.solve()    
                solver.count_results()
                runtime = time.time() - start
                
                if task_type in ['optimization', 'schedule']:
                    # For optimization and scheduling, save the revenue list
                    rev_list = solver.seller_revenue_list
                    if task_type == 'optimization' and run_args.ablation_id == 2:
                        # For INIT, create a flat list
                        rev_list = [solver.seller_revenue] * 20
                    
                    result = {
                        'seed': seed,
                        'runtime': runtime,
                        'revenue_list': rev_list
                    }
                else:  # Module
                    result = {
                        'seed': seed,
                        'runtime': runtime,
                        'seller_revenue': solver.seller_revenue,
                        'avg_buyer_utility': solver.buyer_utilities.mean().item(),
                        'utility_component': solver.utility_component
                    }
                
                # Process result
                result = {k: deep_to_pylist(v) for k, v in result.items()}
                
                # Save results
                results.append(result)
                dumpj(results, result_file)
                
                # Update status info
                status_info[config_key] = len(results)
                completed_experiments += 1
                
                # Update progress
                target_display = num_runs if not persist else '∞'
                if 'seller_revenue' in result:
                    revenue_display = f"{result['seller_revenue']:.1f}"
                else:
                    revenue_display = f"{result['revenue_list'][0]:.1f}..."
                    
                round_pbar.set_postfix({
                    'status': 'completed',
                    'revenue': revenue_display,
                    'runs': f"{len(results)}/{target_display}"
                })
                
            except Exception as e:
                logging.error(f"Error in {task_type} with config {config} and seed {seed}: {str(e)}")
                round_pbar.set_postfix({
                    'status': 'error',
                    'runs': f"{current_runs}/{num_runs if not persist else '∞'}"
                })
                continue
        
        logging.info(f"=== Round {round_num} Complete ===")
        
        # Check stopping condition (only in non-persist mode)
        if not persist:
            min_completed = min(status_info.values()) if status_info else 0
            if min_completed >= num_runs:
                print(f"\nAll configurations have reached {num_runs} runs. Stopping.")
                break
        
        # In persist mode, show current status periodically
        if persist and round_num % 5 == 0:
            min_runs = min(status_info.values()) if status_info else 0
            max_runs = max(status_info.values()) if status_info else 0
            print(f"\nStatus after {round_num} rounds: {min_runs}-{max_runs} runs per config")
    
    # Final summary
    if status_info:
        min_runs_final = min(status_info.values())
        max_runs_final = max(status_info.values())
        total_final = sum(status_info.values())
    else:
        min_runs_final = max_runs_final = total_final = 0
    
    logging.info(f"""
    ====== Ablation Experiments Complete ======
    Total configurations: {len(all_configurations)}
    Experiments completed this session: {completed_experiments}
    Final run range: {min_runs_final} to {max_runs_final} runs per configuration
    Total runs: {total_final}
    Persist mode: {'ON' if persist else 'OFF'}
    Results saved in: {base_dir}
    
    Directory structure:
    {base_dir}/
    ├── fatapeclub_ChildProject_optimization[0-2]_results.json
    ├── fatapeclub_ChildProject_schedule[0-2]_results.json
    └── fatapeclub_[Breeding]_module[0-2]_results.json
    ================================
    """)


if __name__ == "__main__":
    check_completion_status()