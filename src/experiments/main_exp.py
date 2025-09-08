import time
import torch
import logging
import random
from pathlib import Path
from tqdm import tqdm

from solver import get_solver
from arguments import default_args, nft_project_names, Breeding_Types, Baseline_Methods
from utils import dumpj, loadj, deep_to_pylist, set_seeds


def run_single_experiment(args, nft_project, method, breed_type, seed):
    """Run a single experiment with a specific seed."""
    # Set the seed for this run
    set_seeds(seed)
    
    args.nft_project_name = nft_project
    args.breeding_type = breed_type
    
    # Create and run solver
    solver = get_solver(args, method)
    
    start_time = time.time()
    add_time = solver.solve() or 0
    solver.count_results()
    runtime = time.time() - start_time + add_time
    
    # Collect results
    result = {
        'seed': seed,
        'runtime': runtime,
        'seller_revenue': solver.seller_revenue,
        'avg_buyer_utility': solver.buyer_utilities.mean().item(),
        'utility_component': solver.utility_component
    }
    
    # Also save the detailed data
    detailed_data = {
        'buyer_utilities': solver.buyer_utilities.cpu(),
        'pricing': solver.pricing.cpu()
    }
    
    return deep_to_pylist(result), detailed_data


def check_completion_status():
    """Check completion status and return detailed information."""
    base_dir = Path('ckpt/main_exp')
    
    print("\n===== Experiment Status =====")
    print("(Balanced round-based execution)")
    
    status_info = {}
    total_runs_all = 0
    
    for nft_project in nft_project_names:
        print(f"\n{nft_project}:")
        project_dir = base_dir / nft_project
        
        if not project_dir.exists():
            print("  No experiments run yet")
            for method in Baseline_Methods:
                for breed_type in Breeding_Types:
                    config_key = (nft_project, method, breed_type)
                    status_info[config_key] = 0
            continue
        
        for method in Baseline_Methods:
            method_dir = project_dir / method
            
            for breed_type in Breeding_Types:
                config_key = (nft_project, method, breed_type)
                result_file = method_dir / f'{breed_type}_results.json'
                
                if result_file.exists():
                    results = loadj(result_file)
                    num_runs = len(results)
                    status_info[config_key] = num_runs
                    total_runs_all += num_runs
                    print(f"  {method:12} / {breed_type:15} : {num_runs} runs")
                else:
                    status_info[config_key] = 0
    
    # Fill in missing configurations with 0 runs
    for nft_project in nft_project_names:
        for method in Baseline_Methods:
            for breed_type in Breeding_Types:
                config_key = (nft_project, method, breed_type)
                if config_key not in status_info:
                    status_info[config_key] = 0
    
    if status_info:
        min_runs = min(status_info.values())
        max_runs = max(status_info.values())
        avg_runs = total_runs_all / len(status_info)
        print(f"\nTotal: {total_runs_all} runs across {len(status_info)} configurations")
        print(f"Range: {min_runs} to {max_runs} runs per configuration")
        print(f"Average: {avg_runs:.1f} runs per configuration")
    
    return status_info


def run_main_exp(num_runs=10, persist=False):
    """
    Run main experiments with balanced distribution across configurations.
    
    Args:
        num_runs: Target number of runs per configuration (default: 10)
        persist: If True, continue running indefinitely. If False, stop when all configs reach num_runs.
    """
    args = default_args()
    
    # Create organized directory structure
    base_dir = args.ckpt_dir / 'main_exp'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Show current status first
    print("\n" + "="*60)
    print("MAIN EXPERIMENT - BALANCED EXECUTION")
    print("="*60)
    print(f"Target runs per configuration: {num_runs}")
    print(f"Persist mode: {'ON (continue indefinitely)' if persist else 'OFF (stop at target)'}")
    print("\nChecking current status...")
    
    status_info = check_completion_status()
    
    print("\nStarting experiments...")
    print("="*60 + "\n")
    
    # Create all configurations
    all_configurations = []
    for nft_project in nft_project_names[::-1]:
        for method in Baseline_Methods:
            for breed_type in Breeding_Types:
                all_configurations.append((nft_project, method, breed_type))
    
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
        
        for nft_project, method, breed_type in round_pbar:
            config_key = (nft_project, method, breed_type)
            current_runs = status_info.get(config_key, 0)
            
            # Set up file paths
            project_dir = base_dir / nft_project
            project_dir.mkdir(exist_ok=True)
            method_dir = project_dir / method
            method_dir.mkdir(exist_ok=True)
            
            config_name = f'{breed_type}'
            result_file = method_dir / f'{config_name}_results.json'
            detail_dir = method_dir / f'{config_name}_details'
            detail_dir.mkdir(exist_ok=True)
            
            # Load existing results
            if result_file.exists():
                results = loadj(result_file)
            else:
                results = []
            
            # Generate unique seed for this run
            existing_seeds = {r['seed'] for r in results} if results else set()
            seed = random.randint(0, 2**32 - 1)
            while seed in existing_seeds:
                seed = random.randint(0, 2**32 - 1)
            
            # Run experiment
            try:
                result, detailed_data = run_single_experiment(
                    args, nft_project, method, breed_type, seed
                )
                
                # Store results
                results.append(result)
                
                # Save detailed data separately
                detail_file = detail_dir / f'run_{seed}.pth'
                torch.save(detailed_data, detail_file)
                
                # Save results immediately after each run
                dumpj(results, result_file)
                
                # Update status info
                status_info[config_key] = len(results)
                completed_experiments += 1
                
                # Update progress
                target_display = num_runs if not persist else '∞'
                round_pbar.set_postfix({
                    'status': 'completed',
                    'revenue': f"{result['seller_revenue']:.1f}",
                    'runs': f"{len(results)}/{target_display}"
                })
                
            except Exception as e:
                logging.error(f"Error in {nft_project}/{method}/{breed_type} with seed {seed}: {str(e)}")
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
    ====== Experiments Complete ======
    Total configurations: {len(all_configurations)}
    Experiments completed this session: {completed_experiments}
    Final run range: {min_runs_final} to {max_runs_final} runs per configuration
    Total runs: {total_final}
    Persist mode: {'ON' if persist else 'OFF'}
    Results saved in: {base_dir}
    
    Directory structure:
    {base_dir}/
    └── [project]/
        └── [method]/
            ├── [breeding]_results.json
            └── [breeding]_details/
                └── run_[seed].pth
    ================================
    """)


def load_results_for_config(base_dir, project, method, breeding):
    """Load all results for a specific configuration."""
    result_file = base_dir / project / method / f'{breeding}_results.json'
    if not result_file.exists():
        return None
    
    results = loadj(result_file)
    return results


if __name__ == "__main__":
    check_completion_status()