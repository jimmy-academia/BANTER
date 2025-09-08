import time
import torch
import logging
import random
from pathlib import Path
from tqdm import tqdm

from solver import get_solver
from arguments import default_args, nft_project_names, Breeding_Types
from utils import dumpj, loadj, deep_to_pylist, set_seeds


def run_single_experiment(args, nft_project, method, breed_type, scale, seed):
    """Run a single experiment with a specific seed and scale."""
    # Set the seed for this run
    set_seeds(seed)
    
    args.nft_project_name = nft_project
    args.breeding_type = breed_type
    args.setN = args.setN * scale if args.setN else None  # Scale buyers
    args.setM = args.setM * scale if args.setM else None  # Scale items
    args.large = True if scale > 1 else False
    
    # Create and run solver
    solver = get_solver(args, method)
    
    start_time = time.time()
    add_time = solver.solve() or 0
    solver.count_results()
    runtime = time.time() - start_time + add_time
    
    # Collect results
    result = {
        'seed': seed,
        'scale': scale,
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
    base_dir = Path('ckpt/scale')
    
    print("\n===== Scale Experiment Status =====")
    print("(Balanced round-based execution)")
    
    status_info = {}
    total_runs_all = 0
    
    for subdir in ['large', 'yelp']:
        result_dir = base_dir / subdir
        if not result_dir.exists():
            continue
            
        print(f"\n{subdir.upper()}:")
        
        for result_file in result_dir.glob('*_results.json'):
            filename = result_file.name
            parts = filename.split('_')
            
            if subdir == 'large':
                # Format: {nft_project}_{breed_type}_{method}_scale{scale}_results.json
                if len(parts) < 4 or not parts[3].startswith('scale'):
                    continue
                    
                nft_project = parts[0]
                breed_type = parts[1]
                method = parts[2]
                scale = int(parts[3].replace('scale', ''))
                
                config_key = (subdir, nft_project, method, breed_type, scale)
                
            elif subdir == 'yelp':
                # Format: {nft_project}_{method}_{breed_type}_results.json
                if len(parts) < 3:
                    continue
                    
                nft_project = parts[0]
                method = parts[1]
                breed_type = parts[2].split('_')[0]  # Remove '_results.json'
                
                config_key = (subdir, nft_project, method, breed_type)
            
            if result_file.exists():
                results = loadj(result_file)
                num_runs = len(results)
                status_info[config_key] = num_runs
                total_runs_all += num_runs
                
                if subdir == 'large':
                    print(f"  {nft_project:12} / {breed_type:15} / {method:10} / scale {scale:2}: {num_runs} runs")
                else:
                    print(f"  {nft_project:12} / {breed_type:15} / {method:10}: {num_runs} runs")
    
    if status_info:
        min_runs = min(status_info.values())
        max_runs = max(status_info.values())
        avg_runs = total_runs_all / len(status_info)
        print(f"\nTotal: {total_runs_all} runs across {len(status_info)} configurations")
        print(f"Range: {min_runs} to {max_runs} runs per configuration")
        print(f"Average: {avg_runs:.1f} runs per configuration")
    
    return status_info


def run_scale_exp(num_runs=5, persist=False):
    """
    Run scale experiments with balanced distribution across configurations.
    
    Args:
        num_runs: Target number of runs per configuration (default: 5)
        persist: If True, continue running indefinitely. If False, stop when all configs reach num_runs.
    """
    args = default_args()
    
    # Create organized directory structure
    base_dir = args.ckpt_dir / 'scale'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Show current status first
    print("\n" + "="*60)
    print("SCALE EXPERIMENT - BALANCED EXECUTION")
    print("="*60)
    print(f"Target runs per configuration: {num_runs}")
    print(f"Persist mode: {'ON (continue indefinitely)' if persist else 'OFF (stop at target)'}")
    print("\nChecking current status...")
    
    status_info = check_completion_status()
    
    print("\nStarting experiments...")
    print("="*60 + "\n")
    
    # Create all configurations
    all_configurations = []
    
    # Large-scale experiments
    large_dir = base_dir / 'large'
    large_dir.mkdir(exist_ok=True)
    
    for nft_project in ['fatapeclub']:  # Use only Fat Ape Club for large-scale
        for method in ['Greedy', 'Group', 'NCF', 'LightGCN', 'HetRecSys', 'BANTER']:
            for breed_type in ['Heterogeneous']:  # Use only Heterogeneous for large-scale
                for scale in range(1, 11):  # Scales from 1 to 10
                    all_configurations.append(('large', nft_project, method, breed_type, scale))
    
    # Yelp experiments
    yelp_dir = base_dir / 'yelp'
    yelp_dir.mkdir(exist_ok=True)
    
    for method in ['Greedy', 'Group', 'NCF', 'LightGCN', 'HetRecSys', 'BANTER']:
        for breed_type in Breeding_Types[:-1]:  # Exclude 'None'
            all_configurations.append(('yelp', 'yelp', method, breed_type))
    
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
            experiment_type = config[0]  # 'large' or 'yelp'
            
            if experiment_type == 'large':
                _, nft_project, method, breed_type, scale = config
                config_key = config
                
                # Set up file paths
                result_file = large_dir / f'{nft_project}_{breed_type}_{method}_scale{scale}_results.json'
                detail_dir = large_dir / f'{nft_project}_{breed_type}_{method}_scale{scale}_details'
                
            elif experiment_type == 'yelp':
                _, nft_project, method, breed_type = config
                config_key = config
                
                # Set up file paths
                result_file = yelp_dir / f'{nft_project}_{method}_{breed_type}_results.json'
                detail_dir = yelp_dir / f'{nft_project}_{method}_{breed_type}_details'
            
            detail_dir.mkdir(exist_ok=True)
            
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
            
            # Prepare args for this specific configuration
            run_args = default_args()
            run_args.no_method_cache = True  # Force re-run for scaling
            
            if experiment_type == 'large':
                # Large-scale experiment setup
                N_M_infos = loadj('ckpt/N_M_infos.json')
                base_N = N_M_infos.get(nft_project, {}).get('N', 100)
                base_M = N_M_infos.get(nft_project, {}).get('M', 100)
                
                run_args.setN = base_N
                run_args.setM = base_M
                run_args.task = 'large'
                
                # Run experiment
                try:
                    round_pbar.set_postfix({
                        'type': 'large',
                        'project': nft_project,
                        'method': method,
                        'breeding': breed_type,
                        'scale': scale,
                        'runs': f"{current_runs}/{num_runs if not persist else '∞'}"
                    })
                    
                    result, detailed_data = run_single_experiment(
                        run_args, nft_project, method, breed_type, scale, seed
                    )
                    
                except Exception as e:
                    logging.error(f"Error in large/{nft_project}/{method}/{breed_type}/scale{scale} with seed {seed}: {str(e)}")
                    continue
                
            elif experiment_type == 'yelp':
                # Yelp experiment setup
                run_args.nft_project_name = 'yelp'
                run_args.task = 'yelp'
                
                # Run experiment
                try:
                    round_pbar.set_postfix({
                        'type': 'yelp',
                        'method': method,
                        'breeding': breed_type,
                        'runs': f"{current_runs}/{num_runs if not persist else '∞'}"
                    })
                    
                    # For yelp, scale is always 1
                    result, detailed_data = run_single_experiment(
                        run_args, 'yelp', method, breed_type, 1, seed
                    )
                    
                except Exception as e:
                    logging.error(f"Error in yelp/{method}/{breed_type} with seed {seed}: {str(e)}")
                    continue
            
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
            
            # Update progress with revenue information
            target_display = num_runs if not persist else '∞'
            round_pbar.set_postfix({
                'status': 'completed',
                'revenue': f"{result['seller_revenue']:.1f}",
                'runs': f"{len(results)}/{target_display}"
            })
        
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
    ====== Scale Experiments Complete ======
    Total configurations: {len(all_configurations)}
    Experiments completed this session: {completed_experiments}
    Final run range: {min_runs_final} to {max_runs_final} runs per configuration
    Total runs: {total_final}
    Persist mode: {'ON' if persist else 'OFF'}
    Results saved in: {base_dir}
    
    Directory structure:
    {base_dir}/
    ├── large/
    │   ├── {nft_project}_{breed_type}_{method}_scale{scale}_results.json
    │   └── {nft_project}_{breed_type}_{method}_scale{scale}_details/
    │       └── run_{seed}.pth
    └── yelp/
        ├── yelp_{method}_{breed_type}_results.json
        └── yelp_{method}_{breed_type}_details/
            └── run_{seed}.pth
    ======================================
    """)


if __name__ == "__main__":
    check_completion_status()