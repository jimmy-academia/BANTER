import sys
import argparse
from dataset import prepare_nft_data
from experiments import *
from utils import set_verbose


def main():
    choices = ['main', 'scale', 'ablation', 'sensitivity']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    parser.add_argument('--runs', type=int, default=-1,
                        help='Number of runs for statistical analysis (default: 10 for main, 5 for others)')
    parser.add_argument('--persist', action='store_true',
                        help='Continue running even when default runs have been done')
    args = parser.parse_args()

    # Prepare data first
    prepare_nft_data()

    # Set verbosity level
    set_verbose(2)

    if args.c != 'all':
        if args.c == 'main':
            args.runs = 10 if args.runs == -1 else args.runs
            run_main_exp(num_runs=args.runs, persist=args.persist)
        elif args.c == 'scale':
            args.runs = 5 if args.runs == -1 else args.runs
            run_scale_exp(num_runs=args.runs, persist=args.persist)
        elif args.c == 'ablation':
            args.runs = 5 if args.runs == -1 else args.runs
            run_ablation_exp(num_runs=args.runs, persist=args.persist)
        elif args.c == 'sensitivity':
            args.runs = 5 if args.runs == -1 else args.runs
            run_sensitivity_exp(num_runs=args.runs, persist=args.persist)
    else:
        # For 'all', use appropriate defaults for each experiment type
        print("Running all experiments with multiple runs for statistical analysis:")
        print("- Main: 10 runs")
        print("- Scale: 5 runs") 
        print("- Ablation: 5 runs")
        print("- Sensitivity: 5 runs")
        
        # If persist mode is on, all experiments will continue running indefinitely
        print(f"Persist mode: {'ON (continue indefinitely)' if args.persist else 'OFF (stop at target)'}")
        
        run_main_exp(num_runs=10, persist=args.persist)
        run_scale_exp(num_runs=5, persist=args.persist)
        run_ablation_exp(num_runs=5, persist=args.persist) 
        run_sensitivity_exp(num_runs=5, persist=args.persist)


if __name__ == '__main__':
    main()