'''
This is the main module of the dev-NFT project.
'''

__author__ = "Jimmy Yeh"

import argparse
from data_preprocessing import prepare_nft_data
from experiments import *

def main():
    """
    Run all (unfinished) experiments
    """
    exp_list = [run_experiments, run_sensitivity_tests, run_ablation_tests, run_module_tests, run_schedule_tests, adjust_pruning_tests, do_case_study]
    choices = ['main', 'sensitivity', 'ablation', 'module', 'schedule', 'prunning', 'case']
    parser = argparse.ArgumentParser(description='Visualize NFT data')
    parser.add_argument('-c', choices=choices+['all'], default='all')
    args = parser.parse_args()
    
    prepare_nft_data() # prepares nft data into files 
    
    if args.c == 'all':
        run_experiments() 
        run_sensitivity_tests()
        run_ablation_tests()
        run_module_tests()
        run_schedule_tests()
        adjust_pruning_tests()
        do_case_study()
    else:
        exp_list[choices.index(args.c)]()

if __name__ == "__main__":
    main()