import sys
import argparse
from renders import *
from utils import set_verbose

def main():
    """
    Run all visualizations
    """
    choices = ['stats', 'main', 'scale', 'ablation', 'candidate', 'sensitivity']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    args = parser.parse_args()

    set_verbose(1)

    if args.c != 'all':
        getattr(sys.modules[__name__], f'print_{args.c}')()
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'print_{choice}')()


if __name__ == "__main__":
    main()