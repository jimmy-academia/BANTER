import re
import os
import json
import argparse 
import random
import numpy as np
from pathlib import Path

import torch

thecolors = ['#FFD92F', '#2CA02C', '#FF7F0E', '#1770af', '#ADD8E6', '#D62728']
# thecolors = ['#FFD92F', '#2CA02C', '#FF7F0E', '#1F77B4', '#008080', '#ADD8E6', '#D62728']
themarkers = ['X', '^', 'o', 'P', 's', '*']
thepatterns = ['.', '-', 'x', '']


output_dir = Path('out')
output_dir.mkdir(parents=True, exist_ok=True)


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_verbose(verbose):
    # usages: logging.warning; logging.error, logging.info, logging.debug
    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()],  # Print to terminal
    )
    # Test the logging configuration
    # logging.warning("Logging setup complete - WARNING test")
    # logging.info("Logging setup complete - INFO test")
    # logging.debug("Logging setup complete - DEBUG test")

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        # json.dump(dictionary, f, indent=4)
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

def ask_proceed(objectname='file'):
    ans = input(f'{objectname} exists, proceed?').lower()
    if ans in ['', 'yes', 'y']:
        return True
    else:
        return False

def padd_list(original_list):
    max_length = max(len(sublist) for sublist in original_list)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in original_list]
    return padded_list

def writef(text, path):
    with open(path, 'w') as f:
        f.write(text)

def mkdirpath(dirpath):
    path_dir = Path(dirpath)
    path_dir.mkdir(parents=True, exist_ok=True)
    return path_dir

def inclusive_range(end, step):
    return range(step, end+step, step)

def make_batch_indexes(total, batch_size):
    """
    Create batched indexes for processing large data.
    
    Args:
        total: Total number of items or an iterable object
        batch_size: Size of each batch
        
    Returns:
        List of batch index lists
    """
    # Check if total is an iterable or an integer
    if hasattr(total, '__iter__') and hasattr(total, '__getitem__'):
        # Return a list of batches instead of a generator
        return [total[i:i+batch_size] for i in range(0, len(total), batch_size)]
    elif isinstance(total, int):
        # Return a list of batches instead of a generator
        return [list(range(i, min(i + batch_size, total))) for i in range(0, total, batch_size)]
    else:
        raise ValueError('total must be an iterable or an integer')

def deep_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: deep_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_to_cpu(v) for v in obj]
    else:
        return obj

def deep_to_pylist(obj):
    if isinstance(obj, torch.Tensor):
        # If it's a scalar tensor, use item()
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {k: deep_to_pylist(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_to_pylist(v) for v in obj]
    else:
        return obj
    
def deep_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: deep_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_to_device(v, device) for v in obj]
    else:
        return obj
    
def torch_cleansave(obj, path):
    obj = deep_to_cpu(obj)
    torch.save(obj, path)

def torch_cleanload(path, device):
    obj = torch.load(path, weights_only=True)
    return deep_to_device(obj, device)

def check_file_exists(filepath, item_name=''):
    if filepath.exists():
        print(f'{item_name} exists for {filepath}, skipping...')
        return True
    else:
        print(f'creating {item_name} to {filepath}')
        return False
