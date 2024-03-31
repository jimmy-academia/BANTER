import re
import os
import json
import argparse 
import random
import numpy as np
from types import SimpleNamespace

from pathlib import Path

import torch

import code 
import inspect

NFT_Projects = ['Axies Infinity', 'Bored Ape Yacht Club', 'Crypto Kitties', 'Fat Ape Club', 'Roaring Leader']
nft_project_names = [''.join(Project_Name.split()).lower() for Project_Name in NFT_Projects]
min_purchase = [6, 2, 2, 2, 1, 2, 1]

Baseline_Methods = ['Random', 'Popular', 'Auction', 'Group',  'HetRecSys', 'BANTER']
Breeding_Types = ['Heterogeneous', 'Homogeneous', 'ChildProject', 'None']

thecolors = ['#FFD92F', '#2CA02C', '#FF7F0E', '#1770af', '#ADD8E6', '#D62728']
themarkers = ['X', '^', 'o', 'P', 's', '*']
thepatterns = ['.', '-', 'x', '']


output_dir = Path('out')
output_dir.mkdir(parents=True, exist_ok=True)

def default_args():
    args = SimpleNamespace()
    args.ckpt_dir = Path('ckpt')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.device = torch.device("cuda:0")
    args.breeding_topk = 10
    args.cand_lim = 50
    args.num_child_sample = 100
    args.mutation_rate = 0.1
    args.num_trait_div = 4
    args.num_attr_class = 4
    args.decay = 0.9
    args.ablation_id = 0

    args.schedule_id = 0 # 0: dynamic, 1: fix weight 2: none
    args.module_id = 0 # 0: f_pop *TildeV (homo)/TildeV+attrclass (heter) 1:TildeV 2:rand
    args.gamma1 = 0.1
    args.gamma2 = 0.001
    return args

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

def check():
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    caller_locals = caller_frame.f_locals
    caller_globals = caller_frame.f_globals
    for key in caller_globals:
        if key not in globals():
            globals()[key] = caller_globals[key]

    frame_info = inspect.getframeinfo(caller_frame)
    caller_file = frame_info.filename
    caller_line = frame_info.lineno

    print('### check function called...')
    print(f"Called from {caller_file}")
    print(f"--------->> at line {caller_line}")

    code.interact(local=dict(globals(), **caller_locals))

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
    if hasattr(total, '__iter__') and hasattr(total, '__getitem__'):
        return (total[i:i+batch_size] for i in range(0, len(total), batch_size))
    elif isinstance(total, int):
        return (list(range(i, min(i + batch_size, total))) for i in range(0, total, batch_size))
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
    obj = torch.load(path)
    return deep_to_device(obj, device)

def check_file_exists(filepath, item_name=''):
    if filepath.exists():
        print(f'{item_name} exists for {filepath}, skipping...')
        return True
    else:
        print(f'creating {item_name} to {filepath}')
        return False
