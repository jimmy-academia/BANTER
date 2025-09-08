import random
from pathlib import Path
from collections import defaultdict
# from typing import Tuple, Dict
from tqdm import tqdm

from utils import *
from arguments import nft_project_names

from debug import check

import logging

def prepare_nft_data():
    '''
    load and prepare nft data into [nft_project_data] and save into files
    '''
    data_dir = Path('../NFT_data')
    clean_dir = data_dir/'clean'
    clean_dir.mkdir(parents=True, exist_ok=True)

    for project_name in nft_project_names:        
        data_files = map(lambda x: Path(f'../NFT_data/{x}/{project_name}.json'), ['trades', 'NFT_attributes', 'trait_system'])

        project_file = clean_dir/f'{project_name}.json'

        if not project_file.exists(): 
            logging.info(f'dataset.py: preparing {project_file}...')
            nft_project_data = load_nft_project(project_name, clean_dir, data_files)
            dumpj(nft_project_data, project_file)
        else:
            logging.info(f'dataset.py: {project_file} exists')

def load_nft_project(project_name, tiny_dir, data_files):

    '''
    return:
    nft_project_data: dict_keys(['trait_system', 'asset_traits', 'item_counts', 
            'buyer_budgets', 'buyer_assets', 'buyer_assets_ids'])
    
    inputs 
    - trade_info: dict, 'note', 'result'
    - NFT_info: list of dict
    - trait_system: dict
    '''
    trade_info, NFT_info, trait_system = map(loadj, data_files)
    NFT_info, trait_system = filter_nft_attributes(project_name, NFT_info, trait_system)
    nft_project_data = process_nft_trades(trade_info['result'], NFT_info, trait_system, project_name)
    return nft_project_data

def filter_nft_attributes(project_name: str, NFT_info: list, trait_system: dict) -> tuple:
    """
    Processes NFT attributes and adjusts the trait system for missing data based on the project name.

    Parameters:
    - project_name (str): Name of the NFT project.
    - NFT_info (list of dict): List containing NFT instances, each listing the token_id and its attributes.
    - trait_system (dict): Trait system of the NFT project.

    Returns:
    - Tuple of updated NFT_info and trait_system.
    """
    if project_name in ['axiesinfinity']:
        trigger = 'None' # if project_name == 'axiesinfinity' else 'none'
        NFT_info = [nft for nft in NFT_info if trigger not in nft['trait']]
    if project_name in ['boredapeyachtclub', 'roaringleader', 'cryptokitties']:
        trait_system = {trait: trait_system[trait] + ['none'] for trait in trait_system}
    if project_name in ['roaringleader', 'cryptokitties']:
        if project_name == 'cryptokitties':
            trait_select_indices = [0,1,3,4,5,6,7]
        else:
            trait_select_indices = list(range(11))
        traits = [list(trait_system.keys())[i] for i in trait_select_indices]
        trait_system = {key: trait_system[key] for key in traits}
        NFT_info = [{**nft, 'trait': [nft['trait'][i] for i in trait_select_indices]} for nft in NFT_info]
    return NFT_info, trait_system

def fetchinfo(transaction):
    return transaction['buyer_address'], int(transaction['price']), int(transaction['token_ids'][0])

def process_nft_trades(trade_info, NFT_info, trait_system, project_name):
    '''
    nft_project_data: dict_keys(['trait_system', 
    'asset_traits', 'item_counts', => size N
    'buyer_budgets', 'buyer_assets_ids', => size M])
    '''
    # Initialize dictionaries for buyers and assets
    random_match = True if project_name=='stepn' else False
    buyer_info = defaultdict(lambda: {'budget': 0, 'asset_ids': []})
    asset_info = {'asset_traits':[], 'item_counts':[], 'atuples':[]}

    token_id2asset = {x['tokenId']:x for x in NFT_info}
    # Process each transaction
    for transaction in tqdm(trade_info, desc='Processing transactions', ncols=88):
        buyer_add, price, token_id = fetchinfo(transaction)
        if random_match:
            token_id = random.choice(list(token_id2asset.keys()))
        if token_id in token_id2asset:
            asset_trait = token_id2asset[token_id]['trait']
            atuple = tuple(asset_trait)
            
            if atuple not in asset_info['atuples']:
                aid = len(asset_info['atuples'])
                asset_info['atuples'].append(atuple)
                asset_info['asset_traits'].append(asset_trait)
                asset_info['item_counts'].append(1)
            else:
                aid = asset_info['atuples'].index(atuple)
                if project_name not in ['boredapeyachtclub', 'fatapeclub', 'roaringleader']:
                    asset_info['item_counts'][aid] += 1
            
            buyer_info[buyer_add]['budget'] += price
            buyer_info[buyer_add]['asset_ids'].append(aid)
    
    nft_project_data = {
        'trait_system': trait_system,
        'asset_traits': asset_info['asset_traits'],
        'item_counts': asset_info['item_counts'],
        'buyer_budgets': [buyer_info[buyer_add]['budget'] for buyer_add in buyer_info.keys()],
        'buyer_assets_ids': [buyer_info[buyer_add]['asset_ids'] for buyer_add in buyer_info.keys()]
    }
    return nft_project_data

