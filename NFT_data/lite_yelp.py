import json
import random
from tqdm import tqdm

import sys
sys.path.append('../src')
from utils import dumpj, loadj

yelp = loadj('clean/yelp_full.json')
new_yelp = {}
new_yelp['trait_system'] = {
    "State": ["PA", "FL", "TN", "MO", "IN", "Other"],
    "Stars": ["5.0", "4.5", "4.0", "3.5", "3.0", "Other"],}

asset_traits = []
for attr_list in tqdm(yelp['asset_traits'], ncols=88):
    asset_traits.append(attr_list[:2])

new_yelp['asset_traits'] = asset_traits
new_yelp['item_counts'] = yelp['item_counts']
new_yelp['buyer_budgets'] = yelp['buyer_budgets']
new_yelp['buyer_assets_ids'] = yelp['buyer_assets_ids']


dumpj(new_yelp, 'clean/yelp.json')