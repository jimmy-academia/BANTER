import json
import random

import sys
sys.path.append('../src')
from utils import dumpj
from pathlib import Path

from collections import defaultdict, Counter
from tqdm import tqdm

from debug import *

'''
format the yelp data into a dictionary with keys:
    - 'trait_system'
    - 'asset_traits'
    - 'item_counts'
    - 'buyer_budgets,
    'buyer_assets_ids': buyer_assets_ids,'   [28, 49, 42, ...]
    - 'buyer_assets_ids'   [[0, 2084], [1], [2, 1], [3], ...]

dumpj(yelp_nft_data, datadir/'yelp_nft_data.json')
 
'''

category_mapping = {
    "Nightlife": [
        "Nightlife", "Bars", "Sports Bars", "Cocktail Bars", "Pubs", "Lounges", "Dive Bars", 
        "Beer Bar", "Wine Bars", "Dance Clubs", "Hookah Bars", "Tiki Bars", "Jazz & Blues", "Whiskey Bars"
    ],
    "American": [
        "American (Traditional)", "American (New)", "Burgers", "Fast Food", "Comfort Food", "Soul Food",
        "Southern", "Steakhouses", "Hot Dogs", "Barbeque", "Breakfast & Brunch", "Fish & Chips", 
        "Sandwiches", "Chicken Wings", "Chicken Shop", "Diners", "Cheesesteaks"
    ],
    "Italian": [
        "Italian", "Pizza", "Pasta Shops", "Gelato", "Modern European", "Brasseries"
    ],
    "Mexican": [
        "Mexican", "Tacos", "Tex-Mex", "Cajun/Creole", "Latin American", "New Mexican Cuisine", 
        "Cuban", "Puerto Rican", "Peruvian", "Empanadas"
    ],
    "Cafes": [
        "Cafes", "Coffee & Tea", "Desserts", "Bakeries", "Juice Bars & Smoothies", "Ice Cream & Frozen Yogurt", 
        "Acai Bowls", "Bagels", "Waffles", "Creperies", "Cupcakes", "Coffee Roasteries", "Tea Rooms", 
        "Internet Cafes", "Patisserie/Cake Shop", "Custom Cakes"
    ],
    "Asian": [
        "Asian Fusion", "Chinese", "Japanese", "Sushi Bars", "Thai", "Vietnamese", "Korean", 
        "Ramen", "Dim Sum", "Poke", "Hawaiian", "Pakistani", "Indian", "Cantonese", "Szechuan", "Taiwanese", "Hot Pot", "Pan Asian", "Laotian"
    ],
}

def categorize(keys):
    for key in keys:
        for type_name, keywords in category_mapping.items():
            if any(keyword in key for keyword in keywords):
                return type_name
    return "Other"  # Default if no match is found

trait_system = {
    "State": ["PA", "FL", "TN", "MO", "IN", "Other"],
    "Stars": ["5.0", "4.5", "4.0", "3.5", "3.0", "Other"],
    "Payment": ["Both", "Bitcoin", "CreditCards", "None"],
    "ToGo": ["Both", "TakeOut", "Delivery", "None"],
    "GoodFor": ["Both", "Kids", "Meal", "None"],
    "Type":["Nightlife", "American", "Italian", "Mexican", "Cafes", "Asian", "Other"]
}

yelp_filename = 'yelp_academic_dataset_{}.json'
datadir = Path('../../../../DATASET/yelp')

if not datadir.exists():
    datadir = input('input path to yelp dataset directory')


def main():

    result_clean_json = Path('clean/yelp_full.json')
    if result_clean_json.exists():
        print(result_clean_json, 'result file exists!!!')
        return

        
    asset_traits = []
    item_counts = []
    business_id = []

    counter = Counter()

    count = 0
    with open(datadir/yelp_filename.format('business')) as file:
        for line in tqdm(file, ncols=90, desc='Processing Business Data'):
            bdict = json.loads(line)
            if bdict['review_count'] < 15 or bdict['categories'] is None or "Restaurants" not in bdict['categories'] or bdict['attributes'] is None:
                continue
            state = bdict['state'] if bdict['state'] in trait_system['State'] else 'Other'
            stars = bdict['stars'] if bdict['stars'] in trait_system['Stars'] else 'Other'
            attr = bdict['attributes']
            payment = trait_system['Payment'][bool(attr.get('BusinessAcceptsCreditCards', False)) + 2 * bool(attr.get('BusinessAcceptsBitcoin', False))]
            togo = trait_system['ToGo'][bool(attr.get('RestaurantsTakeOut', False)) + 2 * bool(attr.get('RestaurantsDelivery', False))]
            goodfor = trait_system['GoodFor'][
                bool(attr.get('GoodForKids', False)) + 2 * bool(attr.get('GoodForMeal', False))]
            type = categorize([cat.strip() for cat in bdict['categories'].split(',')])
            asset_traits.append([state, stars, payment, togo, goodfor, type])
            item_counts.append(1)
            business_id.append(bdict['business_id'])


    review_edge = defaultdict(set)
    edge_count = 0

    with open(datadir/yelp_filename.format('review')) as file:
        pbar_review = tqdm(file, ncols=90, desc='Processing Review Data')
        for line in pbar_review:
            rdict = json.loads(line)
            if rdict['business_id'] not in business_id:
                continue
            if rdict['stars'] >= 3:
                review_edge[rdict['user_id']].add(business_id.index(rdict['business_id']))
                edge_count += 1

            if edge_count % 100000 == 0:
                pbar_review.set_postfix(past=len(review_edge))
                print(edge_count, 10, len([k for k in review_edge.keys() if len(review_edge[k])>= 10]))
                print(edge_count, 15, len([k for k in review_edge.keys() if len(review_edge[k])>= 15]))
                print(edge_count, 25, len([k for k in review_edge.keys() if len(review_edge[k])>= 25]))
            pbar_review.set_postfix(count=edge_count)

    input('>pause<')
    user_id_list = list(review_edge.keys())
    for user_id in tqdm(user_id_list, ncols=90):
        if len(review_edge[user_id])<25:
            del review_edge[user_id]

    print(len(review_edge))
    input('>pause<')
    buyer_assets_ids = [list(v) for v in review_edge.values()]

    for bid in range(len(asset_traits)):
        u = random.randint(0, len(buyer_assets_ids)-1)
        if bid not in buyer_assets_ids[u]:
            buyer_assets_ids[u].append(bid)

    min_len = min(len(v) for v in buyer_assets_ids)
    max_len = max(len(v) for v in buyer_assets_ids)
    buyer_budgets = [10 + (len(v) - min_len) * 90 / (max_len - min_len) for v in buyer_assets_ids]

    yelp_nft_data = {
        'trait_system': trait_system,
        'asset_traits': asset_traits,
        'item_counts': item_counts,
        'buyer_budgets': buyer_budgets,
        'buyer_assets_ids': buyer_assets_ids,
    }

    print('N', len(buyer_budgets), 'M', len(asset_traits))
    dumpj(yelp_nft_data, result_clean_json)

if __name__ == '__main__':
    main()