import random
from utils import *

class NFTProject:
    def __init__(self, nft_project_data, setN, setM, nft_project_name):
        self.setN = setN 
        self.setM = setM 
        self.nft_project_name = nft_project_name
        self.trait_dict = nft_project_data['trait_system']
        self.numericalize(nft_project_data)

    def numericalize(self, nft_project_data):
        asset_traits, buyer_assets_ids, buyer_budgets, item_counts = nft_project_data['asset_traits'], nft_project_data['buyer_assets_ids'], nft_project_data['buyer_budgets'], nft_project_data['item_counts']

        max_aid_len = max([len(x) for x in buyer_assets_ids])
        min_aid_len = min_purchase[nft_project_names.index(self.nft_project_name)] # filter buyers with min purchases
        alluser_prefs = []
        buyer_num = self.setN if self.setN is not None else len(buyer_assets_ids)

        aid_set = set()
        num_trades = 0
        for i in range(min(buyer_num, len(buyer_assets_ids))):
            num_trades += len(buyer_assets_ids[i])
            if len(buyer_assets_ids[i]) < min_aid_len:
                continue
            aid_set |= set(buyer_assets_ids[i])
            user_prefs = self.trait2label_vec([asset_traits[aid] for aid in buyer_assets_ids[i]])
            # pad until max length, then change shape
            user_prefs = user_prefs + [user_prefs[-1]] * (max_aid_len - len(user_prefs))
            user_prefs = [list(x) for x in zip(*user_prefs)]
            alluser_prefs.append(user_prefs)

        self.N = self.setN if self.setN is not None else len(buyer_assets_ids)
        self.M = self.setM if self.setM is not None else len(aid_set)

        bid_list = list(range(self.N))
        aid_set = list(aid_set)
        aid_set = [aid_set[i%len(aid_set)] for i in range(self.M)]

        self.item_attributes = self.trait2label_vec([asset_traits[i] for i in aid_set])
        self.user_preferences = [alluser_prefs[i%len(alluser_prefs)] for i in range(self.M)]
        self.user_budgets = [buyer_budgets[bi% len(buyer_budgets)] for bi in bid_list]
        self.item_counts = [item_counts[i% len(item_counts)] for i in aid_set]
        self.num_trades = num_trades

    def trait2label_vec(self, asset_traits):
        item_vec_list = []
        for item in asset_traits:
            item_vec = []
            for (trait, options), choice in zip(self.trait_dict.items(), item):
                choice = 'none' if choice == 'None' else choice
                try:
                    item_vec.append(options.index(choice))
                except:
                    item_vec.append(random.randint(0, len(options)-1))
            item_vec_list.append(item_vec)
        return item_vec_list

    # def compute_trait_stats(self):
    #     trait_counts = []
    #     for trait, options in self.trait_dict.items():
    #         trait_counts.append([1 for __ in options])
    #     for item_vec in self.item_attributes:
    #         for t, choice in enumerate(item_vec):
    #             trait_counts[t][choice] += 1
    #     return trait_counts

    