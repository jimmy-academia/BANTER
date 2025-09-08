import torch
from fast_pytorch_kmeans import KMeans
from .greedy import BaselineSolver
from tqdm import tqdm

from debug import *

class GroupSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)

    def initial_assignment(self):
        # group buyers based on preferences
        # not using kmeans-pytorch https://github.com/subhadarship/kmeans_pytorch/tree/master => nan issue
        # using Fast Pytorch Kmeans https://github.com/DeMoriarty/fast_pytorch_kmeans

        # num_clusters = 256
        num_clusters = 20
        kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(self.market.buyer_preferences)
        labels = labels.cpu()
        buyer_ids_list = [torch.arange(self.market.N)[labels==i] for i in range(num_clusters)]
        _len = max([len(x) for x in buyer_ids_list])
        _assignment = torch.ones((self.market.N, self.k), device=self.args.device).long()

        for batch_buyer_ids in tqdm(buyer_ids_list, leave=False, ncols=88, desc='iterate group equilibrium'):
            ## recommend items to group, each member has a vote on which item to choose
            selections = torch.randint(0, self.market.M, (_len, ), device=self.args.device)

            for __ in range(20): #256
                for buyer_id in batch_buyer_ids:
                    # choose a better item for buyer_id
                    # follow Lucas et al. Users’ satisfaction in recommendation systems for groups: an approach based on noncooperative games. WWW 2013
                    mask = torch.ones_like(self.market.uij[buyer_id], dtype=torch.bool)
                    mask[selections] = False
                    p = torch.argmax(self.market.uij[buyer_id]* mask) # best choice not selected (out of M)
                    q = torch.argmin(self.market.uij[buyer_id].masked_select(~mask)) # worst selected (out of _len)
                    selections[q] = p
            _assignment[batch_buyer_ids] = selections[:self.k]

        return _assignment

