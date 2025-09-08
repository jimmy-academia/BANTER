import time
import random
import torch
from .base import BaseSolver
from utils import make_batch_indexes
from tqdm import tqdm
from debug import *
import logging

class AuctionSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        self.task = getattr(args, 'task', 'default')
    
    def solve(self):

        if self.args.no_method_cache:
            self.run_bidding()
        else:
            cache_path = self._get_cache_path("")/f'Auction.pth'
            if not cache_path.exists():
                start = time.time()
                self.run_bidding()
                runtime = time.time() - start
                torch.save([runtime, self.pricing.cpu(), self.holdings.cpu()], cache_path)
            else:
                runtime, self.pricing, self.holdings = torch.load(cache_path, weights_only=True)
                self.pricing = self.pricing.to(self.args.device)
                self.holdings = self.holdings.to(self.args.device)
                return runtime

    def run_bidding(self):

        remain_budgets = self.market.buyer_budgets.clone()
        self.pricing = (1 + torch.rand(self.market.M, device=self.args.device))* sum(self.market.buyer_budgets)/self.market.M

        x,h,l = map(lambda __: torch.zeros(self.market.N, self.market.M), range(3))
        # x is holdings; h is holdings bought in high price; l is holdings bought in low price

        a = self.market.nft_counts.clone().float().cpu()
        threshold = sum(remain_budgets)/self.market.M
        logging.debug(f'threshold is {threshold}, min init pricing is {min(self.pricing)}')
        eps = 0.01
        for iter_ in tqdm(range(16), ncols=88, desc='auction process'):
            # user_id_list = torch.argsort(remain_budgets, descending=True)
            # random_id_list = random.sample(range(self.market.N), self.market.N)
            user_iterator = make_batch_indexes(self.market.N, self.market.N//50, True)
            for batch_id in tqdm(list(user_iterator), ncols=88):
                if self.task == 'yelp':
                    value, mask = (self.market.uij[batch_id]*self.market.vj).topk(5)[1]
                for i in tqdm(batch_id, ncols=88, leave=False):
                    budget = remain_budgets[i]
                    if budget > threshold:
                        if self.task == 'yelp':
                            j = torch.argmax(value[i]/self.pricing)
                            tj = mask[i][j]
                            check()
                        else:
                            j = torch.argmax((self.market.uij*self.market.vj)[i]/self.pricing)
                        if a[j] != 0:
                            amount = min(a[j].clone(), budget/self.pricing[j])
                            a[j] -= amount
                            x[i][j] += amount
                            l[i][j] += amount
                            remain_budgets[i] -= amount*self.pricing[j]
                        elif l.sum(0)[j] > 0:
                            candidate = [i for i in range(self.market.N) if (l[:, j]>0)[i]]
                            c = random.choice(candidate)
                            amount = min(l[c][j].clone(), budget/self.pricing[j])
                            l[c][j] -= amount
                            x[c][j] -= amount
                            remain_budgets[c] += amount*self.pricing[j]
                            h[i][j] += amount
                            x[i][j] += amount
                            remain_budgets[i] -= amount*self.pricing[j]*(1+eps)
                        else:
                            l[:, j] = h[:, j]
                            h[:, j] = 0
                            self.pricing[j] *= (1+eps)
                    else: 
                        break
        self.holdings = x.to(args.device)

    