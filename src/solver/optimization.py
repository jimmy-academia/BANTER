import random
import torch
from .base import BaseSolver
from tqdm import tqdm

class OptimizationSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

    def optimize_pricing(self):
        raise NotImplementedError
    
    def optimize_spending(self):
        raise NotImplementedError

    def solve(self, set_pricing=None):
        if set_pricing is None:
            self.optimize_pricing()
            spending = self.Uij/self.pricing
            spending = spending/spending.sum(1).unsqueeze(1) * self.buyer_budgets.unsqueeze(1)
            self.holdings = spending/spending.sum(0) * self.nft_counts
        else:
            self.pricing = set_pricing
            self.optimize_spending()

class AuctionSolver(OptimizationSolver):
    def __init__(self, args):
        super().__init__(args)
    
    def optimize_pricing(self):
        pricing = torch.rand(self.nftP.M, device=self.args.device)
        remain_budgets = self.buyer_budgets.clone()

        x,h,l = map(lambda __: torch.zeros(self.nftP.N, self.nftP.M).to(self.args.device), range(3))
        a = self.nft_counts.clone().float()
        eps = sum(remain_budgets)/self.nftP.N/self.nftP.M
        pbar = tqdm(range(128), ncols=88, desc='auction process')
        for __ in pbar:
            random_id_list = random.sample(range(self.nftP.N), self.nftP.N)
            for i in tqdm(random_id_list, leave=False, ncols=88):
                budget = remain_budgets[i]
                if budget > 0:
                    j = torch.argmax(self.Uij[i]/pricing)
                    if a[j] != 0:
                        amount = min(a[j], budget/pricing[j])
                        a[j] -= amount
                        x[i][j] += amount
                        l[i][j] += amount
                        remain_budgets[i] -= amount*pricing[j]
                    elif l.sum(0)[j] > 0:
                        candidate = [i for i in range(self.nftP.N) if (l[:, j]>0)[i]]
                        c = random.choice(candidate)
                        amount = min(l[c][j], budget/pricing[j])
                        l[c][j] -= amount
                        x[c][j] -= amount
                        remain_budgets[c] += amount*pricing[j]
                        h[i][j] += amount
                        x[i][j] += amount
                        remain_budgets[i] -= amount*pricing[j]*(1+eps)
                    else:
                        l[:, j] = h[:, j]
                        h[:, j] = 0
                        pricing[j] *= (1+eps)
                        pbar.set_postfix(val=pricing[j].item())

            if all(remain_budgets < min(pricing)): break
        self.pricing = pricing

    def optimize_spending(self):
        remain_budgets = self.buyer_budgets.clone()

        x = torch.zeros(self.nftP.N, self.nftP.M).to(self.args.device)
        a = self.nft_counts.clone()
        for __ in range(50):
            random_id_list = random.sample(range(self.nftP.N), self.nftP.N)
            for i in random_id_list:
                budget = remain_budgets[i]
                if budget > 0:
                    j = torch.argmax(self.Uij[i]/self.pricing)
                    if self.Uij[i][j] <= self.pricing[j]:
                        break
                    if a[j] != 0:
                        amount = min(a[j], budget/self.pricing[j])
                        a[j] -= amount
                        x[i][j] += amount
                        remain_budgets[i] -= amount*self.pricing[j]
            if all(remain_budgets < min(self.pricing)): break
        
        self.holdings = x