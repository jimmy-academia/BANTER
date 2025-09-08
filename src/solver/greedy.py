import torch
from tqdm import tqdm

from utils import *
from debug import *

from .base import BaseSolver
from utils import make_batch_indexes


class BaselineSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        self.k = 20
        self.add_time = 0

    def initial_assignment(self):
        raise NotImplementedError
    
    def objective_pricing(self):
        pricing = self.market.vj/self.market.vj.mean() * (self.market.buyer_budgets.sum()/self.market.nft_counts.sum())
        return pricing

    def solve(self):
        self.pricing = self.objective_pricing()
        _assignments = self.initial_assignment()
        assert _assignments.shape == torch.Size([self.market.N, self.k])
        self.holdings = self.opt_uniform_holding(_assignments)

        return self.add_time


    def opt_uniform_holding(self, _assignments):
        # iterate over batch buyer to adjust holding recommendation for top k assignemnts
        N, M = self.market.N, self.market.M
        batch_size = N // 500 if N >= 9000 else N // 20

        # spending = torch.zeros(N, M + 1, device=self.args.device)
        # row_indices = torch.arange(spending.size(0), device='cuda:0').unsqueeze(1)
        # spending[row_indices, _assignments] = 1.0

        spending_scales = torch.rand(N, device=self.args.device)
        spending_scales.clamp_(0, 1)

        user_iterator = self.market.buyer_budgets.argsort(descending=True).tolist()
        user_iterator = make_batch_indexes(user_iterator, batch_size)

        pbar = tqdm(range(16), ncols=88, desc='Optimizing holdings', leave=False)

        prev_scale = float('inf')
        for __ in pbar:
            for user_index in user_iterator:
                
                batch_len = len(user_index)

                spending_scale = spending_scales[user_index].clone().detach().requires_grad_(True)
                batch_budgets = self.market.buyer_budgets[user_index]
                batch_assignments = _assignments[user_index]

                # batch_spending = spending[user_index] * spending_scale.unsqueeze(1) * batch_budgets.unsqueeze(1) /self.k
                # holdings = batch_spending[:, :-1] / self.pricing.unsqueeze(0)

                per_buyer_spending = spending_scale * batch_budgets / self.k  
                # budget allocated to each NFT per buyer

                row_indices = torch.arange(batch_len, device=self.args.device).unsqueeze(1).repeat(1, self.k).view(-1)
                col_indices = batch_assignments.view(-1)
                per_spending = per_buyer_spending.unsqueeze(1).repeat(1, self.k).view(-1)
                pricing_selected = self.pricing[col_indices]

                amount = per_spending / pricing_selected
                batch_holdings = torch.zeros(batch_len, M, device=self.args.device)
                batch_holdings[row_indices, col_indices] = amount

                utility = self.calculate_utilities(user_index, batch_holdings)    
                
                # Update spending scales using gradients
                utility.sum().backward()
                _grad = spending_scale.grad

                spending_scales[user_index] += 1e-2 * _grad/_grad.max()
                spending_scales.clamp_(min=0, max=1)

                # Normalize spending
                # spending = spending / spending.sum(dim=1, keepdim=True)
                
            # Track convergence
            delta = (spending_scales - prev_scale).abs().sum()
            prev_scale = spending_scales
            pbar.set_postfix(delta=float(delta))
            
            if delta < 1e-6:
                break
        
        holdings = torch.zeros(N, M, device=self.args.device)
        num_batches = (N + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), ncols=88, desc='batch_assign', leave=False):
            start = i * batch_size
            end = min(N, (i + 1) * batch_size)
            batch_indices = slice(start, end)
            batch_assignments = _assignments[batch_indices]
            batch_spending_scales = spending_scales[batch_indices]
            batch_budgets = self.market.buyer_budgets[batch_indices]
            per_buyer_spending = batch_spending_scales * batch_budgets / self.k  # size (batch_size,)

            # K = batch_assignments.size(1)
            row_indices = torch.arange(end - start, device=self.args.device).unsqueeze(1).repeat(1, self.k).view(-1)
            col_indices = batch_assignments.reshape(-1)
            per_spending = per_buyer_spending.unsqueeze(1).repeat(1, self.k).view(-1)
            pricing_selected = self.pricing[col_indices]

            values = per_spending / pricing_selected
            holdings_batch = torch.zeros(end - start, M, device=self.args.device)
            holdings_batch[row_indices, col_indices] = values
            holdings[batch_indices] = holdings_batch

        # spending = torch.zeros(N, M + 1, device=self.args.device)
        # row_indices = torch.arange(spending.size(0), device='cuda:0').unsqueeze(1)
        # spending[row_indices, _assignments] = 1.0

        # Calculate final demand
        # final_spending = spending * spending_scales.unsqueeze(1) * self.market.buyer_budgets.unsqueeze(1) / self.k
        # holdings = final_spending[:, :-1] / self.pricing.unsqueeze(0)
        return holdings


class RandomSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)
    def initial_assignment(self):
        random_assignments = torch.stack([torch.randperm(self.market.M)[:self.k] for _ in range(self.market.N)]).to(self.args.device)
        return random_assignments

class GreedySolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)

    def initial_assignment(self):
        favorite_assignments = ((self.market.uij+self.market.vj)/self.pricing).topk(self.k)[1]
        # .expand(self.market.N, self.k)
        return favorite_assignments
    