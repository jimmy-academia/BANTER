import torch
from tqdm import tqdm
from utils import make_batch_indexes
from .base import BaseSolver
from utils import *

from debug import *
class BANTERSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

        self.task = getattr(args, 'task', 'default')  # Add this
        self.seller_revenue_list = []  # For tracking

    def solve(self):
        '''
        proposed BANTER method for NFT and pricing recommendation to obtain Market Equilibrium
        '''
        # self.pricing, self.holdings.
        ## pricing initialization
        # ablation_id 0: full, 1: no init, 2: only init
        # self.pricing_list = []

        if self.args.ablation_id in [0, 2]:
            self.pricing = self.greedy_init_pricing()
        else:
            self.pricing = torch.rand(self.market.M, device=self.args.device)

        ## demand-based optimization
        eps = self.args.eps
        pbar = tqdm(range(64), ncols=88, desc='BANTER Solver!') #64
        # self.pricing_list.append(self.pricing)
        if self.args.ablation_id == 2:
            self.holdings = self.solve_user_demand()
            self.count_results()
            return

        if self.args.read_initial_steps:
            self.seller_revenue_list = []

        counter = 0
        best_counter = 0
        best_excess = 1e9
        best_pricing = None
        
        for iter_ in pbar:

            demand = self.solve_user_demand()
            demand = demand.sum(0)
            excess = demand - self.market.nft_counts

            if self.args.schedule_id == 0:
                eps = eps*torch.exp(-self.args.gamma1*excess.norm().sum()/self.market.nft_counts.sum())
                # + self.args.gamma2 * torch.tanh(self.ratio - 1)
            elif self.args.schedule_id == 1:
                eps = eps * 0.99

            self.pricing *= ( 1 +  eps * excess/(excess.abs().sum()))
            self.pricing = torch.where(self.pricing < 1e-10, 1e-10, self.pricing) 
            pbar.set_postfix(excess=float(excess.abs().sum()))

            if excess.abs().sum() < 500:
                counter += 1
            else:
                counter = 0
            if excess.abs().sum() < best_excess:
                best_excess = excess.sum()
                best_pricing = self.pricing.clone()
                best_counter = 0
            else:
                best_counter += 1

            if self.args.read_initial_steps:
                # self.pricing_list.append(self.pricing)
                self.holdings = self.solve_user_demand()
                self.count_results()
                self.seller_revenue_list.append(self.seller_revenue)
                if iter_ == 20: break
        
        self.pricing = best_pricing
        self.holdings = self.solve_user_demand()

    def solve_user_demand(self, set_user_index=None):

        div = 20 if self.market.N < 9000 else self.market.N // 500
        if set_user_index is not None:
            batch_user_iterator = [set_user_index]
        else:
            batch_user_iterator = self.market.buyer_budgets.argsort(descending=True).tolist()
            batch_user_iterator = make_batch_indexes(batch_user_iterator, self.market.N//div)

        if self.args.large:
            spending = torch.rand(self.market.N, self.market.M+1)
        else:
            spending = torch.rand(self.market.N, self.market.M+1).to(self.args.device) # N x M+1 additional column for remaining budget
        
        spending /= spending.sum(1).unsqueeze(1)

        pbar = tqdm(range(16), ncols=88, desc='Solving user demand!', leave=False)
        user_eps = 1e-4
        for __ in pbar:
            buyer_utility = 0
            for user_index in tqdm(batch_user_iterator, ncols=88, leave=False, total=div+1):
                spending_var = spending[user_index]
                if self.args.large:
                    spending_var = spending_var.to(self.args.device)
                spending_var.requires_grad = True
                
                batch_budget = self.market.buyer_budgets[user_index]
                holdings = self.hatrelu(spending_var[:, :-1]*batch_budget.unsqueeze(1)/self.pricing.unsqueeze(0)) #, self.market.nft_counts
                _utility = self.calculate_utilities(user_index, holdings)
                _utility.backward(torch.ones_like(_utility))

                buyer_utility += _utility.detach().mean().item()
                _grad = spending_var.grad.cpu() if self.args.large else spending_var.grad
                spending[user_index] += user_eps* _grad
                # spending = torch.where(spending < 0, 0, spending)
                spending.clamp_(min=0)
                spending /= spending.sum(1).unsqueeze(1)
            
            pbar.set_postfix(delta= float(spending[:, -1].sum() - spending[:, -1].sum()))

        if self.args.large or self.task == 'yelp':
            for user_index in tqdm(batch_user_iterator, ncols=88, leave=False, total=div+1):
                spending[user_index, :-1] *= self.market.buyer_budgets[user_index].unsqueeze(1)
                spending[user_index, :-1] /= self.pricing
            spending = spending.cpu()    
            demand = self.hatrelu(spending[:, :-1])
            demand = demand.to(self.args.device)
        else:
            demand = self.hatrelu(spending[:, :-1]*self.market.buyer_budgets.unsqueeze(1)/self.pricing.unsqueeze(0)) #, self.market.nft_counts
        return demand

    def hatrelu(self, x, threshold=1):
        ## upperbound caps x at threshold
        return threshold - torch.nn.functional.relu(threshold-x)
