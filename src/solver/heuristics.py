import torch
from .base import BaseSolver

class HeuristicsSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        self.k = 20

    def initial_assignment(self):
        raise NotImplementedError
    
    def solve(self, set_pricing=None):
        # random or popular recommendation of NFT to buyers
        _assignments = self.initial_assignment()

        self.holdings = torch.zeros(self.nftP.N, self.nftP.M).to(self.args.device)
        self.holdings[torch.arange(self.nftP.N)[:, None], _assignments] = 1
        
        # all_values = torch.arange(self.nftP.M).to(self.args.device)
        # flat_assignments = _assignments.flatten()
        # zero_tensor = torch.zeros(self.nftP.M, dtype=torch.bool, device=self.args.device)
        # zero_tensor.index_fill_(0, flat_assignments, True)
        # unassigned = all_values[~zero_tensor]

        # unassigned_indices = torch.randint(high=self.nftP.N, size=(len(unassigned),))
        # self.holdings[unassigned_indices, unassigned] = 1

        if set_pricing is None:
            budget_per_item = self.buyer_budgets / self.k
            buyer_spendings = self.holdings * budget_per_item.unsqueeze(1)
            self.pricing = buyer_spendings.sum(0)/self.nft_counts
            self.pricing.clamp_(0.1)
            # self.pricing[unassigned] = 1e-3
            # self.pricing = torch.softmax(self.pricing)
            # print(self.pricing)
            # self.holdings = buyer_spendings/self.pricing * (1-1e-4)
        else:
            self.pricing = set_pricing
            fullfil_ratio = self.buyer_budgets / (self.holdings * self.pricing).sum(1)
            fullfil_ratio.clamp_(0, 1)
            # self.holdings = self.holdings * fullfil_ratio.unsqueeze(1)
        # assert all(self.holdings.sum(0) <= self.nft_counts) ## item constraint
        # assert all((self.holdings*self.pricing).sum(1) <= self.buyer_budgets) ## budget constraint

class RandomSolver(HeuristicsSolver):
    def __init__(self, args):
        super().__init__(args)
    def initial_assignment(self):
        random_assignments = torch.stack([torch.randperm(self.nftP.M)[:1] for _ in range(self.nftP.N)]).to(self.args.device)
        return random_assignments

class PopularSolver(HeuristicsSolver):
# class PopularSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

    def initial_assignment(self):
        # spending = self.Uij
        # spending = spending/spending.sum(1).unsqueeze(1)
        # self.pricing = (spending * self.buyer_budgets.unsqueeze(1)).sum(0) / self.nft_counts

        favorite_assignments = (self.Uij * self.Vj).topk(self.k)[1]  #shape N, k
        return favorite_assignments
