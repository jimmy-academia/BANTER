import torch
from .base import BaseSolver

class HeuristicsSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        self.k = 20

    def initial_assignment(self):
        raise NotImplementedError
    
    def solve(self):
        # random or popular recommendation of NFT to buyers
        _assignments = self.initial_assignment()

        self.holdings = torch.zeros(self.nftP.N, self.nftP.M)
        self.holdings[torch.arange(self.nftP.N)[:, None], _assignments] = 1
        
        budget_per_item = self.buyer_budgets.cpu() / self.k
        buyer_spendings = self.holdings * budget_per_item.unsqueeze(1)
        self.pricing = buyer_spendings.sum(0)/self.nft_counts.cpu()
        self.pricing.clamp_(0.1)
        self.pricing = self.pricing.to(self.args.device)

class RandomSolver(HeuristicsSolver):
    def __init__(self, args):
        super().__init__(args)
    def initial_assignment(self):
        random_assignments = torch.stack([torch.randperm(self.nftP.M)[:1] for _ in range(self.nftP.N)]).to(self.args.device)
        return random_assignments

class PopularSolver(HeuristicsSolver):
    def __init__(self, args):
        super().__init__(args)

    def initial_assignment(self):
        favorite_assignments = (self.Uij * self.Vj).topk(self.k)[1]  #shape N, k
        return favorite_assignments
