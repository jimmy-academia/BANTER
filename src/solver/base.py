import torch
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

from utils import loadj, make_batch_indexes
from .nft_data import NFTMarketData
from .project import NFTProject
from .utility import UtilityCalculator, DemandSolver
from .nftbreeding import NFTBreedingEngine


class BaseSolver:
    """Base class for NFT market solvers."""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.breeding_type = args.breeding_type
        
        self.project, self.market = self._load_market_data()
        
        self.utility = UtilityCalculator(self)
        self.demand = DemandSolver(self)
        
        if self.breeding_type != 'None':
            self.breeding = NFTBreedingEngine(self)
        else:
            self.breeding = None
        
        self.pricing = None
        self.holdings = None
        self.ratio = 1.0  # For dynamic scheduling
        self.training = True  # Differentiable vs hard selection
    
    def _load_market_data(self) -> Tuple[NFTProject, NFTMarketData]:
        cache_path = self._get_cache_path('project.pth')
        
        if cache_path.exists():
            project = torch.load(cache_path, weights_only=False)
            if self.device != project.device:
                project.market_data.to_device(self.device)
                project.device = self.device
        else:
            project_data = self._load_project_data()
            project = NFTProject(
                project_data,
                self.args.setN,
                self.args.setM,
                self.args.nft_project_name,
                self.device
            )
            project.market_data.calculate_matrices()
            torch.save(project, cache_path)
        
        return project, project.market_data
    
    def _load_project_data(self) -> Dict:
        return loadj(f'../NFT_data/clean/{self.args.nft_project_name}.json')
    
    def _get_cache_path(self, filename: str) -> Path:
        cache_dir = self.args.ckpt_dir / 'cache' / f'{self.args.nft_project_name}_N_{self.args.setN}_M_{self.args.setM}'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / filename
    
    def get_parent_nfts(self, user_indices=None, pricing=None, holdings=None):
        if self.breeding is None:
            raise ValueError("Breeding engine not initialized for breeding type 'None'")
        return self.breeding.get_parent_nfts(user_indices, pricing, holdings)
    
    def differentiable_topk(self, scores, k, temperature=0.1):
        """Differentiable approximation of top-k selection."""
        if self.training:
            values, _ = scores.topk(k, dim=1)
            kth_values = values[:, -1].unsqueeze(1)
            mask = torch.sigmoid((scores - kth_values) / temperature)
            return mask
        else:
            _, indices = scores.topk(k, dim=1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, indices, 1)
            return mask
    
    def solve(self):
        raise NotImplementedError("Subclasses must implement the solve method")
    
    def solve_demand(self, user_indices=None):
        return self.demand.solve(user_indices)
    
    def calculate_utilities(self, user_indices, holdings, split=False):
        return self.utility.calculate(user_indices, holdings, split)
    
    def evaluate(self):
        if self.pricing is None:
            raise ValueError("Pricing must be set before evaluation")
        
        self.pricing.clamp_(min=0)
        self.holdings = self.solve_demand()
        self._apply_constraints()
        
        # Calculate utilities in batches
        prev_training = self.training
        self.training = False
        
        self.buyer_utilities = self._batch_calculate_utilities(
            self.market.N, 100, split=True
        )
        
        self.training = prev_training
        
        self.seller_revenue = (self.holdings * self.pricing).sum()
        
        return {
            'seller_revenue': self.seller_revenue,
            'buyer_utilities': self.buyer_utilities,
            'holdings': self.holdings,
            'pricing': self.pricing
        }
    
    def _batch_calculate_utilities(self, total_users, batch_size, split=False):
        """Calculate utilities for all users in batches."""
        
        utilities = []
        for batch in make_batch_indexes(total_users, batch_size):
            batch_utilities = self.calculate_utilities(
                batch, self.holdings[batch], split
            )
            utilities.append(batch_utilities)
        
        return torch.cat(utilities, dim=0)
        
    def _apply_constraints(self):
        
        epsilon = 1e-6
        self.holdings.clamp_(min=0)
        
        # NFT availability constraint
        self.holdings *= torch.clamp(
            self.market.nft_counts / (self.holdings.sum(0) + epsilon), max=1
        )
        
        # Budget constraints in batches
        for batch in make_batch_indexes(self.market.N, 1000):
            batch_holdings = self.holdings[batch]
            batch_spending = (batch_holdings * self.pricing).sum(1)
            budget_ratio = torch.clamp(
                self.market.buyer_budgets[batch] / (batch_spending + epsilon), max=1
            ).view(-1, 1)
            self.holdings[batch] *= budget_ratio

    def greedy_init_pricing(self):
        """Initialize pricing based on objective values."""
        pricing = self.market.vj / self.market.vj.mean() * (
            self.market.buyer_budgets.sum() / self.market.nft_counts.sum()
        )
        return pricing
    
    def count_results(self):
        """Calculate final results after solving."""
        if self.holdings is None or self.pricing is None:
            raise ValueError("Must solve before counting results")
        
        self.training = False  # Use hard selection for final evaluation
        
        # Calculate utilities for all buyers
        self.buyer_utilities = self._batch_calculate_utilities(
            self.market.N, 100, split=True
        )
        
        # Calculate revenue
        self.seller_revenue = (self.holdings * self.pricing).sum()
        
        # Calculate utility components for analysis
        utility_sum = self.buyer_utilities.sum(0)
        self.utility_component = {
            'item': utility_sum[0].item(),
            'collection': utility_sum[1].item(), 
            'breeding': utility_sum[2].item(),
            'budget': utility_sum[3].item()
        }
