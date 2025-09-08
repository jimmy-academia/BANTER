import torch
from tqdm import tqdm
from typing import List, Optional, Union, Any, Tuple

from utils import make_batch_indexes


class UtilityCalculator:
    """
    Calculates buyer utilities in the NFT market.
    
    Encapsulates all utility calculations to simplify the solver classes.
    """
    
    def __init__(self, solver: Any):
        """Initialize with reference to parent solver."""
        self.solver = solver
        self.device = solver.device
        self.market = solver.market
    
    def calculate(self, user_indices, holdings, split=False):
        """
        Calculate all utility components for the given users.
        
        Args:
            user_indices: Indices of users to calculate for
            holdings: User holdings tensor
            split: Whether to return separate utility components
            
        Returns:
            Combined utility tensor or tensor of utility components if split=True
        """
        # Get user budgets
        budgets = self.market.buyer_budgets[user_indices]
        pricing = self.solver.pricing
        
        # Calculate individual utility components
        item_utility = self._item_utility(holdings, self.market.vj)
        collection_utility = self._collection_utility(holdings, user_indices)
        budget_utility = self._budget_utility(holdings, budgets, pricing)
        breeding_utility = self._breeding_utility(holdings, user_indices)
        
        # Update ratio for dynamic scheduling
        if hasattr(self.solver, 'ratio'):
            with torch.no_grad():
                item_sum = item_utility.sum().detach()
                breed_sum = breeding_utility.sum().detach()
                self.solver.ratio = item_sum / (breed_sum + 1e-5)
        
        # Return combined or split utilities
        if not split:
            return item_utility + collection_utility + breeding_utility + budget_utility
        else:
            return torch.stack([item_utility, collection_utility, breeding_utility, budget_utility]).T
    
    def _item_utility(self, holdings, vj):
        """Calculate utility from holding valuable items."""
        return (holdings * vj).sum(1) / 2
    
    def _collection_utility(self, holdings, user_indices, scaling_factor=50):
        """Calculate utility from collecting complementary traits."""
        # Process in smaller chunks to manage memory
        chunk_size = 32
        subtotals = []
        
        # Use nft_attributes directly
        nft_attributes = self.market.nft_attributes
        # Use buyer_preferences directly
        buyer_preferences = self.market.buyer_preferences
        
        # Convert generator to list
        batch_indexes = list(make_batch_indexes(len(holdings), chunk_size))
        for batch in batch_indexes:
            # Calculate trait coverage for each user's holdings
            trait_coverage = (holdings[batch].unsqueeze(2) * nft_attributes).sum(1)
            
            # Add 1 to avoid log(0)
            trait_coverage = trait_coverage + 1
            
            subtotals.append(trait_coverage)
            
        # Combine results
        trait_coverage = torch.cat(subtotals, dim=0)
        
        # Calculate weighted log utility based on preferences
        return (torch.log(trait_coverage) * buyer_preferences[user_indices]).sum(1) * scaling_factor
    
    def _budget_utility(self, holdings, budgets, pricing):
        """Calculate utility from remaining budget."""
        return budgets - (holdings * pricing).sum(1)
    
    def _breeding_utility(self, holdings, user_indices):
        """Calculate utility from breeding potential with differentiable selection."""
        breeding_type = self.solver.breeding_type
        
        if breeding_type == 'None' or breeding_type is None:
            return torch.zeros_like(holdings[:, 0])
        
        # Get parent NFTs and probabilities dynamically
        parents, expectations = self.solver.get_parent_nfts(user_indices, self.solver.pricing, holdings)
        
        # Calculate holding probability for each parent
        parent_probs = []
        for p in range(parents.shape[-1]):
            # Gather holdings for each parent
            parent_holdings = torch.gather(holdings, 1, parents[..., p])
            parent_probs.append(parent_holdings)
        
        # Average probability across parents
        probability = torch.mean(torch.stack(parent_probs), dim=0)
        
        # Apply differentiable top-k selection
        selection_mask = self.solver.differentiable_topk(
            probability, 
            self.solver.args.breeding_topk
        )
        
        # Calculate final breeding utility
        breeding_utility = (selection_mask * expectations).sum(1)
        
        # Scale based on breeding type
        scale = 10 if breeding_type == 'Heterogeneous' else 20
        return breeding_utility * scale


class DemandSolver:
    """
    Solves the user demand problem through gradient-based optimization.
    
    Finds optimal allocation of user budgets to maximize utility.
    """
    
    def __init__(self, solver: Any):
        """Initialize with reference to parent solver."""
        self.solver = solver
        self.device = solver.device
        self.market = solver.market
        self.utility = solver.utility
    
    def solve(self, user_indices=None, max_iterations=16, learning_rate=1e-4):
        """
        Find optimal demand through gradient optimization.
        
        Args:
            user_indices: Optional specific users to solve for
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient steps
            
        Returns:
            Holdings tensor representing optimal demand
        """
        # Get dimensions
        N, M = self.market.N, self.market.M
        
        # Initialize spending distribution (with budget reserve column)
        if self.solver.args.large:
            spending = torch.rand(N, M + 1)
        else:
            spending = torch.rand(N, M + 1).to(self.device)
        
        # Normalize spending proportions
        spending /= spending.sum(1, keepdim=True)
        
        # Determine batch iteration strategy
        if user_indices is not None:
            batch_iterator = [user_indices]
        else:
            # Process buyers in descending order of budget
            budgets = self.market.buyer_budgets
            sorted_users = budgets.argsort(descending=True).tolist()
            
            # Split into batches - use list instead of generator
            batch_size = max(N // 20, 1)  # Adaptive batch size
            # Convert generator to list for len() to work with tqdm
            batch_iterator = list(make_batch_indexes(sorted_users, batch_size))
        
        # Set solver to training mode for differentiable selection
        prev_training = self.solver.training
        self.solver.training = True
        
        # Optimization loop
        from tqdm import tqdm
        pbar = tqdm(range(max_iterations), ncols=88, desc='Solving Demand', leave=False)
        for _ in pbar:
            total_utility = 0
            
            for batch in tqdm(batch_iterator, ncols=88, leave=False):
                # Get spending variables for this batch
                spending_var = spending[batch]
                if self.solver.args.large:
                    spending_var = spending_var.to(self.device)
                
                # Enable gradient tracking
                spending_var.requires_grad = True
                
                # Calculate holdings from spending
                batch_budgets = self.market.buyer_budgets[batch]
                holdings = self._spending_to_holdings(spending_var, batch_budgets)
                
                # Calculate utility and gradient
                batch_utility = self.utility.calculate(batch, holdings)
                batch_utility.backward(torch.ones_like(batch_utility))
                
                # Update spending based on gradient
                _grad = spending_var.grad.cpu() if self.solver.args.large else spending_var.grad
                spending[batch] += learning_rate * _grad
                
                # Enforce constraints
                spending = torch.clamp(spending, min=0)
                spending /= spending.sum(1, keepdim=True)
                
                # Track progress
                total_utility += batch_utility.detach().mean().item()
            
            # Update progress bar
            pbar.set_postfix(utility=float(total_utility))
        
        # Restore original training mode
        self.solver.training = prev_training
        
        # Convert final spending to holdings
        pricing = self.solver.pricing
        budgets = self.market.buyer_budgets
        
        if self.solver.args.large:
            demand = self._spending_to_holdings(
                spending[:, :-1].to(self.device),
                budgets,
                pricing
            )
        else:
            demand = self._spending_to_holdings(
                spending[:, :-1],
                budgets,
                pricing
            )
        
        return demand
        
    def _spending_to_holdings(self, spending, budgets, pricing=None):
        """
        Convert spending proportions to holdings with capped ReLU.
        
        Args:
            spending: Spending proportion tensor
            budgets: Budget tensor
            pricing: Optional pricing tensor (uses solver.pricing if None)
            
        Returns:
            Holdings tensor
        """
        if pricing is None:
            pricing = self.solver.pricing

        if spending.size(1) > pricing.size(0):
            # The last column is the budget reserve
            spending = spending[:, :-1]
            
        # Calculate raw holdings
        holdings = spending * budgets.unsqueeze(1) / pricing.unsqueeze(0)
        
        # Apply capped ReLU (threshold=1)
        threshold = 1.0
        return threshold - torch.nn.functional.relu(threshold - holdings)