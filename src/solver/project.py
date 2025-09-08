import torch
from typing import Dict, List, Optional, Tuple, Any
from .nft_data import NFTMarketData, TraitProcessor


class NFTProject:
    """
    Handles preprocessing and management of NFT project data.
    
    Loads raw data, applies filtering, and creates market data for solvers.
    """
    
    def __init__(self, 
                 project_data: Dict, 
                 num_buyers: Optional[int] = None,
                 num_items: Optional[int] = None, 
                 project_name: str = "",
                 device: Optional[torch.device] = None):
        """
        Initialize NFT project from raw data.
        
        Args:
            project_data: Raw project data dictionary
            num_buyers: Optional limit on number of buyers (None = all)
            num_items: Optional limit on number of items (None = all)
            project_name: Name of the project
            device: Device to place tensors on
        """
        self.project_name = project_name
        self.device = device
        self.trait_system = project_data['trait_system']
        
        # Count trades for metrics
        self.num_trades = sum(len(ids) for ids in project_data['buyer_assets_ids'])
        
        # Extract and potentially limit the data
        asset_traits, item_counts = self._extract_item_data(
            project_data['asset_traits'],
            project_data['item_counts'],
            num_items
        )
        
        buyer_prefs, buyer_budgets = self._extract_buyer_data(
            project_data['buyer_assets_ids'],
            project_data['asset_traits'],
            project_data['buyer_budgets'],
            num_buyers
        )
        
        # Create market data
        self.market_data = TraitProcessor.create_market_data(
            project_name=project_name,
            trait_system=self.trait_system,
            asset_traits=asset_traits,
            item_counts=item_counts,
            buyer_preferences=buyer_prefs,
            buyer_budgets=buyer_budgets,
            device=device
        )
        
        # Set dimensions
        self.N = self.market_data.N
        self.M = self.market_data.M
    
    def _extract_item_data(self, 
                         asset_traits: List[List[str]], 
                         item_counts: List[int],
                         num_items: Optional[int]) -> Tuple[List[List[str]], List[int]]:
        """
        Extract and potentially limit item data.
        
        Args:
            asset_traits: List of trait lists for each NFT
            item_counts: List of item counts
            num_items: Optional limit on number of items
            
        Returns:
            Tuple of (asset_traits, item_counts)
        """
        if num_items is not None and num_items < len(asset_traits):
            return asset_traits[:num_items], item_counts[:num_items]
        else:
            return asset_traits, item_counts
    
    def _extract_buyer_data(self,
                          buyer_assets_ids: List[List[int]],
                          asset_traits: List[List[str]],
                          buyer_budgets: List[float],
                          num_buyers: Optional[int]) -> Tuple[List[List[List[str]]], List[float]]:
        """
        Extract and potentially limit buyer data.
        
        Args:
            buyer_assets_ids: List of asset IDs owned by each buyer
            asset_traits: List of trait lists for each NFT
            buyer_budgets: List of budgets for each buyer
            num_buyers: Optional limit on number of buyers
            
        Returns:
            Tuple of (buyer_preferences, buyer_budgets)
        """
        # Convert asset IDs to actual traits
        buyer_preferences = []
        filtered_budgets = []
        
        # Apply min purchase filter if applicable
        min_purchase = self._get_min_purchase()
        
        for i, (assets, budget) in enumerate(zip(buyer_assets_ids, buyer_budgets)):
            # Skip buyers with too few purchases if needed
            if min_purchase > 0 and len(assets) < min_purchase:
                continue
                
            # Skip if we've reached the buyer limit
            if num_buyers is not None and len(buyer_preferences) >= num_buyers:
                break
                
            # Get traits for assets owned by this buyer
            buyer_traits = [asset_traits[aid] for aid in assets]
            buyer_preferences.append(buyer_traits)
            filtered_budgets.append(budget)
        
        return buyer_preferences, filtered_budgets
    
    def _get_min_purchase(self) -> int:
        """Get minimum purchase requirement for this project."""
        # Import here to avoid circular imports
        from arguments import nft_project_names, min_purchase
        
        # Check if this project has a min purchase requirement
        if hasattr(self, 'project_name') and self.project_name in nft_project_names:
            idx = nft_project_names.index(self.project_name)
            if idx < len(min_purchase):
                return min_purchase[idx]
        
        return 0