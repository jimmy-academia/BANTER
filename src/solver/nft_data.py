import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class NFTMarketData:
    """
    Core data structure for NFT market simulation.
    
    Provides a clean, centralized representation of all market data, including
    NFT attributes, buyer preferences, and calculated market matrices.
    """
    # Basic information
    project_name: str
    trait_system: Dict[str, List[str]]
    
    # NFT data
    nft_attributes: torch.Tensor  # One-hot encoded attributes
    nft_counts: torch.Tensor      # Availability of each NFT
    
    # Buyer data
    buyer_preferences: torch.Tensor  # Normalized preferences 
    buyer_budgets: torch.Tensor      # Available budget for each buyer
    
    # Calculated matrices
    uij: Optional[torch.Tensor] = None  # Buyer-item utility matrix
    vj: Optional[torch.Tensor] = None   # Item objective value vector
    
    # Cached trait counts (for rarity calculation)
    trait_counts: Optional[torch.Tensor] = None
    
    # Dimensions
    num_traits: int = field(init=False)
    max_options: int = field(init=False)
    N: int = field(init=False)  # Number of buyers
    M: int = field(init=False)  # Number of NFTs
    
    def __post_init__(self):
        """Set derived properties from data."""
        # Dimensions
        self.num_traits = len(self.trait_system)
        self.max_options = max(len(options) for options in self.trait_system.values())
        self.N = len(self.buyer_budgets)
        self.M = len(self.nft_counts)
        
        # Calculate trait counts if not provided
        if self.trait_counts is None:
            self.trait_counts = self._calculate_trait_counts()
    
    def _calculate_trait_counts(self) -> torch.Tensor:
        """Calculate frequency of each trait across all NFTs."""
        # Weighted count of each trait
        trait_counts = (self.nft_attributes * self.nft_counts.unsqueeze(1)).sum(0)
        
        # Ensure no zero counts (to avoid division by zero)
        device = self.nft_attributes.device
        return torch.where(
            trait_counts == 0, 
            torch.tensor(1.0, device=device), 
            trait_counts
        )
    
    def to_device(self, device) -> 'NFTMarketData':
        """Move all tensors to the specified device."""
        self.nft_attributes = self.nft_attributes.to(device)
        self.nft_counts = self.nft_counts.to(device)
        self.buyer_preferences = self.buyer_preferences.to(device)
        self.buyer_budgets = self.buyer_budgets.to(device)
        self.trait_counts = self.trait_counts.to(device)
        
        if self.uij is not None:
            self.uij = self.uij.to(device)
            
        if self.vj is not None:
            self.vj = self.vj.to(device)
            
        return self
    
    def calculate_matrices(self) -> 'NFTMarketData':
        """Calculate utility and value matrices."""
        # Calculate objective values (Vj)
        self.vj = self._calculate_objective_values()
        
        # Calculate utility matrix (Uij)
        self.uij = torch.matmul(
            self.buyer_preferences,
            self.nft_attributes.T.float()
        )
        
        return self
    
    def _calculate_objective_values(self) -> torch.Tensor:
        """Calculate objective values based on rarity."""
        # Extract non-zero attributes for rarity calculation
        attr_rarity = self.nft_attributes * self.trait_counts
        mask = attr_rarity != 0
        
        # Set ones for zero values to avoid nan
        attr_rarity = torch.where(
            mask, 
            attr_rarity, 
            torch.ones_like(attr_rarity)
        )
        
        # Reshape for trait-wise calculation
        attr_rarity = attr_rarity.view(self.M, -1)
        
        # Calculate log rarity (higher for rare traits)
        objective_values = torch.log(self.nft_counts.sum() / attr_rarity).sum(1)
        
        # Scale to budget scale
        alpha = self.buyer_budgets.sum() / objective_values.sum()
        
        return objective_values * alpha


class TraitProcessor:
    """
    Processes NFT traits for market simulation.
    
    Handles conversion between raw traits and tensor representations.
    """
    
    @staticmethod
    def create_market_data(
        project_name: str,
        trait_system: Dict[str, List[str]],
        asset_traits: List[List[str]],
        item_counts: List[int],
        buyer_preferences: List[List[List[str]]],
        buyer_budgets: List[float],
        device: Optional[torch.device] = None
    ) -> NFTMarketData:
        """
        Create NFTMarketData from raw inputs.
        
        Args:
            project_name: Name of the NFT project
            trait_system: Dictionary mapping trait names to possible values
            asset_traits: List of trait lists for each NFT
            item_counts: Availability count for each NFT
            buyer_preferences: List of preference trait lists for each buyer
            buyer_budgets: Budget for each buyer
            device: Device to place tensors on
            
        Returns:
            NFTMarketData object with tensor representations
        """
        # Convert traits to tensor representation
        nft_attributes = TraitProcessor.encode_traits(
            trait_system, asset_traits, device
        )
        
        # Convert counts to tensor
        nft_counts = torch.tensor(item_counts, device=device)
        
        # Process buyer preferences (more complex)
        buyer_prefs_tensor = TraitProcessor.encode_preferences(
            trait_system, buyer_preferences, device
        )
        
        # Normalize budgets
        buyer_budgets_tensor = TraitProcessor.normalize_budgets(
            buyer_budgets, device
        )
        
        # Create market data
        market_data = NFTMarketData(
            project_name=project_name,
            trait_system=trait_system,
            nft_attributes=nft_attributes,
            nft_counts=nft_counts,
            buyer_preferences=buyer_prefs_tensor,
            buyer_budgets=buyer_budgets_tensor
        )
        
        # Calculate market matrices
        market_data.calculate_matrices()
        
        return market_data
    
    @staticmethod
    def encode_traits(
        trait_system: Dict[str, List[str]],
        traits_list: List[List[str]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode traits into tensor representation.
        
        Args:
            trait_system: Dictionary mapping trait names to possible values
            traits_list: List of trait lists for each item
            device: Device to place tensor on
            
        Returns:
            One-hot encoded tensor representation
        """
        # Get system dimensions
        num_traits = len(trait_system)
        num_options = [len(options) for options in trait_system.values()]
        max_options = max(num_options)
        
        # Convert traits to indices
        indices = []
        for traits in traits_list:
            item_indices = []
            for (trait_name, options), choice in zip(trait_system.items(), traits):
                # Handle 'none' value
                choice = 'none' if choice == 'None' else choice
                
                try:
                    item_indices.append(options.index(choice))
                except ValueError:
                    # Default to first option if not found
                    item_indices.append(0)
            
            indices.append(item_indices)
        
        # Convert to tensor
        indices_tensor = torch.LongTensor(indices)
        if device:
            indices_tensor = indices_tensor.to(device)
            
        # Reshape for scatter
        indices_tensor = indices_tensor.unsqueeze(2)
        
        # Create one-hot encoding
        result = torch.zeros(len(traits_list), num_traits, max_options)
        if device:
            result = result.to(device)
            
        # Fill with one-hot values
        result.scatter_(2, indices_tensor, 1)
        
        # Flatten trait dimension
        return result.view(len(traits_list), -1)
    
    @staticmethod
    def encode_preferences(
        trait_system: Dict[str, List[str]],
        buyer_preferences: List[List[List[str]]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode buyer preferences from owned assets.
        
        Args:
            trait_system: Dictionary mapping trait names to possible values
            buyer_preferences: List of preference trait lists for each buyer
            device: Device to place tensor on
            
        Returns:
            Normalized preferences tensor
        """
        preferences_list = []
        
        for buyer_traits in buyer_preferences:
            # Encode all assets owned by this buyer
            encoded_traits = [
                TraitProcessor.encode_traits(trait_system, [traits], device).squeeze(0)
                for traits in buyer_traits
            ]
            
            # Average across all owned assets
            if encoded_traits:
                buyer_pref = torch.stack(encoded_traits).mean(dim=0)
            else:
                # Default preferences if no assets
                num_traits = len(trait_system)
                max_options = max(len(options) for options in trait_system.values())
                buyer_pref = torch.zeros(num_traits * max_options, device=device)
            
            preferences_list.append(buyer_pref)
        
        # Stack all buyer preferences
        preferences_tensor = torch.stack(preferences_list)
        
        # Apply softmax normalization
        preferences_tensor = torch.softmax(preferences_tensor, dim=1)
        
        return preferences_tensor
    
    @staticmethod
    def normalize_budgets(
        budgets: List[float],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Normalize buyer budgets to a standard range using log transformation.
        
        Args:
            budgets: List of raw budgets
            device: Device to place tensor on
            
        Returns:
            Normalized budget tensor [10, 100]
        """
        # Convert to float64 to handle potentially large values
        budget_tensor = torch.tensor(budgets, device=device, dtype=torch.float64)
        
        # Ensure non-negative
        budget_tensor.clamp_(min=1e-6)  # Small positive value to avoid log(0)
        
        # Apply log transformation (better for exponentially distributed data)
        budget_tensor = torch.log10(budget_tensor)
        
        # Now scale the log values to [10, 100] range
        if len(budget_tensor) > 0:
            if budget_tensor.max() > budget_tensor.min():
                # Scale log values to [10, 100]
                budget_tensor = (
                    (budget_tensor - budget_tensor.min()) / 
                    (budget_tensor.max() - budget_tensor.min()) * 90 + 10
                )
            else:
                # Default if all budgets are equal
                budget_tensor.fill_(50)
        
        # Convert back to float32 for compatibility
        return budget_tensor.to(torch.float32)