import torch
import math
from tqdm import tqdm
from typing import Tuple, Optional, Any
from pathlib import Path


class NFTBreedingEngine:
    """Handles all breeding-related functionality for NFT market simulation."""
    
    def __init__(self, solver: Any):
        self.solver = solver
        self.device = solver.device
        self.market = solver.market
        self.project = solver.project
        self.args = solver.args
        self.breeding_type = solver.breeding_type
        self.parent_cache = {}
        
        if self.breeding_type != 'None':
            self._setup_breeding_data()
    
    def _setup_breeding_data(self):
        if self.breeding_type == 'Heterogeneous':
            self._setup_heterogeneous_data()
    
    def _setup_heterogeneous_data(self):
        cache_path = self._get_cache_path(
            f'heter_files_{self.args.num_trait_div}_{self.args.num_attr_class}_'
            f'{self.market.N}_{self.market.M}.pth'
        )
        
        if cache_path.exists():
            from utils import torch_cleanload
            self.trait_divisions, self.attribute_classes, self.buyer_types = torch_cleanload(
                cache_path, self.device
            )
        else:
            self.trait_divisions = torch.randint(
                self.args.num_trait_div, (self.market.M,)
            ).to(self.device)
            self.attribute_classes = torch.randint(
                self.args.num_attr_class, (self.market.M,)
            ).to(self.device)
            self.buyer_types = torch.randint(
                2, (self.market.N,)
            ).to(self.device)
            
            from utils import torch_cleansave
            torch_cleansave(
                (self.trait_divisions, self.attribute_classes, self.buyer_types),
                cache_path
            )
    
    def _get_cache_path(self, filename: str) -> Path:
        cache_dir = self.args.ckpt_dir / 'cache' / f'{self.args.nft_project_name}_N_{self.args.setN}_M_{self.args.setM}'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / filename
    
    def _get_price_factor(self, pricing):
        return torch.exp(-pricing / pricing.mean())
    
    def get_parent_nfts(self, user_indices=None, pricing=None, holdings=None):
        if pricing is None:
            pricing = self.solver.pricing
        if pricing is None:
            raise ValueError("Pricing must be set before getting parent NFTs")
            
        # Cache based on pricing sum to avoid storing large tensors
        price_hash = str(torch.sum(pricing).item())
        
        if price_hash in self.parent_cache:
            parents, expectations = self.parent_cache[price_hash]
            if user_indices is not None:
                return parents[user_indices], expectations[user_indices]
            return parents, expectations
            
        if self.breeding_type == 'Heterogeneous':
            parents, expectations = self._get_heterogeneous_parents(pricing, holdings)
        else:
            parents, expectations = self._get_homogeneous_parents(pricing, holdings)
            
        self.parent_cache[price_hash] = (parents, expectations)
        
        if user_indices is not None:
            return parents[user_indices], expectations[user_indices]
        return parents, expectations
    
    def clear_cache(self):
        self.parent_cache = {}
    
    def _get_heterogeneous_parents(self, pricing, holdings):
        candidates = self._get_heterogeneous_candidates(pricing, holdings)
        parent_sets = self._assemble_heterogeneous_pairs(candidates)
        expectations = self._calculate_heterogeneous_expectations(parent_sets, pricing)
        return self._sort_by_expectations(parent_sets, expectations)
    
    def _get_candidates(self, pricing, breeding_type):
        price_factor = self._get_price_factor(pricing)
        base_scores = self.market.uij * self.market.vj * price_factor.unsqueeze(0)
        
        # Add population factor for homogeneous breeding
        if breeding_type == 'Homogeneous' and self.args.module_id in [0, 3]:
            attributes = self.market.nft_attributes
            attr_freq = attributes.float().sum(0)
            population_factor = (attributes * attr_freq).sum(1)
            base_scores *= population_factor
        
        if self.args.module_id == 0:
            return base_scores.topk(self.args.cand_lim)[1]
        elif self.args.module_id == 1:
            scores = self.market.vj * price_factor
            return torch.stack(
                [scores.topk(self.args.cand_lim)[1] for _ in range(self.market.N)]
            ).to(self.device)
        elif self.args.module_id == 2:
            if breeding_type == 'Heterogeneous':
                mask = (torch.rand(self.market.N, self.args.cand_lim) > 0.5).long().to(self.device)
                return (
                    mask * base_scores.topk(self.args.cand_lim)[1] + 
                    (1 - mask) * base_scores.topk(self.args.cand_lim, largest=False)[1]
                )
            else:
                return torch.stack(
                    [torch.randperm(self.market.M)[:self.args.cand_lim] for _ in range(self.market.N)]
                ).to(self.device)
        elif self.args.module_id == 3:
            return base_scores.topk(self.args.cand_lim, largest=False)[1]
        else:
            raise ValueError(f"Invalid module_id: {self.args.module_id}")
    
    def _get_heterogeneous_candidates(self, pricing, holdings):
        return self._get_candidates(pricing, 'Heterogeneous')
    
    def _get_homogeneous_parents(self, pricing, holdings):
        candidates = self._get_candidates(pricing, self.breeding_type)
        parent_sets = []
        expectations = []
        chunk_size = 32
        
        from utils import make_batch_indexes
        for batch_idx in make_batch_indexes(candidates.size(0), chunk_size):
            batch_pairs = self._create_pairs(candidates[batch_idx])
            batch_expectations = self._calculate_homogeneous_expectations(
                batch_idx, batch_pairs, pricing
            )
            parent_sets.append(batch_pairs)
            expectations.append(batch_expectations)
        
        parent_sets = torch.cat(parent_sets)
        expectations = torch.cat(expectations)
        return self._sort_by_expectations(parent_sets, expectations)
    
    def _get_homogeneous_candidates(self, pricing, holdings):
        return self._get_candidates(pricing, self.breeding_type)
    
    def _assemble_heterogeneous_pairs(self, candidates):
        parent_sets = []
        
        for buyer_candidates in tqdm(candidates, ncols=88, desc='Assembling Pairs', leave=False):
            candidate_divisions = self.trait_divisions[buyer_candidates]
            
            # Ensure diversity
            if all(candidate_divisions == candidate_divisions[0]):
                candidate_divisions[:len(candidate_divisions) // 2] = 1 - candidate_divisions[0]
            
            unique_divisions = candidate_divisions.unique(sorted=True)
            division_groups = [
                (candidate_divisions == div).nonzero(as_tuple=True)[0]
                for div in unique_divisions
            ]
            
            rank_lists = [torch.arange(len(group)) for group in division_groups]
            pair_indices = torch.cartesian_prod(*division_groups)
            rank_combinations = torch.cartesian_prod(*rank_lists)
            
            sort_order = rank_combinations.sum(-1).argsort()
            sorted_pairs = pair_indices[sort_order]
            buyer_pairs = buyer_candidates[sorted_pairs]
            parent_sets.append(buyer_pairs)
        
        min_len = min(len(pairs) for pairs in parent_sets)
        parent_sets = [pairs[:min_len] for pairs in parent_sets]
        return torch.stack(parent_sets)
    
    def _create_pairs(self, candidates):
        idx = torch.combinations(
            torch.arange(candidates.size(1), device=self.device), 2
        )
        idx = idx.unsqueeze(0).repeat(candidates.size(0), 1, 1)
        pairs = torch.gather(
            candidates.unsqueeze(1).repeat(1, idx.size(1), 1), 
            2, 
            idx
        )
        return pairs
    
    def _calculate_heterogeneous_expectations(self, parent_sets, pricing):
        price_factor = self._get_price_factor(pricing).unsqueeze(0)
        expectations = (
            self.market.uij * self.market.vj * price_factor
        ).unsqueeze(1).expand(
            -1, parent_sets.size(1), -1
        ).gather(2, parent_sets).sum(-1)
        
        if self.args.module_id == 0:
            niche_ids = (self.buyer_types == 0).nonzero(as_tuple=True)[0]
            eclectic_ids = (self.buyer_types == 1).nonzero(as_tuple=True)[0]
            
            class_factors = torch.zeros(
                parent_sets.shape[:2], dtype=torch.long, device=self.device
            )
            
            if len(niche_ids) > 0:
                self._process_niche_buyers(niche_ids, parent_sets, class_factors)
            if len(eclectic_ids) > 0:
                self._process_eclectic_buyers(eclectic_ids, parent_sets, class_factors)
            
            expectations *= class_factors
        
        return expectations
    
    def _process_niche_buyers(self, niche_ids, parent_sets, class_factors):
        niche_sets = parent_sets[niche_ids]
        labeled_sets = self.attribute_classes[niche_sets]
        majority_label = labeled_sets.mode(dim=-1)[0].unsqueeze(-1).expand_as(labeled_sets)
        same_class_count = (labeled_sets == majority_label).sum(-1)
        class_factors[niche_ids] = same_class_count
    
    def _process_eclectic_buyers(self, eclectic_ids, parent_sets, class_factors):
        eclectic_sets = parent_sets[eclectic_ids]
        labeled_sets = self.attribute_classes[eclectic_sets]
        
        # Hash map for diversity scoring
        hash_map = torch.cartesian_prod(
            *[torch.arange(self.args.num_attr_class)] * self.args.num_trait_div
        ).to(self.device)
        hash_map = hash_map.sort()[0].unique(dim=0)
        num_unique = torch.LongTensor(
            [len(hash.unique()) for hash in hash_map]
        ).to(self.device)
        
        query = labeled_sets.view(-1, self.args.num_trait_div).sort(-1)[0]
        matching_indices = torch.full(
            (query.size(0),), -1, dtype=torch.long, device=self.device
        )
        
        for i, hash in enumerate(hash_map):
            matches = (query == hash).all(dim=1)
            matching_indices[matches] = i
            
        div_class_count = num_unique[matching_indices].view(labeled_sets.shape[:2])
        class_factors[eclectic_ids] = div_class_count
    
    def _calculate_homogeneous_expectations(self, buyer_indices, parent_pairs, pricing):
        expectations = torch.zeros(parent_pairs.shape[:2]).to(self.device)
        parent_attributes = self.market.nft_attributes[parent_pairs]
        num_traits = self.market.num_traits
        max_options = self.market.max_options
        price_factor_mean = self._get_price_factor(pricing).mean()
        
        # Monte Carlo sampling
        for _ in tqdm(range(self.args.num_child_sample), ncols=88, desc='Sampling Children', leave=False):
            child_attr = self._generate_child(parent_attributes, num_traits, max_options)
            
            child_utility = (
                self.market.buyer_preferences[buyer_indices].unsqueeze(1) * child_attr
            ).sum(-1)
            child_value = self._calculate_child_value(child_attr)
            expectations += (child_utility * child_value * price_factor_mean)
        
        return expectations / self.args.num_child_sample
    
    def _generate_child(self, parent_attributes, num_traits, max_options):
        _shape = (*parent_attributes.shape[:2], num_traits)
        
        if self.breeding_type == 'Homogeneous':
            # 50/50 inheritance from each parent
            trait_mask = torch.randint(0, 2, _shape).to(self.device)
        else:  # Child Project with mutation
            r = self.args.mutation_rate
            trait_mask = torch.multinomial(
                torch.tensor([(1-r)/2, (1-r)/2, r]), 
                math.prod(_shape), 
                replacement=True
            ).view(_shape).to(self.device)
        
        inheritance_mask = trait_mask.repeat_interleave(max_options, dim=2)
        child_attr = (
            (inheritance_mask == 0).float() * parent_attributes[:, :, 0, :] + 
            (inheritance_mask == 1).float() * parent_attributes[:, :, 1, :]
        )
        
        # Add mutations for Child Project
        if self.breeding_type == 'ChildProject':
            mutations = self._generate_random_traits(_shape[:-1], num_traits, max_options)
            child_attr += (inheritance_mask == 2).float() * mutations
            
        return child_attr
    
    def _generate_random_traits(self, batch_shape, num_traits, max_options):
        indices = []
        for i, options in enumerate(self.project.trait_system.values()):
            trait_indices = torch.randint(
                0, len(options), 
                (math.prod(batch_shape),)
            ) + i * max_options
            indices.append(trait_indices)
        
        indices = torch.stack(indices).T.to(self.device)
        traits = torch.zeros(
            math.prod(batch_shape), num_traits * max_options
        ).to(self.device)
        traits.scatter_(1, indices, 1)
        return traits.view(*batch_shape, -1)
    
    def _calculate_child_value(self, child_attr):
        flat_attr = child_attr.view(-1, child_attr.size(-1))
        attr_rarity = flat_attr * self.market.trait_counts
        attr_rarity = torch.where(
            attr_rarity != 0,
            attr_rarity,
            torch.ones_like(attr_rarity)
        )
        values = torch.log(
            self.market.nft_counts.sum() / attr_rarity
        ).sum(1)
        alpha = self.market.buyer_budgets.sum() / values.sum()
        values = values * alpha
        return values.view(child_attr.shape[:2])
    
    def _sort_by_expectations(self, parent_sets, expectations):
        sorted_indices = expectations.argsort(descending=True)
        ranked_parents = torch.gather(
            parent_sets, 
            1, 
            sorted_indices.unsqueeze(-1).expand(-1, -1, parent_sets.size(-1))
        )
        ranked_expectations = torch.gather(expectations, 1, sorted_indices)
        return ranked_parents, ranked_expectations
    
    def get_breeding_data(self, data_type):
        """Get heterogeneous breeding data by type."""
        if self.breeding_type != 'Heterogeneous':
            return None
        
        data_map = {
            'trait_divisions': getattr(self, 'trait_divisions', None),
            'attribute_classes': getattr(self, 'attribute_classes', None),
            'buyer_types': getattr(self, 'buyer_types', None)
        }
        return data_map.get(data_type)
    
    def get_trait_divisions(self):
        return self.get_breeding_data('trait_divisions')
    
    def get_attribute_classes(self):
        return self.get_breeding_data('attribute_classes')
    
    def get_buyer_types(self):
        return self.get_breeding_data('buyer_types')