import time
import random
from tqdm import tqdm

from .greedy import BaselineSolver
from utils import *

# N buyer M instance
from debug import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# self.add_time

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_hidden_layers=[64, 32, 16, 8]):
        super(NCFModel, self).__init__()
        
        # GMF part (Generalized Matrix Factorization)
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP part (Multi-Layer Perceptron)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        mlp_layers = []
        input_size = embedding_dim * 2
        for hidden_units in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_units))
            mlp_layers.append(nn.ReLU())
            input_size = hidden_units
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Concatenate GMF and MLP outputs
        self.fc = nn.Linear(mlp_hidden_layers[-1] + embedding_dim, 1)
        
    def forward(self, user_indices, item_indices):
        # GMF forward pass
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_embedding_gmf * item_embedding_gmf  # Element-wise product
        
        # MLP forward pass
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # Concatenate
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP output
        final_input = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = torch.sigmoid(self.fc(final_input))
        
        return prediction

class NCFSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if not args.large:
            # load afterwards modifications for args.large
            self.do_preparations()

    def do_preparations(self):
        self.model = NCFModel(self.market.N, self.market.M)
        self.model.to(self.args.device)

    def initial_assignment(self):
        N, M = self.market.uij.shape
        assert N == self.market.N 
        assert M == self.market.M
        user_indices = torch.randperm(N)
        item_indices = torch.randperm(M)

        # Create all combinations of user-item pairs
        # user_indices = user_indices.repeat_interleave(M)  
        # item_indices = item_indices.repeat(N) 
        # targets = self.market.uij[user_indices, item_indices]
        # dataset = TensorDataset(user_indices, item_indices, targets)
        # self.dataloader = DataLoader(dataset, batch_size=10240, shuffle=True)

        cache_path = self._get_cache_path("")/f'NCF.pth'

        if self.args.no_method_cache:
            self.prepare_dataset()
            self.train_model()
        else:
            if not cache_path.exists():
                start = time.time()
                self.prepare_dataset()
                self.train_model()
                runtime = time.time() - start
                torch.save({'runtime':runtime, 'weight': self.model.cpu().state_dict()}, cache_path)
                self.model.to(self.args.device)
            else:
                data = torch.load(cache_path, weights_only=True)
                self.add_time += data.get('runtime')
                self.model.load_state_dict(data.get('weight'))
                self.model.to(self.args.device)

        item_indices = torch.arange(M).to(self.args.device)

        _assignment = []
        for i in range(N):
            i = torch.LongTensor([i]).to(self.args.device)
            i = i.repeat(M)
            prediction = self.model(i, item_indices).squeeze()
            __, topk_indices = prediction.topk(self.k)
            _assignment.append(topk_indices)

        _assignment = torch.stack(_assignment)
        return _assignment

    def prepare_dataset(self):
        # Select positive edges using the top-k values from self.market.uij
        N, M = self.market.uij.shape
        k = 64  # Number of positive samples per user (you can adjust this)
        
        user_indices = torch.arange(N)
        item_indices = torch.arange(M)

        pos_user_indices = []
        pos_item_indices = []
        neg_user_indices = []
        neg_item_indices = []

        for user in range(N):
            # Get top-k positive items for this user
            _, topk_item_indices = torch.topk(self.market.uij[user], k)
            pos_user_indices.append(user * torch.ones(k, dtype=torch.long))
            pos_item_indices.append(topk_item_indices)

            # Negative sampling: randomly select items that are not in the positive set
            all_items = torch.arange(M)
            neg_items = torch.tensor(list(set(all_items.tolist()) - set(topk_item_indices.tolist())))
            neg_sample = neg_items[torch.randint(0, neg_items.size(0), (k,))]
            neg_user_indices.append(user * torch.ones(k, dtype=torch.long))
            neg_item_indices.append(neg_sample)

        # Flatten and combine positive and negative samples
        pos_user_indices = torch.cat(pos_user_indices).to(self.args.device)
        pos_item_indices = torch.cat(pos_item_indices).to(self.args.device)
        neg_user_indices = torch.cat(neg_user_indices).to(self.args.device)
        neg_item_indices = torch.cat(neg_item_indices).to(self.args.device)

        # Create targets: 1 for positive edges, 0 for negative edges
        pos_targets = torch.ones_like(pos_user_indices, dtype=torch.float32)
        neg_targets = torch.zeros_like(neg_user_indices, dtype=torch.float32)

        # Combine positive and negative samples
        user_indices = torch.cat([pos_user_indices, neg_user_indices])
        item_indices = torch.cat([pos_item_indices, neg_item_indices])
        targets = torch.cat([pos_targets, neg_targets])

        # Create the TensorDataset and DataLoader
        dataset = TensorDataset(user_indices, item_indices, targets)
        self.dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    def train_model(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()  # Binary cross-entropy loss for positive/negative classification

        # Training loop
        if self.args.large or self.args.task == 'yelp':
            num_epochs = 4    
        else:
            num_epochs = 64
        pbar = tqdm(range(num_epochs), ncols=90, desc='NCF training')
        for epoch in pbar:
            for batch in tqdm(self.dataloader, ncols=90, desc='iter', leave=False):
                user_batch, item_batch, target_batch = batch
                user_batch, item_batch, target_batch = user_batch.to(self.args.device), item_batch.to(self.args.device), target_batch.to(self.args.device)

                # Forward pass
                predictions = self.model(user_batch, item_batch).squeeze()
                
                # Compute loss
                loss = criterion(predictions, target_batch)
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
