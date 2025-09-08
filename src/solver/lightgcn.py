import time
import random
import torch
from torch_geometric.nn import LightGCN

from tqdm import tqdm

from .greedy import BaselineSolver

from debug import *
from utils import *

# next: NCF

# N buyer M instance

class LightGCNSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)
        self.cache_index_path = self._get_cache_path("")/'lightgcn_edge_index.pth'

        if not args.large:
            self.do_preparations()
            
    def do_preparations(self):
        self.model = LightGCN(self.market.N+self.market.M, 64, 5)
        self.model.to(self.args.device)

    def initial_assignment(self):

        cache_path = self._get_cache_path("")/f'LightGCN.pth'
        if self.args.no_method_cache:
            self.edge_index, self.neg_edge_index = self.prepare_data()
            self.train_model()
        else:
            if not self.cache_index_path.exists():
                start = time.time()
                edge_index, neg_edge_index = self.prepare_data()
                torch.save(edge_index, neg_edge_index, self.cache_index_path)

            self.edge_index, self.neg_edge_index = torch.load(self.cache_index_path)
            self.edge_index = self.edge_index.to(self.args.device)
            self.neg_edge_index = self.neg_edge_index.to(self.args.device)

            if not cache_path.exists():
                self.train_model()
                runtime = time.time() - start
                torch.save({'runtime':runtime, 'weight': self.model.cpu().state_dict()}, cache_path)
                self.model.to(self.args.device)
                
            else:
                data = torch.load(cache_path, weights_only=True)
                self.add_time += data.get('runtime')
                self.model.load_state_dict(data.get('weight'))
                self.model.to(self.args.device)

        dst_index = torch.arange(self.market.N, self.market.N+self.market.M).to(self.args.device)
        _assignment = self.model.recommend(self.edge_index, src_index=torch.arange(self.market.N), dst_index=dst_index, k=self.k)
        _assignment = _assignment - self.market.N
    
        return _assignment

    def prepare_data(self):


        # use self.market.uij = preference dot attribute to derive edge_index
        k = self.num_negatives = 128 if self.market.M > 1000 else self.market.M//10

        __, topk_indices = torch.topk(self.market.uij, k, dim=1)

        edge_index = []
        for i in range(self.market.N):
            for j in topk_indices[i]:
                edge_index.append([i, j.item()+ self.market.N])

        edge_index = torch.tensor(edge_index).T             
        return edge_index, self.gen_neg_edge(edge_index)

    def gen_neg_edge(self, edge_index):
        neg_edge_index = []
        for user in tqdm(edge_index[0].unique(), ncols=80, desc='prepare neg edge'):
            for _ in range(self.num_negatives):
                negative_item = torch.randint(self.market.N, self.market.N+self.market.M, (1,))
                while torch.any(torch.logical_and(edge_index[0] == user, edge_index[1] == negative_item)):
                    negative_item = torch.randint(self.market.N, self.market.N+self.market.M, (1,))
                neg_edge_index.append([user.item(), negative_item.item()])
        
        neg_edge_index = torch.tensor(neg_edge_index).T
        return neg_edge_index


    def train_model(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.edge_index = self.edge_index.to(self.args.device)
        self.neg_edge_index = self.neg_edge_index.to(self.args.device)

        # Training loop
        if self.args.large or self.args.task == 'yelp':
            num_epochs = 80
        else:
            num_epochs = 1280

        pbar = tqdm(range(num_epochs), ncols=90, desc='LightGCN training')
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass to compute rankings
            nd_embeddings = self.model.get_embedding(self.edge_index)

            pos_src, pos_dst = self.edge_index
            neg_src, neg_dst = self.neg_edge_index

            # Calculate positive and negative edge rankings (dot product of embeddings)
            pos_edge_rank = (nd_embeddings[pos_src] * nd_embeddings[pos_dst]).sum(dim=-1)
            neg_edge_rank = (nd_embeddings[neg_src] * nd_embeddings[neg_dst]).sum(dim=-1)

            loss = self.model.recommendation_loss(pos_edge_rank, neg_edge_rank)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Epoch':{epoch+1}, 'Loss': f'{loss.cpu().item():.4f}'})



