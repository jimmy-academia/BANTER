import random
import torch
from tqdm import tqdm

from .consisrec import GraphConsis
from .heuristics import HeuristicsSolver

from utils import *

class HetRecSysSolver(HeuristicsSolver):
    def __init__(self, args):
        super().__init__(args)
        args.embed_dim = 16
        args.percent = 0.6
        args.reg = 1
        self.prepare_Nums_Lists_Data()
        self.model = GraphConsis(args, self.Nums, self.Lists)
        self.model.to(self.args.device)

    def prepare_Nums_Lists_Data(self):
        '''
        Nums = user_num, item_num
        Lists = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists]

        '''
        self.Nums = [self.nftP.N, self.nftP.M]
        history_u_lists = [x.tolist() for x in self.Uij.topk(10)[1]]
        history_ur_lists = [[5]*10]*self.nftP.N 
        history_v_lists = [] 
        history_vr_lists = []
        for j in tqdm(range(self.nftP.M), ncols=88, desc='make Lists', leave=False):
            u_list = [i for i in range(self.nftP.N) if j in history_u_lists[i]]
            if len(u_list) == 0:
                u = random.choice(range(self.nftP.N))
                ulist = [u]
                history_u_lists[u].append(j)
                history_ur_lists[u].append(5)
            history_v_lists.append(u_list)
            history_vr_lists.append([5]*len(u_list))

        social_adj_lists = self.find_connections(self.buyer_preferences)
        item_adj_lists = self.find_connections(self.nft_attributes)
        self.Lists = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists]

        self.Data = []
        for i in range(self.nftP.N):
            for j in history_u_lists[i]:
                self.Data.append([i,j,5])

    def find_connections(self, data):
        data = data.float()
        adj_lists = []
        for batch_indexes in make_batch_indexes(len(data), 128):
            distances = torch.cdist(data[batch_indexes], data)
            _, k_nearest_neighbors = distances.topk(16, largest=False, dim=1)
            adj_lists += [vec.tolist() for vec in k_nearest_neighbors]

        return adj_lists

    def train_model(self):
        train_data = torch.utils.data.TensorDataset(torch.LongTensor(self.Data))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-5)
        best_validate = 1e10
        best_model_state_dict = endure_count = va_loss = 0
        self.model.train()
        for epoch in range(10):
            for data in tqdm(train_loader, ncols=88, desc=f'train epoch:{epoch}', leave=False):
                batch_u = data[0][:, 0].to(self.args.device)
                batch_v = data[0][:, 1].to(self.args.device)
                labels = data[0][:, 2].to(self.args.device)
                loss = self.model.loss(batch_u, batch_v, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def initial_assignment(self):
        self.train_model()
        _len = 32
        _assignment = torch.ones((self.nftP.N, _len), device=self.args.device).long()
        with torch.no_grad():
            for i in range(self.nftP.N):
                topk = (self.model.u2e.weight[i] @ self.model.v2e.weight.T).topk(_len)[1]
                _assignment[i] = topk

        return _assignment