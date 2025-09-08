import time
import random
import torch
from tqdm import tqdm
from .greedy import BaselineSolver
from utils import *

class HetRecSysSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)
        args.embed_dim = 16
        args.percent = 0.6
        args.reg = 1
    
        self.cache_path = self._get_cache_path("")/f'HetRecSys.pth'
        if not args.large:
            # load afterwards modifications for args.large
            self.do_preparations()

    def do_preparations(self):
        self.prepare_Nums_Lists_Data()
        self.model = GraphConsis(self.args, self.Nums, self.Lists)
        self.model.to(self.args.device)


    def prepare_Nums_Lists_Data(self):
        '''
        Nums = user_num, item_num
        Lists = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists]

        '''
        self.Nums = [self.market.N, self.market.M]
        history_u_lists = [x.tolist() for x in self.market.uij.topk(10)[1]]
        history_ur_lists = [[5]*10]*self.market.N 
        history_v_lists = [] 
        history_vr_lists = []
        for j in tqdm(range(self.market.M), ncols=88, desc='make Lists', leave=False):
            u_list = [i for i in range(self.market.N) if j in history_u_lists[i]]
            if len(u_list) == 0:
                u = random.choice(range(self.market.N))
                ulist = [u]
                history_u_lists[u].append(j)
                history_ur_lists[u].append(5)
            history_v_lists.append(u_list)
            history_vr_lists.append([5]*len(u_list))

        social_adj_lists = self.find_connections(self.market.buyer_preferences)
        item_adj_lists = self.find_connections(self.market.nft_attributes)
        self.Lists = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists]

        self.Data = []
        for i in range(self.market.N):
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
        if self.args.large or self.args.task == 'yelp':
            num_epochs = 1
        else:
            num_epochs = 10
        for epoch in tqdm(range(num_epochs), ncols=88, leave=False):
            for data in tqdm(train_loader, ncols=88, desc=f'train epoch:{epoch}', leave=False):
                batch_u = data[0][:, 0].to(self.args.device)
                batch_v = data[0][:, 1].to(self.args.device)
                labels = data[0][:, 2].to(self.args.device)
                loss = self.model.loss(batch_u, batch_v, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def initial_assignment(self):

        if self.args.no_method_cache:
            self.train_model()
        else:
            if not self.cache_path.exists():
                start = time.time()
                self.train_model()
                runtime = time.time() - start
                torch.save({'runtime':runtime, 'weight': self.model.cpu().state_dict()}, self.cache_path)
                self.model.to(self.args.device)
                
            else:
                data = torch.load(self.cache_path, weights_only=True)
                self.add_time += data.get('runtime')
                self.model.load_state_dict(data.get('weight'))
                self.model.to(self.args.device)

        _len = self.k
        _assignment = torch.ones((self.market.N, _len), device=self.args.device).long()
        with torch.no_grad():
            for i in range(self.market.N):
                topk = (self.model.u2e.weight[i] @ self.model.v2e.weight.T).topk(_len)[1]
                _assignment[i] = topk

        return _assignment


# code from https://github.com/jimmy-academia/MSOPDS/blob/main/src/networks/consisrec.py
# original source https://github.com/YangLiangwei/ConsisRec

import torch
import torch.nn as nn

# specs: list of lists, list of lists

class GraphConsis(nn.Module):
    def __init__(self, args, Nums, Lists):
        super(GraphConsis, self).__init__()
        self.device = args.device
        self.num_users, self.num_items = Nums
        self.u2e = nn.Embedding(self.num_users, args.embed_dim).to(self.device)
        self.v2e = nn.Embedding(self.num_items, args.embed_dim).to(self.device)
        self.r2e = nn.Embedding(7, args.embed_dim).to(self.device)

        self.u2e.weight.data.uniform_(-1, 1)
        self.v2e.weight.data.uniform_(-1, 1)

        # 0~5 for ciao, 1~5 for other data, 6 for neighbor
        self.node_enc = Node_Encoder(args, [self.u2e, self.v2e, self.r2e], Lists)
        self.tolong = lambda l: torch.LongTensor(l).to(self.device)
        self.reg = args.reg
        
    def forward(self, nodes_u, nodes_v):
        embeds_u = self.node_enc(nodes_u, nodes_v, isitem = False)
        embeds_v = self.node_enc(nodes_v, nodes_u, isitem = True)
        scores = torch.mul(embeds_u, embeds_v).sum(1)
        return scores

    def loss(self, nodes_u, nodes_v, labels_):
        embeds_u = self.node_enc(nodes_u, nodes_v, isitem = False)
        embeds_v = self.node_enc(nodes_v, nodes_u, isitem = True)
        scores = torch.mul(embeds_u, embeds_v).sum(1)
        
        if self.training:
            return torch.sum((scores - labels_) ** 2) + self.reg*(embeds_u.norm(2).pow(2) + embeds_v.norm(2).pow(2))
        else:
            return torch.sum((scores - labels_) ** 2) 

    def grid_results(self, nodes_u, nodes_v):
        ulen, vlen = map(len, [nodes_u, nodes_v])
        nodes_u = nodes_u*vlen
        nodes_v = [x for x in nodes_v for __ in range(ulen)]
        ans = self.forward(nodes_u, nodes_v)
        return ans.view(ulen, vlen)


class Node_Encoder(nn.Module):

    def __init__(self, args, Embeddings, Lists):
        super(Node_Encoder, self).__init__()

        self.u2e, self.v2e, __ = Embeddings
        self.history_u_lists, self.history_ur_lists, self.history_v_lists, self.history_vr_lists, self.social_adj_lists, self.item_adj_lists = Lists

        self.aggregator = Node_Aggregator(args, Embeddings)

        self.device = args.device
        self.embed_dim = args.embed_dim
        self.percent = args.percent
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.linear2 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.bn1 = nn.BatchNorm1d(self.embed_dim)
        self.bn2 = nn.BatchNorm1d(self.embed_dim)
        self.ItemLists = [self.history_v_lists, self.history_vr_lists, self.item_adj_lists, self.v2e, self.u2e]
        self.UserLists = [self.history_u_lists, self.history_ur_lists, self.social_adj_lists, self.u2e, self.v2e]

        self.device = args.device

    def forward(self, nodes, nodes_target, isitem):        
        history_n_list, history_r_list, node_adj, node_emb, target_emb  = self.ItemLists if isitem else self.UserLists

        history_n = []
        history_r = []
        adj = []
        for nid in nodes:
            history_n.append(history_n_list[nid])
            history_r.append(history_r_list[nid])
            adj.append(list(node_adj[nid]))
        
        if type(nodes) == list:
            nodes = torch.LongTensor(nodes).to(self.device)
        if type(nodes_target) == list:
            nodes_target = torch.LongTensor(nodes_target).to(self.device)

        node_feats = node_emb(nodes)
        target_feats = target_emb(nodes_target)
        

        neigh_feats = self.aggregator.forward(nodes, node_feats, target_feats, history_n, history_r, adj, isitem, self.percent)
        combined_emb = torch.cat((node_feats, neigh_feats), dim = -1)
        combined_emb = torch.relu(self.linear1(combined_emb))

        return combined_emb


class Node_Aggregator(nn.Module):
    
    def __init__(self, args, Embeddings):
        super(Node_Aggregator, self).__init__()
        self.u2e, self.v2e, self.r2e = Embeddings
        self.device = args.device
        self.embed_dim = args.embed_dim
        self.relation_att = nn.Parameter(torch.randn(2 * self.embed_dim, requires_grad=True).to(self.device))
        self.linear = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.softmax1 = nn.Softmax(dim = 0)
        self.softmax2 = nn.Softmax(dim = 0)

    def neighbor_agg(self, q, hist_feat, rate_feat, percent):
        prob = -torch.norm(q - hist_feat, dim = 1)
        prob = self.softmax1(prob)
        select_ind = torch.multinomial(prob, max(1,int(percent * len(hist_feat))))
        relation_selected = rate_feat[select_ind]
        history_selected = hist_feat[select_ind]
        selected = torch.cat((history_selected, relation_selected), 1)
        selected = torch.mm(selected, self.relation_att.unsqueeze(0).t()).squeeze(-1)
        prob = self.softmax2(selected)

        return torch.mm(history_selected.transpose(0,1), prob.unsqueeze(-1)).squeeze(-1)

    def forward(self, nodes, node_feats, target_feats, history_n, history_r, adj, isitem, percent):

        Emb = torch.zeros(len(history_n), self.embed_dim, dtype=torch.float).to(self.device)
        query = self.linear(torch.cat((node_feats, target_feats), dim = -1))
        hist_emb, neighbor_emb = [self.u2e, self.v2e] if isitem else [self.v2e, self.u2e]

        for emb, n, q, hnodes, hrates, neibs in zip(Emb, nodes, query, history_n, history_r, adj):
            hist_feat = torch.cat((hist_emb.weight[hnodes], neighbor_emb.weight[neibs]))
            num_history, num_neighbor = len(hnodes), len(neibs)
            rate_feat = self.r2e.weight[hrates + [6]*num_neighbor]

            if num_history+num_neighbor != 0:
                emb = self.neighbor_agg(q, hist_feat, rate_feat, percent)

        return Emb

