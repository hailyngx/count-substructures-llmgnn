import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import global_add_pool

def induce_augment(full_x, edge_index, v, r1):
    """ takes node and gives induced subgraph of egonet \ ego
        with augmented node feature
    """
    if isinstance(v, torch.Tensor):
        v = v.item()
        
    # if empty subgraph or if v is isolated
    if edge_index.shape[1] == 0 or v not in edge_index.flatten().unique():
        new_feature = torch.zeros(full_x.shape[0],1).to(full_x)
        full_x = torch.cat((full_x, new_feature), axis=1)
        hv = full_x[v]
        sub_ids = torch.tensor([], dtype=torch.long)
        return hv, sub_ids, full_x, edge_index
        
    sub_ids, sub_edge_index, _, _ = k_hop_subgraph(v, r1, edge_index)

    # augment neighbor features
    if sub_edge_index.shape[1] == 0: # if subgraph is empty
        new_feature = torch.zeros(full_x.shape[0],1).to(full_x)
        full_x = torch.cat((full_x, new_feature), axis=1)
        hv = full_x[v]
        sub_ids = sub_ids[sub_ids != v]
        return hv, sub_ids, full_x, sub_edge_index
        
    one_hop_nbrs, _, _, _ = k_hop_subgraph(v, 1, sub_edge_index)
    one_hop_nbrs = one_hop_nbrs[one_hop_nbrs != v]

    sub_ids = sub_ids[sub_ids != v]
    new_feature = torch.zeros(full_x.shape[0],1).to(full_x)
    for sub_node in sub_ids:
        if sub_node in one_hop_nbrs:
            new_feature[sub_node] = 1

    full_x = torch.cat((full_x, new_feature), axis=1)
    hv = full_x[v]

    # only keep edges not incident to ego
    non_ego_edges = (sub_edge_index[0] != v) * (sub_edge_index[1] != v)
    sub_edge_index = sub_edge_index[:, non_ego_edges]
    return hv, sub_ids, full_x, sub_edge_index

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
            
    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn: x = self.bns[i](x)
            if self.use_ln: x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
            
class FCGIN(nn.Module):
    '''fully connected GIN'''
    def __init__(self, mlp):
        super(FCGIN, self).__init__()
        self.mlp = mlp
        self.eps = 0
    
    def forward(self, hv, hvu):
        return self.mlp((1+self.eps)*hv + hvu.sum(axis=0))

class RNPGNNBase(nn.Module):
    def __init__(self, r, in_channels, hidden_channels, out_channels, num_mlp_layers=2, use_bn=False, verbose=False, use_ln=False, dropout=0.5, activation='relu'):
        super(RNPGNNBase, self).__init__()
        self.r = r
        self.t = len(r) # number of recursion updates
        self.convs = nn.ModuleList()
        
        for l in range(self.t-1):
            self.convs.append(FCGIN(
                MLP(hidden_channels+l+1, hidden_channels, hidden_channels+l, num_mlp_layers, use_bn=False, use_ln=use_ln, dropout=dropout, activation=activation)))
            
        self.convs.append(FCGIN(MLP(hidden_channels+self.t, hidden_channels, hidden_channels+self.t-1, num_mlp_layers, use_bn=False, use_ln=use_ln, dropout=dropout, activation=activation)))
            
        
        self.induce_time = 0
        self.verbose = verbose
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_channels)
    
    def forward(self, x, edge_index, batch):
        start_forward_time = time.time()
        nodes = list(range(x.shape[0])) 
        self.induce_time = 0
        x = self.rnp(nodes, x, edge_index, 0)
        if self.use_bn: x = self.bn(x)
        
        return x
    
    def rnp(self, nodes, full_x, edge_index, i):
        # full_x is node features of full graph
        # edge_index is of sub graph
        H = []
        for v in nodes:
            hv, sub_ids, new_x, sub_edge_index = induce_augment(full_x, edge_index, v, self.r[i])
            if i < self.t-1:
                hvu = self.rnp(sub_ids, new_x, sub_edge_index, i+1)
            else: # last layer (recursion base case)
                hvu = new_x[sub_ids]
                
            hv = self.convs[i](hv, hvu.to(hv))
            H.append(hv.unsqueeze(0))
        if len(H) == 0:
            # no nodes case: nbr message is zero
            return torch.zeros(0,1).to(full_x)
        embedding = torch.cat(H, axis=0)
        return embedding

class RNPGNN(nn.Module):
    def __init__(self, r, in_channels, hidden_channels, out_channels, num_layers=1, num_mlp_layers=2, use_bn=False, verbose=False, use_ln=False, dropout=0.5, activation='relu'):
        super(RNPGNN, self).__init__()
        
        self.rnp_layers = nn.ModuleList()
        
        for l in range(num_layers-1):
            self.rnp_layers.append(RNPGNNBase(r, hidden_channels, hidden_channels, hidden_channels, num_mlp_layers=num_mlp_layers, use_bn=use_bn, verbose=verbose, use_ln=use_ln, dropout=dropout, activation=activation))
            
        self.rnp_layers.append(RNPGNNBase(r, hidden_channels, hidden_channels, out_channels, num_mlp_layers=num_mlp_layers, use_bn=use_bn, verbose=verbose, use_ln=use_ln, dropout=dropout, activation=activation))
            
        self.init_project = nn.Linear(in_channels, hidden_channels)
        self.final_mlp = MLP(hidden_channels, hidden_channels, out_channels, num_mlp_layers, use_bn=False, use_ln=use_ln, dropout=dropout, activation=activation)
        
        self.induce_time = 0
        self.verbose = verbose
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_channels)
    
    def forward(self, data):
        start_forward_time = time.time()
        nodes = list(range(data.x.shape[0])) 
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.init_project(x)
        self.induce_time = 0
        for layer in self.rnp_layers:
            x = layer(x, edge_index, batch)
        if self.use_bn: x = self.bn(x)
        
        x = global_add_pool(x, batch)
        x = self.final_mlp(x)
        return x
    
