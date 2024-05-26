import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, JumpingKnowledge

class GIN(nn.Module):
    def __init__(self, in_channels, dim, out_channels, use_bn=True, use_dropout=True, use_jk=True):
        super(GIN, self).__init__()

        self.conv1 = GINConv(Sequential(Linear(in_channels, dim), BatchNorm1d(dim) if use_bn else nn.Identity(), ReLU(), Linear(dim, dim), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(dim, dim), BatchNorm1d(dim) if use_bn else nn.Identity(), ReLU(), Linear(dim, dim), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(dim, dim), BatchNorm1d(dim) if use_bn else nn.Identity(), ReLU(), Linear(dim, dim), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(dim, dim), BatchNorm1d(dim) if use_bn else nn.Identity(), ReLU(), Linear(dim, dim), ReLU()))
        self.convs = nn.ModuleList((self.conv1, self.conv2, self.conv3, self.conv4))
        
        self.use_dropout=use_dropout
        self.use_jk = use_jk
        if self.use_jk:
            self.jump = JumpingKnowledge('cat')
            self.lin1 = Linear(4*dim, dim)
        else:
            self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.use_jk:
            x = self.jump(xs)
        else:
            x = xs[-1]
        x = global_add_pool(x, batch)
        if self.use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x).relu()
        if self.use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class GCN(nn.Module):
    # TODO: implement batch norm
    def __init__(self, in_channels, dim, out_channels, use_dropout=True, use_jk=True, use_bn=True):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, dim)
        self.conv2 = GCNConv(dim, dim)
        self.conv3 = GCNConv(dim, dim)
        self.conv4 = GCNConv(dim, dim)
        self.convs = nn.ModuleList((self.conv1, self.conv2, self.conv3, self.conv4))
        
        self.use_dropout=use_dropout
        self.use_jk = use_jk
        self.use_bn = use_bn

        if self.use_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(4)])
        else:
            self.bns = [None] * 4

        if self.use_jk:
            self.jump = JumpingKnowledge('cat')
            self.lin1 = Linear(4*dim, dim)
        else:
            self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            xs += [x]
        if self.use_jk:
            x = self.jump(xs)
        else:
            x = xs[-1]
        x = global_add_pool(x, batch)
        if self.use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x).relu()
        if self.use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x