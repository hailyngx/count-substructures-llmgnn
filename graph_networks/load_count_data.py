import numpy as np

import torch
from torch_geometric.data import DataLoader, Data
import dgl
from dgl.data.utils import load_graphs

def count_data_to_tg(dataset_num=1, task='star'):
    """ dataset_num can be 1 or 2,
    task can be subgraph choices:
    ['star', 'triangle', 'tailed_triangle', 'chordal_cycle', 'attributed_triangle'] """
    # NOTE: only correct for unattributed right now
    
    dataset_path = '../count_experiments/data/'
    dataset_prefix = 'dataset' + str(dataset_num)
    glist, all_labels = load_graphs(dataset_path + dataset_prefix + '.bin')
    graphs = glist

    labels = all_labels[task]

    variance = np.std(labels.numpy()) ** 2
    print("Label variance: ", variance)

    train_idx = []
    with open(dataset_path + dataset_prefix + '_train.txt', "r") as f:
        for line in f:
            train_idx.append(int(line.strip()))

    val_idx = []
    with open(dataset_path + dataset_prefix + '_val.txt', "r") as f:
        for line in f:
            val_idx.append(int(line.strip()))

    test_idx = []
    with open(dataset_path + dataset_prefix + '_test.txt', "r") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    
    tg_list = []
    for i in range(len(glist)):
        tg_data = Data()
        edge_index = torch.stack(glist[i].edges(), dim=0)
        tg_data.edge_index = edge_index
        tg_data.y = labels[i]
        # NOTE: have not implemented attributes
        tg_data.x = torch.ones(glist[i].num_nodes(), 1)
        tg_list.append(tg_data)
            
    train_tg_list = []
    val_tg_list = []
    test_tg_list = []
    for i in train_idx:
        train_tg_list.append(tg_list[i])
    for i in val_idx:
        val_tg_list.append(tg_list[i])
    for i in test_idx:
        test_tg_list.append(tg_list[i])
        
    return train_tg_list, val_tg_list, test_tg_list, variance

