import torch
from dgl.data.utils import load_graphs
import numpy as np

def fetch_graph(dataset_num=2, task='star'):
    dataset_path = './count_experiments/data/'
    dataset_prefix = 'dataset' + str(dataset_num)
    glist, all_labels = load_graphs(dataset_path + dataset_prefix + '.bin')
    return glist, all_labels

def count_graphs_by_difficulty_and_edges(dataset_num=2):
    glist, _ = fetch_graph(dataset_num)  # Assuming labels are not needed for this specific task
    # Initializing counters for each difficulty and edge category
    counts = {
        'Easy': {'Sparse': 0, 'Dense': 0},
        'Medium': {'Sparse': 0, 'Dense': 0},
        'Hard': {'Sparse': 0, 'Dense': 0}
    }
    max_edges = 0  # Initialize max edges tracker

    for graph in glist:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        max_possible_edges = num_nodes * (num_nodes - 1) 
        if num_edges > max_edges:
            max_edges = num_edges  # Update max edges if current graph has more edges

        edge_ratio = num_edges / max_possible_edges if max_possible_edges > 0 else 0

        # Determine edge category
        if edge_ratio <= 0.2:
            edge_category = 'Sparse'
        else:
            edge_category = 'Dense'

        # Categorize by difficulty based on node counts
        if num_nodes == 10:
            counts['Easy'][edge_category] += 1
        elif num_nodes == 15 or num_nodes == 20:
            counts['Medium'][edge_category] += 1
        elif num_nodes == 30:
            counts['Hard'][edge_category] += 1

    return counts, max_edges

difficulty_edge_counts = count_graphs_by_difficulty_and_edges(dataset_num=2)
print("Counts of graphs by difficulty and edge level:", difficulty_edge_counts)
