import torch
from dgl.data.utils import load_graphs
import numpy as np

def graph_to_text(graph, label):
    """
    Converts a graph to a textual description.
    This function needs to be adapted based on the specific characteristics of your graphs.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edges = graph.edges()

    description = f"This graph has {num_nodes} nodes and {num_edges} edges. "
    description += "The edges are as follows: "
    for i in range(num_edges):
        description += f"({edges[0][i].item()}, {edges[1][i].item()}) "
    
    # Adding adjacency list information
    description += "\nAdjacency list for each node: "
    for node in range(num_nodes):
        neighbors = graph.successors(node).numpy()
        description += f"\n1-hop neighbor set of Node {node} is {{" + ", ".join(map(str, neighbors)) + "}"
        
        # Calculate 2-hop neighbors
        two_hop_neighbors = set()
        for neighbor in neighbors:
            two_hop_neighbors.update(graph.successors(neighbor).numpy())
        
        # Remove the current node 
        two_hop_neighbors.discard(node)
        # Remove direct neighbors from 2-hop neighbors
        # two_hop_neighbors.difference_update(neighbors)

        description += f"\n2-hop neighbor set of Node {node} is {{" + ", ".join(map(str, two_hop_neighbors)) + "}"

    # description += f"The label or count for the graph is {label}."
    return description

def get_labels_and_variance(dataset_num=1, task='triangle'):
    dataset_path = './count_experiments/data/'
    dataset_prefix = 'dataset' + str(dataset_num)
    _, all_labels = load_graphs(dataset_path + dataset_prefix + '.bin')
    labels = all_labels[task].numpy()
    variance = np.var(labels)
    return labels, variance

def create_prompts(dataset_num=1, task='triangle'):
    dataset_path = './count_experiments/data/'
    dataset_prefix = 'dataset' + str(dataset_num)
    glist, all_labels = load_graphs(dataset_path + dataset_prefix + '.bin')
    labels = all_labels[task]

    prompts = []
    for i, graph in enumerate(glist):
        label = labels[i].item()
        prompt = graph_to_text(graph, label)
        prompts.append(prompt)
    
    return prompts

# Example usage
prompts = create_prompts()
true_counts, variance = get_labels_and_variance(dataset_num=1, task='triangle')
for prompt in prompts[:4]:  # Displaying first 3 prompts for demonstration
    print(prompt)
    print("True Counts: ", true_counts[:4])
    
