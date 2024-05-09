import re
import os
from dgl.data.utils import load_graphs
import torch
import numpy as np
import sys
sys.path.append('/scratch/lynguyen/count-substructure-llm')
from util import load_graph_list, get_edge_list, get_labels_and_variance, save_metrics_to_file, calculate_metrics
from triangle_validation import extract_data_from_file, is_valid_triangle

dataset_num = 1
task = 'triangle'
prompt_type = 'cot_algo'

def main():
    base_path = f'/scratch/lynguyen/count-substructure-llm/count_triangles/results/dataset_{dataset_num}/{prompt_type}/triangle'  
    glist = load_graph_list(2)
    true_counts, variance = get_labels_and_variance(dataset_num, task)

    dataset_path = '../count_experiments/data/'
    dataset_prefix = 'dataset' + str(dataset_num)
    _, all_labels = load_graphs(dataset_path + dataset_prefix + '.bin')

    # Group graphs by difficulty level
    easy_graphs = []
    medium_graphs = []
    hard_graphs = []

    for i, graph in enumerate(glist):
        num_nodes = graph.number_of_nodes()
        if num_nodes == 10:
            easy_graphs.append((i, graph))
        elif num_nodes == 15 or num_nodes == 20:
            medium_graphs.append((i, graph))
        elif num_nodes == 30:
            hard_graphs.append((i, graph))

    # Calculate metrics for each difficulty level
    for difficulty, graphs in [('easy', easy_graphs), ('medium', medium_graphs), ('hard', hard_graphs)]:
        results = []
        total_triangles_evaluated = 0
        valid_triangle_count = 0
        predicted_counts = []

        for idx, graph in graphs:
            edge_list = get_edge_list(graph)
            file_path = os.path.join(base_path, f'response_{idx}.txt')
            if os.path.exists(file_path):
                _, _, valid_triangles, triangles_evaluated = extract_data_from_file(file_path, edge_list)
                valid_triangle_count += len(valid_triangles)
                total_triangles_evaluated += triangles_evaluated
                results.append((valid_triangle_count, total_triangles_evaluated))
                predicted_count, _, _, _ = extract_data_from_file(file_path, edge_list)
                predicted_counts.append(predicted_count)
            else:
                print(f"File not found: {file_path}")
                predicted_counts.append(None)

        # Calculate metrics for the current difficulty level
        true_counts_difficulty = [true_counts[idx] for idx, _ in graphs]
        variance_difficulty = np.var(true_counts_difficulty)
        mae, mse_divided_by_variance, accuracy = calculate_metrics(predicted_counts, true_counts_difficulty, variance_difficulty)

        # Process the results as needed
        # for i, (valid_triangle_count, total_triangles_evaluated) in enumerate(results):
        #     print(f"File {i}: Count = {valid_triangle_count}, triangles Evaluated = {total_triangles_evaluated}")
        
        if total_triangles_evaluated > 0:
            valid_percentage = (valid_triangle_count / total_triangles_evaluated) * 100
            result_text = f"Percentage of Valid Triangles: {valid_percentage:.2f}%"
        else:
            result_text = "No triangles evaluated or all had format errors."

        print(f"Metrics for {difficulty} graphs:")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  MSE divided by variance: {mse_divided_by_variance:.2f}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(result_text)
        
        save_metrics_to_file(mae, mse_divided_by_variance, accuracy, dataset_num, task, f"{prompt_type}_{difficulty}")

if __name__ == "__main__":
    main()
