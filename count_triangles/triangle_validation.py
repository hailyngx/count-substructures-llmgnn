import re
import os
from dgl.data.utils import load_graphs
import torch
import numpy as np
import sys
sys.path.append('/scratch/lynguyen/count-substructure-llm')
from util import load_graph_list, get_edge_list, get_labels_and_variance, save_metrics_to_file, calculate_metrics

dataset_num = 1
task = 'triangle'
prompt_type = 'cot_algo'

def extract_data_from_file(file_path, edge_set):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extracting the predicted count
    predicted_count = re.search(r'\[\s*(\d+)\s*\]', content)
    predicted_count = int(predicted_count.group(1)) if predicted_count else None

    # Extracting the number of triangles
    triangle_count = re.search(r'\[\s*(\d+)\s*\]', content)
    triangle_count = int(triangle_count.group(1)) if triangle_count else 0

    # Extracting unique triangles
    triangles = re.findall(r'<([^>]+)>', content)
    valid_triangles = []
    triangles_evaluated = 0

    for tri in triangles:
        try:
            if is_valid_triangle(tri, edge_set):
                valid_triangles.append(tri)
            triangles_evaluated += 1
        except ValueError:
            # Skip triangles that cause a format error
            continue

    return predicted_count, triangle_count, valid_triangles, triangles_evaluated

def is_valid_triangle(triangle, edge_set):
    # try:
        nodes = [int(node.strip()) for node in triangle.split(',')]
        # print(nodes)
        if len(nodes) != 3 or len(set(nodes)) != 3:
            return False

        # Check if all edges of the triangle exist in the edge set
        return all((nodes[i], nodes[j]) in edge_set for i in range(3) for j in range(i+1, 3))
    # except ValueError:
    #     # Handle the case where conversion to int fails
    #     print(f"Invalid triangle format: {triangle}")
    #     return False

def main():
    base_path = f'/scratch/lynguyen/count-substructure-llm/count_triangles/results/dataset_1/{prompt_type}/triangle'  
    results = []
    glist = load_graph_list(1)
    total_triangles_evaluated = 0
    valid_triangle_count = 0

    true_counts, variance = get_labels_and_variance(dataset_num, task)
    predicted_counts = []

    for i, graph in enumerate(glist):
        edge_list = get_edge_list(graph)
        # print(edge_list)
        file_path = os.path.join(base_path, f'response_{i}.txt')
        if os.path.exists(file_path):
            _, _, valid_triangles, triangles_evaluated = extract_data_from_file(file_path, edge_list)
            valid_triangle_count += len(valid_triangles)
            total_triangles_evaluated += triangles_evaluated
            results.append((valid_triangle_count, total_triangles_evaluated))
            predicted_count, _, _, _ = extract_data_from_file(file_path, [])
            predicted_counts.append(predicted_count)
        else:
            print(f"File not found: {file_path}")
            predicted_counts.append(None)

    # Process the results as needed
    for i, (valid_triangle_count, total_triangles_evaluated) in enumerate(results):
        print(f"File {i}: Count = {valid_triangle_count}, Triangles = {total_triangles_evaluated}")
    
    if total_triangles_evaluated > 0:
        valid_percentage = (valid_triangle_count / total_triangles_evaluated) * 100
        result_text = f"Percentage of Valid Triangles: {valid_percentage:.2f}%"
    else:
        result_text = "No triangles evaluated or all had format errors."

    with open(os.path.join(base_path, 'triangle_validation.txt'), 'w') as file:
        file.write(result_text)

    # Calculate metrics
    mae, mse_divided_by_variance, accuracy = calculate_metrics(predicted_counts, true_counts, variance)
    print(f"Mean Absolute Error: {mae}")
    print(f"MSE divided by variance: {mse_divided_by_variance}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    save_metrics_to_file(mae, mse_divided_by_variance, accuracy, dataset_num, task, prompt_type)

    print(result_text)

if __name__ == "__main__":
    main()
