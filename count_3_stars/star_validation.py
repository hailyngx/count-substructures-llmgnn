import re
import os
from dgl.data.utils import load_graphs
import torch
import numpy as np
import sys
sys.path.append('/scratch/lynguyen/count-substructure-llm')
from util import load_graph_list, get_edge_list, get_labels_and_variance, save_metrics_to_file, calculate_metrics

dataset_num = 1
task = 'star'
prompt_type = 'cot_bag'

def extract_data_from_file(file_path, edge_set):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extracting the predicted count
    predicted_count = re.search(r'\[\s*(\d+)\s*\]', content)
    predicted_count = int(predicted_count.group(1)) if predicted_count else None

    # Extracting the number of 3-stars
    star_count = re.search(r'\[\s*(\d+)\s*\]', content)
    star_count = int(star_count.group(1)) if star_count else 0

    # Extracting unique 3-stars
    # stars = re.findall(r'<([^>]+)>', content)
    stars = re.findall(r'<(\d+,\s*\([^\)]+\))>', content)
    valid_stars = []
    stars_evaluated = 0
    for star in stars:
        try:
            if is_valid_3_star(star, edge_set):
                valid_stars.append(star)
            stars_evaluated += 1
        except ValueError:
            # Skip stars that cause a format error
            continue

    return predicted_count, star_count, valid_stars, stars_evaluated

def is_valid_3_star(star, edge_set):
    # try:
        # Remove angle brackets and split the central node from the connected nodes
        star = star.strip('<>')
        # Split the central node from the connected nodes
        central_node, connected_nodes = star.split(', ', 1)
        central_node = int(central_node.strip())  # Convert central node to int

        # Strip parentheses and convert connected nodes to a tuple of integers
        connected_nodes = connected_nodes.strip('()[]')
        connected_nodes = tuple(map(int, connected_nodes.split(', ')))  # Convert string numbers to integers

        # Check if all edges from the central node to each of the connected nodes exist
        return all((central_node, node) in edge_set or (node, central_node) in edge_set for node in connected_nodes)
        # else:
        #     return False
    # except ValueError as e:
    #     print(f"Invalid star format or conversion error in star: {star}, Error: {e}")
    #     return False
    # except SyntaxError as e:
    #     print(f"Syntax error in parsing star: {star}, Error: {e}")
    #     return False

def main():
    base_path = f'/scratch/lynguyen/count-substructure-llm/count_3_stars/results/dataset_{dataset_num}/{prompt_type}/star'  
    results = []
    glist = load_graph_list(1)
    total_stars_evaluated = 0
    valid_star_count = 0

    true_counts, variance = get_labels_and_variance(dataset_num, task)
    predicted_counts = []

    for i, graph in enumerate(glist):
        edge_list = get_edge_list(graph)
        file_path = os.path.join(base_path, f'response_{i}.txt')
        if os.path.exists(file_path):
            _, _, valid_stars, stars_evaluated = extract_data_from_file(file_path, edge_list)
            valid_star_count += len(valid_stars)
            total_stars_evaluated += stars_evaluated
            results.append((valid_star_count, total_stars_evaluated))
            predicted_count, _, _, _ = extract_data_from_file(file_path, [])
            predicted_counts.append(predicted_count)
        else:
            print(f"File not found: {file_path}")
            predicted_counts.append(None)

    # Process the results as needed
    # for i, (valid_star_count, total_stars_evaluated) in enumerate(results):
    #     print(f"File {i}: Count = {valid_star_count}, Stars Evaluated = {total_stars_evaluated}")
    
    if total_stars_evaluated > 0:
        valid_percentage = (valid_star_count / total_stars_evaluated) * 100
        result_text = f"Percentage of Valid 3-Stars: {valid_percentage:.2f}%"
    else:
        result_text = "No 3-stars evaluated or all had format errors."

    with open(os.path.join(base_path, 'star_validation.txt'), 'w') as file:
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
