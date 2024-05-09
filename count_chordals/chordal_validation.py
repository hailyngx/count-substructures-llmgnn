import re
import os
from dgl.data.utils import load_graphs
import torch
import numpy as np
import sys
sys.path.append('/scratch/lynguyen/count-substructure-llm')
from util import load_graph_list, get_edge_list, get_labels_and_variance, save_metrics_to_file, calculate_metrics

dataset_num = 2
task = 'chordal_cycle'
prompt_type = 'few_shot_cot'

def extract_data_from_file(file_path, edge_set):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extracting the predicted count
    predicted_count = re.search(r'\[\s*(\d+)\s*\]', content)
    predicted_count = int(predicted_count.group(1)) if predicted_count else None

    # Extracting unique chordal 4-cycles
    cycles = re.findall(r'<(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*\1>', content)
    valid_cycles = []
    cycles_evaluated = 0
    for cycle in cycles:
        try:
            if is_valid_chordal_4_cycle(cycle, edge_set):
                valid_cycles.append(cycle)
            cycles_evaluated += 1
        except ValueError:
            continue

    return predicted_count, len(valid_cycles), valid_cycles, cycles_evaluated

def is_valid_chordal_4_cycle(cycle, edge_set):
    nodes = tuple(map(int, cycle))
    edges = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[3]),
        (nodes[3], nodes[0])
    ]

    # Define potential chords
    chords = [(nodes[0], nodes[2]), (nodes[1], nodes[3])]

    # Check if all cycle edges are valid
    cycle_valid = all((edge in edge_set or tuple(reversed(edge)) in edge_set) for edge in edges)

    # Count valid chords
    valid_chords = [chord for chord in chords if (chord in edge_set or tuple(reversed(chord)) in edge_set)]
    chord_count = len(valid_chords)

    # Ensure exactly one valid chord is present
    return cycle_valid and chord_count == 1

def main():
    base_path = f'/scratch/lynguyen/count-substructure-llm/count_chordals/results/dataset_{dataset_num}/{prompt_type}/chordal_cycle'  
    glist = load_graph_list(2)
    true_counts, variance = get_labels_and_variance(dataset_num, task)

    print("Var: ", variance)
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
        total_chordals_evaluated = 0
        valid_chordal_count = 0
        predicted_counts = []

        for idx, graph in graphs:
            edge_list = get_edge_list(graph)
            file_path = os.path.join(base_path, f'response_{idx}.txt')
            if os.path.exists(file_path):
                _, _, valid_chordals, chordals_evaluated = extract_data_from_file(file_path, edge_list)
                valid_chordal_count += len(valid_chordals)
                total_chordals_evaluated += chordals_evaluated
                results.append((valid_chordal_count, total_chordals_evaluated))
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
        # for i, (valid_chordal_count, total_chordals_evaluated) in enumerate(results):
        #     print(f"File {i}: Count = {valid_chordal_count}, chordals Evaluated = {total_chordals_evaluated}")
        
        if total_chordals_evaluated > 0:
            valid_percentage = (valid_chordal_count / total_chordals_evaluated) * 100
            result_text = f"Percentage of Valid chordals: {valid_percentage:.2f}%"
        else:
            result_text = "No chordals evaluated or all had format errors."

        print(f"Metrics for {difficulty} graphs:")
        print(f"  Mean Absolute Error: {mae}")
        print(f"  MSE divided by variance: {mse_divided_by_variance}")
        print(f"  Accuracy: {accuracy:.2f}")
        print(result_text)
        
        save_metrics_to_file(mae, mse_divided_by_variance, accuracy, dataset_num, task, f"{prompt_type}_{difficulty}")

if __name__ == "__main__":
    main()
