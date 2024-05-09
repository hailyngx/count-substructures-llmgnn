import re
import os
from dgl.data.utils import load_graphs
import torch
import numpy as np
import openai

def load_graph_list(dataset_num):
    dataset_path = '../count_experiments/data/'
    dataset_prefix = 'dataset' + str(dataset_num)
    glist, all_labels = load_graphs(dataset_path + dataset_prefix + '.bin')
    return glist

def get_edge_list(graph):
    num_edges = graph.number_of_edges()
    edges = graph.edges()
    edge_list = []

    for i in range(num_edges):
        edge_list.append((edges[0][i].item(), edges[1][i].item()))

    return edge_list

def get_labels_and_variance(dataset_num, task):
    dataset_path = '../count_experiments/data/'
    dataset_prefix = 'dataset' + str(dataset_num)
    _, all_labels = load_graphs(dataset_path + dataset_prefix + '.bin')
    labels = all_labels[task].numpy()
    variance = np.var(labels)
    return labels, variance

def send_prompt_to_openai(prompt, detailed_instruction):
    max_retries = 3  # Maximum number of retries
    wait_time = 1  # Initial wait time in seconds

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  
                messages=[
                    {"role": "system", "content": detailed_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.ServiceUnavailableError:
            print(f"Service unavailable. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            break  # Exit on other errors

    return None  # Return None if all retries fail


def save_metrics_to_file(mae, mse_div_variance, accuracy, dataset_num, task, prompt_type):
    # Local directory in scratch
    scratch_directory = f"./results/dataset_{dataset_num}/{prompt_type}/{task}"
    scratch_file_path = f"{scratch_directory}/metrics.txt"

    # Directory in jumbo
    jumbo_base_directory = "/jumbo/graphmind/lynguyen"
    jumbo_directory = f"{jumbo_base_directory}/dataset_{dataset_num}/{prompt_type}/{task}"
    jumbo_file_path = f"{jumbo_directory}/metrics.txt"

    # Ensure base directory in jumbo exists
    if not os.path.exists(jumbo_base_directory):
        os.makedirs(jumbo_base_directory)

    # Function to save metrics to a given file path
    def save_metrics(file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w') as file:
            file.write(f"Mean Absolute Error: {mae}\n")
            file.write(f"MSE divided by Variance: {mse_div_variance}\n")
            file.write(f"Accuracy: {accuracy*100:.2f}%\n")

    # Save to both scratch and jumbo locations
    save_metrics(scratch_file_path)
    save_metrics(jumbo_file_path)

def calculate_metrics(predicted_counts, true_counts, variance):
    # Filtering valid counts for MAE and MSE calculations
    valid_indices = [i for i, count in enumerate(predicted_counts) if count is not None]
    valid_predicted_counts = [predicted_counts[i] for i in valid_indices]
    valid_true_counts = [true_counts[i] for i in valid_indices]

    # Calculate MAE
    mae = np.mean(np.abs(np.array(valid_predicted_counts) - np.array(valid_true_counts)))

    # Calculate MSE
    mse = np.mean((np.array(valid_predicted_counts) - np.array(valid_true_counts)) ** 2)
    mse_divided_by_variance = mse / variance

    # Calculate Accuracy
    correct_predictions = sum(1 for pred, true in zip(valid_predicted_counts, valid_true_counts) if pred == true)
    accuracy = correct_predictions / len(valid_predicted_counts)

    return mae, mse_divided_by_variance, accuracy

def save_response_to_file(response, dataset_num, task, prompt_type, file_index, prompt):
    # Create a directory if it doesn't exist
    directory = f"./results/dataset_{dataset_num}/{prompt_type}/{task}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path
    file_path = f"{directory}/response_{file_index}.txt"

    # Write response to the file
    with open(file_path, 'w') as file:
        file.write("Prompt:\n" + prompt + "\n\nResponse:\n" + response)

def extract_number(text):
    """Extracts the first number found within brackets in the provided text."""
    match = re.search(r"\[\s*(\d+\.?\d*)\s*\]", text)
    return float(match.group(1)) if match else None
