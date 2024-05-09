import openai
import sys
sys.path.append('/scratch/lynguyen/count-substructure-llm')
from zero_shot_prompt import create_prompts, get_labels_and_variance, graph_to_text
import numpy as np
import re, os
from util import save_metrics_to_file, calculate_metrics, save_response_to_file, extract_number, send_prompt_to_openai

dataset_num = 2
task = 'triangle'
prompt_type = 'cot_algo'

detailed_instruction = (
    """
    Count the number of triangles in the given graph below. 
    Let's construct a graph with the nodes and edges first.   
    To count triangles using the Degree-Based Ordering method, first sort vertices by their number of connections from least to most. 
    Relabel the vertices accordingly. 
    For each edge, proceed only if the vertex with the lower label has fewer connections. This ensures that each triangle is only counted once.
    Check for shared connections between the two vertices of each edge; these shared connections form triangles.         
    Output only the total number of triangles inside brackets [] once.
    List the unique triangles found inside <> each only once in the answer such as <0, 2, 4>.
    """
)

def main():
    openai.api_key = "sk-proj-2SDLXt8arB8zke9ybA9tT3BlbkFJ1l8P8ejYj2iIrSrtVH5r"

    prompts = create_prompts(dataset_num, task)
    true_counts, variance = get_labels_and_variance(dataset_num, task)
    correct_predictions = 0

    predicted_counts = []
    # Load previously saved counts if available
    saved_counts_file = f"./results/dataset_{dataset_num}/{prompt_type}/{task}/saved_counts.npy"
    if os.path.exists(saved_counts_file):
        saved_counts = np.load(saved_counts_file, allow_pickle=True)
        predicted_counts = saved_counts.tolist()
        start_index = len(predicted_counts)
        print(f"Loaded {start_index} saved counts.")
    else:
        start_index = 0

    for i, prompt in enumerate(prompts[start_index:]):
        try:
            response_text = send_prompt_to_openai(prompt, detailed_instruction)
            if response_text is not None:
                save_response_to_file(response_text, dataset_num, task, prompt_type, start_index + i, prompt)
                print(f"Processed {start_index + i + 1}/{len(prompts)} prompts")
                predicted_count = extract_number(response_text)
                print("predicted: ", predicted_count)
                print("true count: ", true_counts[i])
                predicted_counts.append(predicted_count)
                if predicted_count == true_counts[start_index + i]:
                    correct_predictions += 1
            else:
                print(f"Received None response at prompt {start_index + i + 1}")

        except Exception as e:
            print(f"Error occurred at prompt {start_index + i + 1}: {str(e)}")
            break  # Exit the loop on error

    # Save predicted counts to file
    np.save(saved_counts_file, np.array(predicted_counts))

    # Filtering valid counts for MAE and MSE calculations
    valid_indices = [i for i, count in enumerate(predicted_counts) if count is not None]
    valid_predicted_counts = [predicted_counts[i] for i in valid_indices]
    valid_true_counts = [true_counts[i] for i in valid_indices]

    mae, mse_divided_by_variance, accuracy = calculate_metrics(valid_predicted_counts, valid_true_counts, variance)
    print(f"Mean Absolute Error: {mae}")
    print(f"MSE divided by Variance: {mse_divided_by_variance}")
    print(f"Accuracy: {accuracy:.2f}")

    save_metrics_to_file(mae, mse_divided_by_variance, accuracy, dataset_num, task, prompt_type)

if __name__ == "__main__":
    main()
