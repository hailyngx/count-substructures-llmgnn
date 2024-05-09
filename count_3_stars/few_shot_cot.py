import openai
import sys
sys.path.append('/scratch/lynguyen/count-substructure-llm')
from one_hop_prompt import create_prompts, get_labels_and_variance, graph_to_text
import numpy as np
import re, os
from util import save_metrics_to_file, calculate_metrics, save_response_to_file, extract_number, send_prompt_to_openai

dataset_num = 2
task = 'star'
prompt_type = 'few_shot_cot'

detailed_instruction = (
    """
    A 3-star graph consists of a central node, called the center, which is connected to exactly three other nodes by edges. 
    This is an example of how you can count the 3-stars in a graph step-by-step applied to each node in the graph: 
    This graph has 10 nodes and 28 edges. The edges are as follows: (0, 2) (0, 4) (0, 5) (0, 7) (1, 4) (1, 7) (2, 0) (2, 4) (2, 7) (3, 9) (4, 0) (4, 1) (4, 2) (4, 6) (4, 9) (5, 0) (5, 6) (5, 7) (6, 4) (6, 5) (6, 7) (7, 0) (7, 1) (7, 2) (7, 5) (7, 6) (9, 3) (9, 4) 
    Adjacency list for each node: 
    Node 0 is connected to node 2, 4, 5, and node 7
    Node 1 is connected to node 4, and node 7
    Node 2 is connected to node 0, 4, and node 7
    Node 3 is connected to node 9
    Node 4 is connected to node 0, 1, 2, 6, and node 9
    Node 5 is connected to node 0, 6, and node 7
    Node 6 is connected to node 4, 5, and node 7
    Node 7 is connected to node 0, 1, 2, 5, and node 6
    Node 8 is not connected to any other nodes.
    Node 9 is connected to node 3, and node 4

    To find the total number of 3-stars, we'll apply the combinatorial calculation \( \binom{n}{3} \) to each node that has three or more neighbors. If it has exactly 3 neighbors, there’s only 1 3-star formed with that node. No 3-star is formed if less than 3 neighbors.  
    Calculation of 3-Stars for Each Node
    Given the adjacency list you provided, we'll examine each node:
    - **Node 0**: Neighbors = {2, 4, 5, 7} (4 neighbors)
    - Number of 3-star configurations = \(\binom{4}{3} = 4\)
    - **Node 1**: Neighbors = {4, 7} (2 neighbors)
    - Cannot form a 3-star as it has less than 3 neighbors.
    - **Node 2**: Neighbors = {0, 4, 7} (3 neighbors)
    - Number of 3-star configurations = \(\binom{3}{3} = 1\)
    - **Node 3**: Neighbors = {9} (1 neighbor)
    - Cannot form a 3-star as it has less than 3 neighbors.
    - **Node 4**: Neighbors = {0, 1, 2, 6, 9} (5 neighbors)
    - Number of 3-star configurations = \(\binom{5}{3} = 10\)
    - **Node 5**: Neighbors = {0, 6, 7} (3 neighbors)
    - Number of 3-star configurations = \(\binom{3}{3} = 1\)
    - **Node 6**: Neighbors = {4, 5, 7} (3 neighbors)
    - Number of 3-star configurations = \(\binom{3}{3} = 1\)
    - **Node 7**: Neighbors = {0, 1, 2, 5, 6} (5 neighbors)
    - Number of 3-star configurations = \(\binom{5}{3} = 10\)
    - **Node 8**: No neighbors
    - Cannot form a 3-star as it has no neighbors.
    - **Node 9**: Neighbors = {3, 4} (2 neighbors)
    - Cannot form a 3-star as it has less than 3 neighbors.
    Summing Up the Total Number of 3-Stars
    Add up all the valid configurations:
    - From Node 0: 4
    - From Node 2: 1
    - From Node 4: 10
    - From Node 5: 1
    - From Node 6: 1
    - From Node 7: 10
    Total number of 3-stars = 4 + 1 + 10 + 1 + 1 + 10 = 27
    Therefore, there are a total of [27] 3-stars in your graph. This count represents how many unique 3-star subgraphs can be formed based on the adjacency list provided.
    
    Do exactly as instructed in the example above to find the number of 3-stars in the graph below. Output the final answer inside brackets [].
    List the unique 3-stars found inside <> each only once. For example, <1, (2, 4, 5)> represents a 3-star with node 1 as the central node and nodes 2, 4, and 5 as the leaves.
    """
)

def main():
    openai.api_key = "sk-proj-2SDLXt8arB8zke9ybA9tT3BlbkFJ1l8P8ejYj2iIrSrtVH5r"

    prompts = create_prompts(dataset_num, task)
    true_counts, variance = get_labels_and_variance(dataset_num, task)
    correct_predictions = 0

    predicted_counts = []
    # Load previously saved counts if available
    saved_counts_file = f"./results/dataset_{dataset_num}/few_shot_cot/{task}/saved_counts.npy"
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
