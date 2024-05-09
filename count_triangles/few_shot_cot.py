import openai
import sys
sys.path.append('/scratch/lynguyen/count-substructure-llm')
from one_hop_prompt import create_prompts, get_labels_and_variance, graph_to_text
import numpy as np
import re, os
from util import save_metrics_to_file, calculate_metrics, save_response_to_file, extract_number, send_prompt_to_openai

dataset_num = 2
task = 'triangle'
prompt_type = 'few_shot_cot'

detailed_instruction = (
    """
    This is an example of how you can count the triangles in a graph step-by-step applied to each node in the graph: 
    This graph has 10 nodes and 38 edges. The edge list is as follows: (0, 2) (0, 4) (0, 6) (1, 4) (1, 5) (1, 8) (2, 0) (2, 4) (2, 5) (2, 8) (2, 9) (3, 5) (3, 7) (3, 8) (4, 0) (4, 1) (4, 2) (4, 5) (4, 6) (4, 7) (4, 8) (5, 1) (5, 2) (5, 3) (5, 4) (6, 0) (6, 4) (6, 8) (6, 9) (7, 3) (7, 4) (8, 1) (8, 2) (8, 3) (8, 4) (8, 6) (9, 2) (9, 6) 
    Adjacency list for each node: 
    1-hop neighbor set of Node 0 is {2, 4, 6}
    1-hop neighbor set of Node 1 is {4, 5, 8}
    1-hop neighbor set of Node 2 is {0, 4, 5, 8, 9}
    1-hop neighbor set of Node 3 is {5, 7, 8}
    1-hop neighbor set of Node 4 is {0, 1, 2, 5, 6, 7, 8}
    1-hop neighbor set of Node 5 is {1, 2, 3, 4}
    1-hop neighbor set of Node 6 is {0, 4, 8, 9}
    1-hop neighbor set of Node 7 is {3, 4}
    1-hop neighbor set of Node 8 is {1, 2, 3, 4, 6}
    1-hop neighbor set of Node 9 is {2, 6}

    * Node 0:
        * Pair 1: (2, 4) -> 2 and 4 are both 1-hop neighbors of 0, and 2 and 4 are also connected as shown in the edge list.
        * Pair 2: (2, 6) -> 2 and 6 are not connected because (2, 6) is not in the edge list. No triangle formed.
        * Pair 3: (4, 6) -> 4 and 6 are both 1-hop neighbors of 0, and 4 and 6 are also connected as shown in the edge list.
        * Total triangles with Node 0 as the smallest numbered node: 2

    * Node 1:
        * Pair 1: (4, 5) -> 4 and 5 are both 1-hop neighbors of 1, and 4 and 5 are also connected as (4, 5) shown in the edge list.
        * Pair 2: (4, 8) -> 4 and 8 are both 1-hop neighbors of 1, and 4 and 8 are also connected as (4, 8) shown in the edge list.
        * Pair 3: (5, 8) -> 5 and 8 are not 1-hop neighbors of each other because (5, 8) is not in the edge list. No triangle formed.
        * Total triangles with Node 1 as the smallest numbered node: 2

    * Node 2:
        * Skip any pair that includes Node 0 or 1, as they are smaller than Node 2.
        * Pair 1: (4, 5) -> Both 4 and 5 are 1-hop neighbors of 2, and 4 and 5 are directly connected as (4, 5) shown in the edge list. Triangle formed.
        * Pair 2: (4, 8) -> Both 4 and 8 are 1-hop neighbors of 2, and 4 and 8 are directly connected as (4, 8) shown in the edge list. Triangle formed.
        * Pair 3: (4, 9) -> Both 4 and 9 are 1-hop neighbors of 2, but 4 and 9 are not directly connected as there is no edge (4, 9) in the list. No triangle formed.
        * Pair 4: (5, 8) -> Both 5 and 8 are 1-hop neighbors of 2, but 5 and 8 are not directly connected as there is no edge (5, 8) in the list. No triangle formed.
        * Pair 5: (5, 9) -> Both 5 and 9 are 1-hop neighbors of 2, but 5 and 9 are not directly connected as there is no edge (5, 9) in the list. No triangle formed.
        * Pair 6: (8, 9) -> Both 8 and 9 are 1-hop neighbors of 2, but 8 and 9 are not directly connected as there is no edge (8, 9) in the list. No triangle formed.
        * Total triangles with Node 2 as the smallest numbered node: 2

    * Node 3:
        * Pair 1: (5, 7) -> Both 5 and 7 are 1-hop neighbors of 3, but 5 and 7 are not directly connected as there is no edge (5, 7) in the list. No triangle formed.
        * Pair 2: (5, 8) -> Both 5 and 8 are 1-hop neighbors of 3, but 5 and 8 are not directly connected as there is no edge (5, 8) in the list. No triangle formed.
        * Pair 3: (7, 8) -> Both 7 and 8 are 1-hop neighbors of 3, but 7 and 8 are not directly connected as there is no edge (7, 8) in the list. No triangle formed.
        * Total triangles with Node 3 as the smallest numbered node: 0

    * Node 4:
        * Skip any pairs with Nodes 0, 1, 2, and 3 as they are smaller than 4.
        * Pair 1: (5, 6) -> Both 5 and 6 are 1-hop neighbors of 4, but 5 and 6 are not directly connected as there is no edge (5, 6) in the list. No triangle formed.
        * Pair 2: (5, 7) -> Both 5 and 7 are 1-hop neighbors of 4, but 5 and 7 are not directly connected as there is no edge (5, 7) in the list. No triangle formed.
        * Pair 3: (5, 8) -> Both 5 and 8 are 1-hop neighbors of 4, but 5 and 8 are not directly connected as there is no edge (5, 8) in the list. No triangle formed.
        * Pair 4: (6, 8) -> Both 6 and 8 are 1-hop neighbors of 4, and 6 and 8 are directly connected as (6, 8) shown in the edge list. Triangle formed.
        * Pair 5: (6, 9) -> Both 6 and 9 are 1-hop neighbors of 4, but 6 and 9 are not directly connected as there is no edge (6, 9) in the list. No triangle formed.
        * Total triangles with Node 4 as the smallest numbered node: 1

    * Node 5:
        * Skip any pairs with Nodes 1, 2, 3, and 4 as they are smaller than 5. No other nodes left to form triangles.
        * Total triangles with Node 5 as the smallest numbered node: 0

    Alright, let's continue this process for Nodes 6, 7, 8, and 9.

    * Node 6:
        * Skip any pairs with Nodes 0 and 4 as they are smaller than 6.
        * Pair 1: (8, 9) - Both 8 and 9 are 1-hop neighbors of 6, but 8 and 9 are not directly connected as there is no edge (8, 9) in the list. No triangle formed.
        * Total triangles with Node 6 as the smallest numbered node: 0

    * Node 7:
        * The only pair we could consider is (3, 4), but since 3 and 4 are both smaller than 7, this pair is skipped.
        * Total triangles with Node 7 as the smallest numbered node: 0

    * Node 8 and Node 9:
        * Skipped because there aren’t at least 2 nodes greater than 8 or 9.

    Summarizing the triangles:
    * Total triangles in the graph = 2 (Node 0) + 2 (Node 1) + 2 (Node 2) + 0 (Node 3) + 1 (Node 4) + 0 (Node 5) + 0 (Node 6) + 0 (Node 7) + 0 (Node 8) + 0 (Node 9) = 7.

    Therefore, there are 7 triangles in the given graph.
    
    Do exactly as instructed in the example above to find the number of triangles in the graph below. Output the final answer inside brackets [].
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
