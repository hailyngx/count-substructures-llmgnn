import anthropic
from cot_sc_prompt import create_prompts, get_labels_and_variance, graph_to_text
import numpy as np
import time
import re, os

def send_prompt_to_claude(prompt, previous_response=None):
    max_retries = 3  # Maximum number of retries
    wait_time = 1  # Initial wait time in seconds
    delay_between_requests = 30  # Delay between requests in seconds

    default_message = (
        "To count and list the triangles formed with Node 0 in the given graph, we'll follow a systematic approach:\n"
        "Identify Neighbors of Node 0:\n"
        "Node 0 is connected to Nodes 2, 4, and 6.\n"
        "Consider Pairs of Neighbors: The neighbor pairs for Node 0 are (2, 4), (4, 6), and (2, 6).\n"
        "Verify Triangles for Each Pair:\n"
        "Pair (2, 4):\n"
        "Check Connection with Node 0: Both Node 2 and Node 4 are connected with Node 0.\n"
        "Check Direct Connection: Node 2 and Node 4 are connected with each other.\n"
        "Since both conditions are met, Nodes 0, 2, and 4 form a triangle.\n"
        "Pair (4, 6):\n"
        "Check Connection with Node 0: Both Node 4 and Node 6 are connected with Node 0.\n"
        "Check Direct Connection: Node 4 and Node 6 are connected with each other.\n"
        "Since both conditions are met, Nodes 0, 4, and 6 form a triangle.\n"
        "Pair (2, 6):\n"
        "Check Connection with Node 0: Both Node 2 and Node 6 are connected with Node 0.\n"
        "Check Direct Connection: Node 2 and Node 6 are not connected with each other.\n"
        "Since Node 2 and Node 6 are not directly connected, Nodes 0, 2, and 6 do not form a triangle.\n"
        "Count Unique Triangles Involving Node 0: There are two unique triangles involving Node 0:\n"
        "Triangle formed by Nodes 0, 2, and 4.\n"
        "Triangle formed by Nodes 0, 4, and 6.\n"
        "Thus, following this method, Node 0 is part of 2 distinct triangles 0, 2, 4 and 0, 4, 6 in the graph.\n\n"
        "Now, our goal is to count the total and list the unique triangles for this graph."
    )

    for attempt in range(max_retries):
        try:
            client = anthropic.Client(api_key=anthropic.api_key)
            response = client.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {default_message}\n{prompt}\n\n{anthropic.AI_PROMPT}",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-v1",
                max_tokens_to_sample=1000,
            )
            time.sleep(delay_between_requests)  # Add delay between requests
            return response.completion.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
            else:
                raise

def save_response_to_file(response, dataset_num, task, file_index, prompt):
    # Create a directory if it doesn't exist
    directory = f"./results/dataset_{dataset_num}/claude_cot_sc/{task}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path
    file_path = f"{directory}/response_{file_index}.txt"

    # Write response to the file
    with open(file_path, 'a') as file:
        file.write("\nPrompt:\n" + prompt + "\n\nResponse:\n" + response)

def extract_number(text):
    """Extracts the first number found within brackets in the provided text."""
    match = re.search(r"\[\s*(\d+\.?\d*)\s*\]", text)
    return float(match.group(1)) if match else None

def save_metrics_to_file(mae, mse_div_variance, accuracy, dataset_num, task):
    # Local directory in scratch
    scratch_directory = f"./results/dataset_{dataset_num}/claude_cot_sc/{task}"
    scratch_file_path = f"{scratch_directory}/metrics.txt"

    # Directory in jumbo
    jumbo_base_directory = "/jumbo/graphmind/lynguyen"
    jumbo_directory = f"{jumbo_base_directory}/dataset_{dataset_num}/claude_cot_sc/{task}"
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
            file.write(f"Accuracy: {accuracy:.2f}\n")

    # Save to both scratch and jumbo locations
    save_metrics(scratch_file_path)
    save_metrics(jumbo_file_path)

def main():
    anthropic.api_key = "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"

    dataset_num = 1
    task = "triangle"
    prompts = create_prompts(dataset_num, task)
    true_counts, variance = get_labels_and_variance(dataset_num, task)
    correct_predictions = 0

    predicted_counts = []
    # Load previously saved counts if available
    saved_counts_file = f"./results/dataset_{dataset_num}/claude_cot_sc/{task}/saved_counts.npy"
    if os.path.exists(saved_counts_file):
        saved_counts = np.load(saved_counts_file, allow_pickle=True)
        predicted_counts = saved_counts.tolist()
        start_index = len(predicted_counts)
        print(f"Loaded {start_index} saved counts.")
    else:
        start_index = 0

    for i, prompt in enumerate(prompts[start_index:]):
        if i > 5:
            break
        try:
            sub_prompts = prompt.split("---")
            response_text = ""
            previous_response = None
            for sub_prompt in sub_prompts:
                sub_prompt = sub_prompt.strip()
                if sub_prompt:
                    print(f"Sending sub-prompt: {sub_prompt}")
                    response = send_prompt_to_claude(sub_prompt, previous_response)
                    response_text += "Prompt:\n" + sub_prompt + "\n\nResponse:\n" + response + "\n\n"
                    save_response_to_file(response, dataset_num, task, start_index + i, sub_prompt)
                    previous_response = response

            print(f"Processed {start_index + i + 1}/{len(prompts)} prompts")
            predicted_count = extract_number(response_text)
            print("Predicted count:", predicted_count)
            print("True count:", true_counts[start_index + i])
            predicted_counts.append(predicted_count)
            if predicted_count == true_counts[start_index + i]:
                correct_predictions += 1
        except Exception as e:
            print(f"Error occurred at prompt {start_index + i + 1}: {str(e)}")
            break  # Exit the loop on error

    # Save predicted counts to file
    os.makedirs(os.path.dirname(saved_counts_file), exist_ok=True)
    np.save(saved_counts_file, np.array(predicted_counts))

    # Filtering valid counts for MAE and MSE calculations
    valid_indices = [i for i, count in enumerate(predicted_counts) if count is not None]
    valid_predicted_counts = [predicted_counts[i] for i in valid_indices]
    valid_true_counts = [true_counts[i] for i in valid_indices]

    # Calculate MAE
    mae = np.mean(np.abs(np.array(valid_predicted_counts) - np.array(valid_true_counts)))

    # Calculate MSE divided by variance
    mse = np.mean((np.array(valid_predicted_counts) - np.array(valid_true_counts)) ** 2)
    mse_divided_by_variance = mse / variance
    # Calculate Accuracy
    accuracy = correct_predictions / len(prompts)

    print(f"Mean Absolute Error: {mae}")
    print(f"MSE divided by Variance: {mse_divided_by_variance}")
    print(f"Accuracy: {accuracy:.2f}")

    save_metrics_to_file(mae, mse_divided_by_variance, accuracy, dataset_num, task)

if __name__ == "__main__":
    main()