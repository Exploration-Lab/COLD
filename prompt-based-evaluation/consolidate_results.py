import os
import json

"""
this script is used to consolidate the results of the prompt-based evaluation
the results are saved in the results directory
"""

results_logs_dir = "./results"

results = {}

for file_name in os.listdir(results_logs_dir):
    if file_name.endswith(".json") and not file_name.endswith("_evaluatedSamples.json"):
        with open(os.path.join(results_logs_dir, file_name), "r") as f:
            results[file_name] = json.load(f)


consolidated_df = []

for key, value in results.items():
    # print(key, value)

    model_name = key.split("_")[2]
    activity_name = key.split("_")[1]
    shots = key.split("_")[3]
    prompt_template = key.split("_")[4]

    accuracy = value["accuracy"]
    success_rate = value["success_rate"]

    consolidated_df.append(
        {
            "model_name": model_name,
            "activity_name": activity_name,
            "shots": shots,
            "prompt_template": prompt_template,
            "accuracy": accuracy,
            "success_rate": success_rate,
        }
    )


print("Model Name | ", end="")
for activity in [
    "baking a cake",
    "going grocery shopping",
    "going on a train",
    "planting a tree",
    "riding on a bus",
]:
    print(activity, end=" | ")
print()
for model in [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "google/gemma-2b",
    "EleutherAI/gpt-neo-2.7B",
    "microsoft/phi-2",
    "EleutherAI/gpt-j-6B",
    "NousResearch/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b",
    "meta-llama/Meta-Llama-3-8B",
]:
    print(model, end=" | ")
    model = model.replace("/", "--")
    for activity in [
        "baking a cake",
        "going grocery shopping",
        "going on a train",
        "planting a tree",
        "riding on a bus",
    ]:

        for item in consolidated_df:
            if item["activity_name"] == activity and item["model_name"] == model:
                # print(item["accuracy"]*100, end=" | ")
                print(f"{item['accuracy']*100:.2f}", end=" & ")
    print()
