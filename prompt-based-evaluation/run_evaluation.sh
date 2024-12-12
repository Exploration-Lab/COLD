activity_names=(
    "baking a cake"
    "going grocery shopping"
    "going on a train"
    "planting a tree"
    "riding on a bus"
    )

model_names=(
    "EleutherAI/gpt-neo-125M"
    "EleutherAI/gpt-neo-1.3B"
    "google/gemma-2b"
    "EleutherAI/gpt-neo-2.7B"
    "microsoft/phi-2"
    "EleutherAI/gpt-j-6B"
    "NousResearch/Llama-2-7b-chat-hf"
    "mistralai/Mistral-7B-v0.1"
    "google/gemma-7b"
    "meta-llama/Meta-Llama-3-8B"
    )

seed=42

for model_name in "${model_names[@]}"; do
    for activity_name in "${activity_names[@]}"; do
        echo "Model: $model_name, Activity: $activity_name"
        CUDA_VISIBLE_DEVICES=0 python evaluate_using_mcqa_prompts.py --model_name "$model_name" --activity_name "$activity_name" --data_directory_path "../data/causal_query_triplets_sampled_instances/" --shots 0 --prompt_template "random" --seed $seed
    done
done

# CUDA_VISIBLE_DEVICES=0 python evaluate_using_mcqa_prompts.py --model_name "microsoft/phi-2" --activity_name "riding on a bus" --data_directory_path "../data/causal_query_triplets_sampled_instances/" --shots 0 --prompt_template "template1" --seed 42