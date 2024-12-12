import argparse
import sys
import os
import torch
import random
import numpy as np

from tqdm import tqdm
from typing import Tuple


from evaluation_templates.mcqa import mcqa, PerformanceTrackerMCQA

from utils import load_model_and_tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_prompt(model, tokenizer, args, device, verbose=False):
    # Load the dataset
    args.dataset_path = os.path.join(
        args.data_directory_path,
        f"{args.activity_name}.csv",
    )
    dataset = mcqa(
        args.dataset_path,
        args.prompt_template,
        args.shots,
        shuffle_options=args.shuffle_options,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    model.eval()
    model.to(device)

    performance_tracker = PerformanceTrackerMCQA()

    for idx, (prompt, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        prompt = prompt[0]
        label = label[0]

        tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True)

        input_token_ids = tokenized_prompt["input_ids"].to(device)

        with torch.no_grad():
            generated_logits = model(input_token_ids).logits
        generated_logits = generated_logits[:, -1, :]
        predicted_choice = tokenizer.decode(
            torch.argmax(generated_logits, dim=-1).item()
        )

        option_ids = [tokenizer.encode(f" {chr(65 + i)}")[-1] for i in range(2)]
        option_logits = generated_logits[:, option_ids]

        performance_tracker.update(
            prompt=prompt,
            correct_choice=chr(65 + label),
            generated_argmax_token=predicted_choice,
            choice_A_logit=option_logits[0][0].item(),
            choice_B_logit=option_logits[0][1].item(),
        )

        if verbose and idx < 10:
            print(f"Prompt: {prompt}")
            print(f"Correct choice: {chr(65 + label)}")
            print(f"Predicted choice: {predicted_choice}")
            print(f"Choice A logit: {option_logits[0][0].item()}")
            print(f"Choice B logit: {option_logits[0][1].item()}")
            print("\n")

        if idx % 1000 == 0:
            performance_tracker.print_results(verbose=True)
            if not os.path.exists(args.save_dir_path):
                os.makedirs(args.save_dir_path)
            model_tag = args.model_name.replace("/", "--")
            performance_tracker.save(
                os.path.join(
                    args.save_dir_path,
                    f"mcqa_{args.activity_name}_{model_tag}_{args.shots}_shot_{args.prompt_template}.json",
                )
            )

    performance_tracker.print_results(verbose=verbose)
    performance_tracker.save(
        os.path.join(
            args.save_dir_path,
            f"mcqa_{args.activity_name}_{model_tag}_{args.shots}_shot_{args.prompt_template}.json",
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="The model name of the model to be evaluated",
        choices=[
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
        ],
    )
    parser.add_argument(
        "--activity_name",
        type=str,
        default="going grocery shopping",
        help="The name of the activity to be evaluated",
        choices=[
            "baking a cake",
            "going grocery shopping",
            "going on a train",
            "planting a tree",
            "riding on a bus",
        ],
    )
    parser.add_argument(
        "--data_directory_path",
        type=str,
        default="../data/causal_query_triplets_sampled_instances/",
        help="The path to the directory containing the data",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="The number of shots to be used for the evaluation",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        required=True,
        choices=[
            "template1",
            "template2",
            "template3",
            "template4",
            "template5",
            "random",
        ],
        help="The prompt template to be used for the evaluation, or 'random' for randomly selecting a template",
    )
    parser.add_argument(
        "--shuffle_options",
        action="store_true",
        help="Whether to shuffle the options in the prompt",
    )
    parser.add_argument(
        "--save_dir_path",
        type=str,
        default="./results/",
        help="The path to the directory where the results will be saved",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to be used for the random number generator",
    )

    args = parser.parse_args()

    if args.model_name == "meta-llama/Meta-Llama-3-70B":
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)

    evaluate_prompt(model, tokenizer, args, device)


# CUDA_VISIBLE_DEVICES=0 python evaluate_using_mcqa_prompts.py --model_name "EleutherAI/gpt-neo-125M" --activity_name "riding on a bus" --data_directory_path "../data/causal_query_triplets_sampled_instances/" --shots 0 --prompt_template "template1" --seed 42

# microsoft/phi-2
# CUDA_VISIBLE_DEVICES=0 python evaluate_using_mcqa_prompts.py --model_name "microsoft/phi-2" --activity_name "riding on a bus" --data_directory_path "../data/causal_query_triplets_sampled_instances/" --shots 0 --prompt_template "template1" --seed 42
