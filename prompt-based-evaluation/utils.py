import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTJForCausalLM


def load_model_and_tokenizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name == "EleutherAI/gpt-neo-125M":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    elif args.model_name == "EleutherAI/gpt-neo-1.3B":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    elif args.model_name == "EleutherAI/gpt-neo-2.7B":
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif args.model_name == "EleutherAI/gpt-j-6B":
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    elif args.model_name == "NousResearch/Llama-2-7b-chat-hf":
        PATH_TO_CONVERTED_WEIGHTS = "NousResearch/Llama-2-7b-chat-hf"
        model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(device)
        # model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(device)
        tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    elif args.model_name == "mistralai/Mistral-7B-v0.1":
        PATH_TO_CONVERTED_WEIGHTS = "mistralai/Mistral-7B-v0.1"
        model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    elif args.model_name == "meta-llama/Meta-Llama-3-8B":
        PATH_TO_CONVERTED_WEIGHTS = "meta-llama/Meta-Llama-3-8B"
        model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(device)
        tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    elif args.model_name == "google/gemma-2b":
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2b").to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    elif args.model_name == "google/gemma-7b":
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b").to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    elif args.model_name == "microsoft/phi-2":
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    else:
        raise ValueError("Model not found")

    model = model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
