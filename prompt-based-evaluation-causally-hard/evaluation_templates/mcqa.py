import os
import json
import torch

import numpy as np
import pandas as pd

from string import ascii_uppercase
from torch.utils.data import Dataset, DataLoader


def prompt_templates_mcqa():
    """
    Consider the activity of {activity name}.
    [ in-context examples (if few-shot/in-context learning experiment) ]
    Which of the following events (given as options A or B) is a plausible question
    (cause/effect) of the event {premise}?
    A. choice1
    B. choice2
    Answer: A
    The following are multiple choice questions about activity name. You should directly
    answer the question by choosing the correct option.
    [ in-context examples (if few-shot/in-context learning experiment) ]
    Which of the following events (given as options A or B) is a plausible question
    (cause/effect) of the event premise?
    A. choice1
    B. choice2
    Answer: A
    """
    prompt_templates = {
        "template1": lambda activity_name, premise, choices, causal_question: f"Consider the activity of '{activity_name}'. Which of the following events (given as options A or B) is a more plausible {causal_question} of the event '{premise}'?\n" + "\n".join([f"{ascii_uppercase[i]}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:",
        "template2": lambda activity_name, premise, choices, causal_question: f"Consider the activity of '{activity_name}'. Which of the following events (given as options A or B) is a plausible {causal_question} of the event '{premise}'?\n" + "\n".join([f"{ascii_uppercase[i]}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:",
        "template3": lambda activity_name, premise, choices, causal_question: f"While performing the activity '{activity_name}', which of the following events (given as options A or B), will be a more plausible {causal_question} of the event '{premise}'?\n" + "\n".join([f"{ascii_uppercase[i]}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:",
        "template4": lambda activity_name, premise, choices, causal_question: f"In the context of the activity '{activity_name}', which of the following events (given as options A or B) is a plausible {causal_question} of the event '{premise}'?\n" + "\n".join([f"{ascii_uppercase[i]}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:",
        "template5": lambda activity_name, premise, choices, causal_question: f"The following are multiple choice questions about '{activity_name}'. You should directly answer the question by choosing the correct option.\nWhich of the following events (given as options A or B) is a plausible {causal_question} of the event '{premise}'?\n" + "\n".join([f"{ascii_uppercase[i]}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:",
    }
    return prompt_templates



def get_question_text(activity_name, premise, choices, causal_question, template="template1"):
    prompt_templates = prompt_templates_mcqa()
    if template == "random":
        template_id = np.random.choice(list(prompt_templates.keys()))
    else:
        template_id = template
    template = prompt_templates[template_id]
    template_text = template(activity_name, premise, choices, causal_question)
    return template_text



class mcqa(Dataset):
    def __init__(self, dataset_path, prompt_template, shots=0, shuffle_options=True, n_options=2, seed=42):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset(dataset_path)
        self.prompt_template = prompt_template
        self.shuffle_options = shuffle_options
        self.n_options = n_options
        self.seed = seed
        np.random.seed(seed)
        self.activity_name_in_prompt = self.dataset_path.split("/")[-1].replace(".csv", "").replace("_", " ")
        # shuffle the dataset
        print(f"Shuffling dataset {dataset_path}")
        np.random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # {'Unnamed: 0': 5268, 'premise': 'go in store.', 'choice1': 'drive to store.', 'choice2': 'bagger bags groceries.', 'label': 0, 'question': 'cause'}
        # activity_name, premise, choices, causal_question, template="template1"
        premise, choices, causal_question = item["premise"], [item["choice1"], item["choice2"]], item["question"]
        if self.shuffle_options:
            correct_choice = choices[item["label"]]
            np.random.shuffle(choices)
            label = choices.index(correct_choice)

        prompt = get_question_text(
            activity_name=self.activity_name_in_prompt,
            premise=premise,
            choices=choices,
            causal_question=causal_question,
            template=self.prompt_template,
        )
        label = item["label"]
        return prompt, label

    def load_dataset(self, dataset_path):
        if dataset_path.endswith(".json"):
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
        elif dataset_path.endswith(".csv"):
            dataset = pd.read_csv(dataset_path)
            dataset = dataset.to_dict("records")
        return dataset

    def format_prompt(self, question, options, answer, correct_option):
        try:
            correct_option = options.index(answer)
        except ValueError:
            raise ValueError(
                f"Answer {answer} not found in options {options} for question {question} in dataset {self.dataset_path}"
            )
        # correct_option = ord(correct_option) - 65

        wrong_options = [i for i in range(self.n_options) if i != correct_option]

        # sample wrong options based on n_options
        wrong_options = np.random.choice(
            wrong_options, self.n_options - 1, replace=False
        )
        options = [options[correct_option]] + [
            options[wrong_option] for wrong_option in wrong_options
        ]

        if self.shuffle_options:
            np.random.shuffle(options)

        correct_option = options.index(answer)

        prompt_template_text = f"Question: {question} \n"
        for i, option in enumerate(options):
            # A. option1 \n B. option2 \n C. option3 \n D. option4 \n
            prompt_template_text += f"{chr(65 + i)}. {option}\n"
        prompt_template_text += "Answer:"

        label = chr(65 + correct_option)

        return prompt_template_text, label

class PredictionSample(object):
    def __init__(
        self,
        prompt,
        correct_choice,
        generated_argmax_token,
        predicted_argmax_token_based_choice,
        choice_A_logit_value,
        choice_B_logit_value,
        predicted_logit_based_choice,
    ):
        self.prompt = prompt
        self.correct_choice = correct_choice
        self.generated_argmax_token = generated_argmax_token
        self.predicted_argmax_token_based_choice = predicted_argmax_token_based_choice
        self.choice_A_logit_value = choice_A_logit_value
        self.choice_B_logit_value = choice_B_logit_value
        self.predicted_logit_based_choice = predicted_logit_based_choice

        
    def __repr__(self):
        return f"Prompt: {self.prompt}\nCorrect Choice: {self.correct_choice}\nGenerated Argmax Token: {self.generated_argmax_token}\nPredicted Argmax Token Based Choice: {self.predicted_argmax_token_based_choice}\nChoice A Logit Value: {self.choice_A_logit_value}\nChoice B Logit Value: {self.choice_B_logit_value}\nPredicted Logit Based Choice: {self.predicted_logit_based_choice}"

    def __str__(self):
        return f"Prompt: {self.prompt}\nCorrect Choice: {self.correct_choice}\nGenerated Argmax Token: {self.generated_argmax_token}\nPredicted Argmax Token Based Choice: {self.predicted_argmax_token_based_choice}\nChoice A Logit Value: {self.choice_A_logit_value}\nChoice B Logit Value: {self.choice_B_logit_value}\nPredicted Logit Based Choice: {self.predicted_logit_based_choice}"

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "correct_choice": self.correct_choice,
            "generated_argmax_token": self.generated_argmax_token,
            "predicted_argmax_token_based_choice": self.predicted_argmax_token_based_choice,
            "choice_A_logit_value": self.choice_A_logit_value,
            "choice_B_logit_value": self.choice_B_logit_value,
            "predicted_logit_based_choice": self.predicted_logit_based_choice,
        }
    
    
class PerformanceTrackerMCQA(object):
    """
    Class to keep track of the performance of a model on multiple choice question answering tasks
    the probability over the options is used to determine the predicted label
    """

    def __init__(
        self,
    ):
        self.tracker = {
            "total": 0,
            "correct_argmax_based_count": 0,
            "no_option_in_argmax_count": 0,
            "correct_logit_based_count": 0,
            "accuracy": 0.0,
            "success_rate": 0.0,
            "total_per_class": {},
            "correct_per_class": {},
            "accuracy_per_class": {},
        }

        # save the evaluated samples for further analysis
        self.evaluated_samples = {}
        
    def update(self, prompt, correct_choice, generated_argmax_token, choice_A_logit, choice_B_logit):
        self.tracker["total"] += 1
        self.tracker["total_per_class"][correct_choice] = self.tracker["total_per_class"].get(correct_choice, 0) + 1
        
        if "A" in generated_argmax_token:
            predicted_argmax_token_based_choice = "A"
        elif "B" in generated_argmax_token:
            predicted_argmax_token_based_choice = "B"
        else:
            predicted_argmax_token_based_choice = -1

        if choice_A_logit > choice_B_logit:
            predicted_logit_based_choice = "A"
        else:
            predicted_logit_based_choice = "B"

        # compute success rate based on the argmax token
        if predicted_argmax_token_based_choice == correct_choice:
            self.tracker["correct_argmax_based_count"] += 1
        elif predicted_argmax_token_based_choice == -1:
            self.tracker["no_option_in_argmax_count"] += 1

        # compute accuracy based on the logit values
        if predicted_logit_based_choice == correct_choice:
            self.tracker["correct_logit_based_count"] += 1
            self.tracker["correct_per_class"][correct_choice] = self.tracker["correct_per_class"].get(correct_choice, 0) + 1

        # save the evaluated samples for further analysis
        self.evaluated_samples[self.tracker["total"]] = PredictionSample(
            prompt,
            correct_choice,
            generated_argmax_token,
            predicted_argmax_token_based_choice,
            choice_A_logit,
            choice_B_logit,
            predicted_logit_based_choice,
        )


    def get_performance(self):
        self.tracker["accuracy"] = self.tracker["correct_logit_based_count"] / self.tracker["total"]
        self.tracker["success_rate"] = (self.tracker["correct_argmax_based_count"] + 0.5 * self.tracker["no_option_in_argmax_count"]) / self.tracker["total"]
        return self.tracker["accuracy"], self.tracker["success_rate"]
    
    def get_performance_per_class(self):
        for label in self.tracker["total_per_class"]:
            if label not in self.tracker["correct_per_class"]:
                self.tracker["correct_per_class"][label] = 0
            self.tracker["accuracy_per_class"][label] = self.tracker["correct_per_class"][label] / self.tracker["total_per_class"][label]
        return self.tracker["accuracy_per_class"]
    
    def reset(self):
        self.tracker = {
            "total": 0,
            "correct_argmax_based_count": 0,
            "no_option_in_argmax_count": 0,
            "correct_logit_based_count": 0,
            "accuracy": 0.0,
            "success_rate": 0.0,
            "total_per_class": {},
            "correct_per_class": {},
            "accuracy_per_class": {},
        }
        return self
    
    def save(self, save_file_path):
        # save tracker to a json file
        with open(save_file_path, "w") as f:
            json.dump(self.tracker, f)
        
        # save evaluated samples to a json file
        evaluated_samples = {k: v.to_dict
                                () for k, v in self.evaluated_samples.items()}
        with open(save_file_path.replace(".json", "_evaluatedSamples.json"), "w") as f:
            json.dump(evaluated_samples, f)
        return self
    
    def print_results(self, verbose=True):
        accuracy, success_rate = self.get_performance()
        accuracy_per_class = self.get_performance_per_class()
        if verbose:
            print(f"Accuracy: {accuracy}")
            print(f"Success Rate: {success_rate}")
            print(f"Success Rate per class: {accuracy_per_class}")
        return accuracy, success_rate, accuracy_per_class
    
    

