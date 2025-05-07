import json
import pickle
import os
import numpy as np
from tqdm import tqdm
from random import uniform

def load_grammar_data(file_path):
    with open(file_path, "r") as f:
        grammar_data = [json.loads(line) for line in f]
    return grammar_data


def load_combined_grammar_data(data_path: str = "../data/blimp", n_sample: int = 100):
    combined_grammar_data = []
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".jsonl"):
            grammar_data = load_grammar_data(os.path.join(data_path, file))
            if n_sample is not None:
                combined_grammar_data.extend(
                    np.random.choice(grammar_data, size=n_sample, replace=False)
                )
            else:
                combined_grammar_data.extend(grammar_data)

    data = []
    for example in combined_grammar_data:
        data.append(example)
    return data


def load_grammar_data_by_category(
    data_path: str = "../data/blimp",
    category: str = "linguistics_term",
    value: str = "island_effects",
    sample_rate: float = 1.0,
    use_tqdm: bool = False,
):
    combined_grammar_data = []
    for file in tqdm(os.listdir(data_path)) if use_tqdm else os.listdir(data_path):
        if file.endswith(".jsonl"):
            grammar_data = load_grammar_data(os.path.join(data_path, file))
            for example in grammar_data:
                if example[category] == value and np.random.rand() < sample_rate:
                    combined_grammar_data.append(example)

    data = []
    for example in combined_grammar_data:
        data.append(
            {
                "good": example["sentence_good"],
                "bad": example["sentence_bad"],
            }
        )
    return data


def load_numerical_data(
    data_path: str = "../data/arithmetic/addition_subtraction.jsonl",
    margin: int = 25,
    use_tqdm: bool = False,
):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    ret_data = []
    for example in data:
        ret_data.append(
            {
                "good": example["good"],
                "bad": (
                    example[f"bad_p_{margin}"]
                    if uniform(0, 1) < 0.5
                    else example[f"bad_m_{margin}"]
                ),
            }
        )
    return ret_data


def load_word_problems_data(
    data_path: str = "../data/word_problems/fill_in_addition_subtraction_simple.jsonl",
    use_tqdm: bool = False,
):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print(line)
    return data


def load_hidden_data(model_name, grammar_name, data_path="../data/blimp-hidden"):
    with open(f"{data_path}/{model_name}/{grammar_name}.pkl", "rb") as f:
        hidden_data = pickle.load(f)
    return hidden_data
