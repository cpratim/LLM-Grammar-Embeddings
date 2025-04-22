import json
import pickle
import os
import numpy as np
from tqdm import tqdm


def load_grammar_data(file_path):
    with open(file_path, 'r') as f:
        grammar_data = [json.loads(line) for line in f]
    return grammar_data


def load_combined_grammar_data(data_path: str = '../data/blimp', n_sample: int = 100):
    combined_grammar_data = []
    for file in tqdm(os.listdir(data_path)):
        if file.endswith('.jsonl'):
            grammar_data = load_grammar_data(os.path.join(data_path, file))
            if n_sample is not None:
                combined_grammar_data.extend(np.random.choice(grammar_data, size=n_sample, replace=False))
            else:
                combined_grammar_data.extend(grammar_data)
    return combined_grammar_data


def load_grammar_data_by_category(data_path: str = '../data/blimp', category: str = 'linguistics_term', value: str = 'island_effects', sample_rate: float = 1.0, use_tqdm: bool = False):
    combined_grammar_data = []
    for file in tqdm(os.listdir(data_path)) if use_tqdm else os.listdir(data_path):
        if file.endswith('.jsonl'):
            grammar_data = load_grammar_data(os.path.join(data_path, file))
            for example in grammar_data:
                if example[category] == value and np.random.rand() < sample_rate:
                    combined_grammar_data.append(example)
    return combined_grammar_data


def load_hidden_data(model_name, grammar_name, data_path='../data/blimp-hidden'):
    with open(f'{data_path}/{model_name}/{grammar_name}.pkl', 'rb') as f:
        hidden_data = pickle.load(f)
    return hidden_data


