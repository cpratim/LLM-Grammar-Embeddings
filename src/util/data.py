import json
import pickle

def load_grammar_data(file_path):
    with open(file_path, 'r') as f:
        grammar_data = [json.loads(line) for line in f]
    return grammar_data


def load_hidden_data(model_name, grammar_name, data_path='../data/blimp-hidden'):
    with open(f'{data_path}/{model_name}/{grammar_name}.pkl', 'rb') as f:
        hidden_data = pickle.load(f)
    return hidden_data


