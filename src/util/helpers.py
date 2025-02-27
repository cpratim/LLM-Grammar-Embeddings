import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import json
import os
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def load_blimp_(grammar_file):
    with open(grammar_file, 'r') as f:
        data = [json.loads(line) for line in f]

    positive = [l['sentence_good'] for l in data]
    negative = [l['sentence_bad'] for l in data]

    return positive, negative


def get_blimp_dataset(data_dir='../data/blimp/'):
    
    positive, negative = [], []
    for file in os.listdir(data_dir):
        if file.endswith('.jsonl'):
            with open(os.path.join(data_dir, file), 'r') as f:
                data = [json.loads(line) for line in f]

            positive.extend([l['sentence_good'] for l in data])
            negative.extend([l['sentence_bad'] for l in data])
    return positive, negative


def load_model(model_id="meta-llama/Llama-3.2-1B", device="cuda:1"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
        device_map=device,
        attn_implementation="eager"
    )
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_dataset_from_file(grammar_file, model, tokenizer, add_grammar_prompt=False, batch_size=100):

    with open(grammar_file, 'r') as f:
        data = [json.loads(line) for line in f]

    positive = [l['sentence_good'] for l in data]
    negative = [l['sentence_bad'] for l in data]

    if add_grammar_prompt:
        positive = [
            "Output 'YES' if the following sentence is grammatically correct, otherwise output 'NO': " + p for p in positive
        ]
        negative = [
            "Output 'YES' if the following sentence is grammatically correct, otherwise output 'NO': " + n for n in negative
        ]

    positive_encodings, negative_encodings = [], []

    for i in tqdm(range(0, len(positive), batch_size)):

        with torch.no_grad():
            positive_tokens = tokenizer(positive[i:i+batch_size], return_tensors='pt', padding=True, truncation=True)
            positive_tokens = {k: v.to(device) for k, v in positive_tokens.items()}

            positive_outputs = model(
                **positive_tokens,
                max_length=1,
                output_hidden_states=True,
                return_dict=True
            )
            positive_hidden_states = positive_outputs.hidden_states[-1][:, -1, :]

            negative_tokens = tokenizer(negative[i:i+batch_size], return_tensors='pt', padding=True, truncation=True)
            negative_tokens = {k: v.to(device) for k, v in negative_tokens.items()}
            negative_outputs = model(
                **negative_tokens,
                max_length=1,
                output_hidden_states=True,
                return_dict=True
            )
            negative_hidden_states = negative_outputs.hidden_states[-1][:, -1, :]
            
            positive_encodings.append(positive_hidden_states)
            negative_encodings.append(negative_hidden_states)

    positive_encodings = torch.cat(positive_encodings, dim=0)
    negative_encodings = torch.cat(negative_encodings, dim=0)
    
    return positive_encodings, negative_encodings


def load_complete_dataset(data_dir='../data/blimp/', add_grammar_prompt=False, batch_size=100, n_files=None):
    model, tokenizer = load_model()
    
    positive_encodings, negative_encodings = [], []
    for file in os.listdir(data_dir)[:n_files]:
        if file.endswith('.jsonl'):
            positive, negative = load_dataset_from_file(os.path.join(data_dir, file), model, tokenizer, add_grammar_prompt, batch_size)
            positive_encodings.append(positive)
            negative_encodings.append(negative)

    positive_encodings = torch.cat(positive_encodings, dim=0)
    negative_encodings = torch.cat(negative_encodings, dim=0)

    return positive_encodings, negative_encodings
