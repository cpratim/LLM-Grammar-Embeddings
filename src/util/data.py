import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from warnings import filterwarnings

import json
import pandas as pd
import numpy as np
import logging

from tqdm import tqdm
from random import shuffle, uniform
from transformers import pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)
filterwarnings('ignore')

__system_message_multiple_choice = """
You are a helpful assistant that will help me understand which sentence is grammatically correct. 
You should only output the number of the correct sentence: 1 / 2
"""

__system_message_binary = """
You are a helpful assistant that will help me understand if a sentence is grammatically correct. 
You should only output '1' if the sentence is grammatically correct, and '0' if it is not.
"""

__question_format_multiple_choice = """
Which sentence is more grammatical and native-like? 
1) {sentence_1} 
2) {sentence_2}
"""

__question_format_binary = """
Is this sentence grammatically correct? 
1) {sentence_1}
"""


def load_grammar_data(file_path):
    with open(file_path, 'r') as f:
        grammar_data = [json.loads(line) for line in f]
    return grammar_data


def load_LLM_messages_multiple_choice(grammar_data, n_shot=5):

    messages = [
        {
            "role": "system",
            "content": __system_message_multiple_choice
        }
    ]
    shuffle(grammar_data)
    for idx in range(n_shot):
        good_sentence = grammar_data[idx]['sentence_good']
        bad_sentence = grammar_data[idx]['sentence_bad']
        messages.extend([
            {
                "role": "user",
                "content": __question_format_multiple_choice.format(
                    sentence_1=good_sentence, 
                    sentence_2=bad_sentence
                )
            },
            {
                "role": "assistant",
                "content": '1'
            },
            {
                "role": "user",
                "content": __question_format_multiple_choice.format(
                    sentence_1=bad_sentence, 
                    sentence_2=good_sentence
                )
            },
            {
                "role": "assistant",
                "content": '2'
            }
        ])
    return messages


def load_LLM_messages_binary(grammar_data, n_shot=5):
    messages = [
        {
            "role": "system",
            "content": __system_message_binary
        }
    ]
    shuffle(grammar_data)
    for idx in range(n_shot):
        good_sentence = grammar_data[idx]['sentence_good']
        bad_sentence = grammar_data[idx]['sentence_bad']
        messages.extend([
            {
                "role": "user",
                "content": __question_format_binary.format(sentence_1=good_sentence)
            },
            {
                "role": "assistant",
                "content": '1'
            },
            {
                "role": "user",
                "content": __question_format_binary.format(sentence_1=bad_sentence)
            },
            {
                "role": "assistant",
                "content": '0'
            }
        ])
    return messages


def load_multiple_choice_dataset(file_path, model, tokenizer, device, n_shot=5, limit=None):

    with open(file_path, 'r') as f:
        grammar_data = [json.loads(line) for line in f]

    generation_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

    multiple_choice_chat_template = load_LLM_messages_multiple_choice(grammar_data, n_shot=n_shot)
    multiple_choice_dataset = []
    model_outputs = []

    for _, example in enumerate(tqdm(grammar_data[:limit])):
        
        unif = uniform(0, 1)
        sentences = [
            example['sentence_good'],
            example['sentence_bad']
        ]
        
        answer = '1' if unif < 0.5 else '2'
        sentences = sentences if unif < 0.5 else sentences[::-1]

        multiple_choice_chat_template.append({
            "role": "user",
            "content": __question_format_multiple_choice.format(
                sentence_1=sentences[0], 
                sentence_2=sentences[1]
            )
        })        

        formatted_chat = tokenizer.apply_chat_template(
            multiple_choice_chat_template, 
            tokenize=False, 
            add_generation_prompt=True
        )
        tokens = tokenizer(formatted_chat, return_tensors='pt').to(device)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        multiple_choice_chat_template.pop()

        with torch.no_grad():
            forward_pass = model(
                **tokens,
                output_hidden_states=True,
                # output_attentions=True,
                return_dict=True
            )
            output = generation_pipeline(
                formatted_chat,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                max_new_tokens=1
            )
            output = output[0]['generated_text'].strip()[0]

        embedding = forward_pass.hidden_states[-1][:, -1, :].detach().cpu().numpy().flatten()
        
    
        model_outputs.append({
            'embedding': embedding
        })
  
        multiple_choice_dataset.append({
            'good_sentence': example['sentence_good'],
            'bad_sentence': example['sentence_bad'],
            'answer': answer,
            'output': output,
        })

    return multiple_choice_dataset, model_outputs

def load_binary_dataset(file_path, model, tokenizer, device, n_shot=5, limit=None):
    with open(file_path, 'r') as f:
        grammar_data = [json.loads(line) for line in f]

    generation_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

    binary_chat_template = load_LLM_messages_binary(grammar_data, n_shot=n_shot)

    binary_dataset = []
    model_outputs = []

    for _, example in enumerate(tqdm(grammar_data[:limit])):
        good_sentence = example['sentence_good']
        bad_sentence = example['sentence_bad']

        unif = uniform(0, 1)
        answer = '1' if unif < 0.5 else '0'
        sentence = good_sentence if unif < 0.5 else bad_sentence

        binary_chat_template.append({
            "role": "user",
            "content": __question_format_binary.format(sentence_1=sentence)
        })

        formatted_chat = tokenizer.apply_chat_template(
            binary_chat_template, 
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = tokenizer(formatted_chat, return_tensors='pt').to(device)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        binary_chat_template.pop()

        with torch.no_grad():
            forward_pass = model(
                **tokens,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            output = generation_pipeline(
                formatted_chat,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                max_new_tokens=1
            )
            output = output[0]['generated_text'].strip()[0]

        embedding = forward_pass.hidden_states[-1][:, -1, :].detach().cpu().numpy().flatten()
        model_outputs.append({
            'embedding': embedding
        })
        
        binary_dataset.append({
            'good_sentence': example['sentence_good'],
            'bad_sentence': example['sentence_bad'],
            'answer': answer,
            'output': output,
        })

    return binary_dataset, model_outputs


def load_raw_embeddings_dataset(file_path, model, tokenizer, device, limit=None):

    with open(file_path, 'r') as f:
        grammar_data = [json.loads(line) for line in f]

    embeddings = []
    for _, example in enumerate(tqdm(grammar_data[:limit])):

        sentences = [
            example['sentence_good'],
            example['sentence_bad']
        ]

        example_embeddings = {}

        for i, sentence in enumerate(sentences):

            tokens = tokenizer(sentence, return_tensors='pt').to(device)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                forward_pass = model(
                    **tokens,
                    output_hidden_states=True,
                    return_dict=True
                )

            embedding = forward_pass.hidden_states[-1][:, -1, :].detach().cpu().numpy().flatten()
            if i == 0:
                example_embeddings['sentence_good'] = embedding
            else:
                example_embeddings['sentence_bad'] = embedding
        embeddings.append(example_embeddings)
    return embeddings
