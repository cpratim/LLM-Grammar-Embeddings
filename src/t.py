from module.blocked_attention import load_altered_attention_model
from util.data import load_numerical_data, load_grammar_data_by_category
from util.numerical import load_word_problems_data
import torch
from random import shuffle
from module.algo import layer_select, head_select
from util.model import clear_cuda_cache
import json
from pprint import pprint
import os
import sys
import uuid

if __name__ == "__main__":
    
    idx = 0
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")

    while True:
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        if free_memory / total_memory < 0.9:
            idx += 1
            device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
        else:
            break

    print(f"Using device: {device}")

    model_type = sys.argv[1]
    problem_type = sys.argv[2]

    
    if model_type == "gemma":
        model_name = "google/gemma-2-9b-it"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    model, tokenizer = load_altered_attention_model(
        model_name, device, model_type=model_type
    )

    iterations = 10
    batch_size = 50
    print(f"Running {model_type} model")
    for i in range(iterations):
        numerical_data = load_numerical_data(margin=50)
        shuffle(numerical_data)
        numerical_data = numerical_data[:batch_size]

        determiner_noun_agreement_data = load_grammar_data_by_category(
            category="linguistics_term", value="determiner_noun_agreement", sample_rate=1
        )
        shuffle(determiner_noun_agreement_data)
        determiner_noun_agreement_data = determiner_noun_agreement_data[:batch_size]

        word_problems_data = load_word_problems_data()
        shuffle(word_problems_data)
        word_problems_data = word_problems_data[:batch_size]

        data = {
            "grammar": determiner_noun_agreement_data,
            "word_problems": word_problems_data,
            "arithmetic": numerical_data,
        }
    
        if model_type not in os.listdir("layers"):
            os.makedirs(f"layers/{model_type}")
        if model_type not in os.listdir("heads"):
            os.makedirs(f"heads/{model_type}")

        uid = uuid.uuid4()
        print(f"Running {model_type} model for {problem_type} with uid {uid}")

        layer_data_file = f"layers/{model_type}/{problem_type}_{uid}.json"
        if not os.path.exists(layer_data_file):
            layer_data = layer_select(
                data[problem_type],
                model,
                tokenizer,
                device,
                metric_type="raw",
                problem_type=problem_type,
                k=5,
                outfile=layer_data_file,
            )

        else:
            with open(layer_data_file, "r") as f:
                layer_data = json.load(f)

        clear_cuda_cache()
        head_data_file = f"heads/{model_type}/{problem_type}_{uid}.json"
        head_data = None
        if os.path.exists(head_data_file):
            with open(head_data_file, "r") as f:
                head_data = json.load(f)
        if not head_data or head_data['terminated'] == False:
            head_select(
                data[problem_type],
                model, 
                tokenizer, 
                device, 
                epsilon=.25,
                n_splits=10,
                T=10,
                layer_data=layer_data,
                metric_type="raw",
                problem_type=problem_type,
                outfile=head_data_file,
            )

        clear_cuda_cache()