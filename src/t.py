from module.blocked_attention import load_altered_attention_model
from util.data import load_numerical_data, load_grammar_data_by_category
from util.numerical import load_word_problems_data
import torch
from random import shuffle
from module.algo import layer_select, head_select
import json
from pprint import pprint
import os


if __name__ == "__main__":
    

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    # Load the model
    # model_name = "qwen/qwen3-8b"
    # model_type = "qwen"
    # model_type = "gemma"
    # model_name = "google/gemma-2-9b-it"
    model_type = "llama"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model, tokenizer = load_altered_attention_model(
        model_name, device, model_type=model_type
    )

    for i in range(10):
        numerical_data = load_numerical_data(margin=50)
        shuffle(numerical_data)
        numerical_data = numerical_data[:50]

        quantifiers_data = load_grammar_data_by_category(
            category="linguistics_term", value="quantifiers", sample_rate=1
        )
        shuffle(quantifiers_data)
        quantifiers_data = quantifiers_data[:50]

        word_problems_data = load_word_problems_data()
        shuffle(word_problems_data)
        word_problems_data = word_problems_data[:50]

        data = {
            "numerical": numerical_data,
            "quantifiers": quantifiers_data,
            "word_problems": word_problems_data,
        }
    
        for problem_type, data in data.items():
            layer_data_file = f"layers/{model_type}/{problem_type}_{i}.json"
            if not os.path.exists(layer_data_file):
                layer_data = layer_select(
                    data,
                    model,
                    tokenizer,
                    device,
                    metric_type="raw",
                    problem_type=problem_type,
                    k=5,
                    outfile=f"layers/{model_type}/{problem_type}_{i}.json",
                )

            else:
                with open(layer_data_file, "r") as f:
                    layer_data = json.load(f)

            head_select(
                data,
                model, 
                tokenizer, 
                device, 
                epsilon=.1,
                n_splits=10,
                layer_data=layer_data,
                outfile=f"heads/{model_type}/{problem_type}_{i}.json",
            )

