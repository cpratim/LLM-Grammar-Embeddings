from module.blocked_attention import load_altered_attention_model
from util.data import load_numerical_data, load_grammar_data_by_category, load_word_problems_data
import torch
from random import shuffle
from module.algo import layer_select, head_select
import json
from pprint import pprint

if __name__ == "__main__":
    numerical_data = load_numerical_data(margin=50)
    shuffle(numerical_data)
    numerical_data = numerical_data[:50]

    noun_agreement_data = load_grammar_data_by_category(
        category="linguistics_term", value="determiner_noun_agreement", sample_rate=1
    )
    shuffle(noun_agreement_data)
    noun_agreement_data = noun_agreement_data[:50]

    word_problems_data = load_word_problems_data()

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_altered_attention_model(
        "google/gemma-2-9b-it", device, model_type="gemma"
    )
    with open("outputs/gemma_removed_layers_numerical.json", "r") as f:
        numerical_layer_data = json.load(f)

    with open("outputs/gemma_removed_layers_word_problems.json", "r") as f:
        word_problems_layer_data = json.load(f)

    with open("outputs/gemma_removed_layers_noun_agreement.json", "r") as f:
        noun_agreement_layer_data = json.load(f)

    head_select(
        word_problems_data,
        model,
        tokenizer,
        device,
        layer_data=word_problems_layer_data,
        metric_type="raw",
        problem_type="word_problems",
        use_system_message=False,
        outfile="outputs/gemma_removed_head_pool_word_problems.json",
    )

    # layer_select(
    #     numerical_data,
    #     model,
    #     tokenizer,
    #     device,
    #     metric_type="raw",
    #     problem_type="numerical",
    #     use_system_message=False,
    #     percentages=[1/20],
    #     outfile="outputs/gemma_removed_layers_numerical.json",
    # )
    # layer_select(
    #     word_problems_data,
    #     model,
    #     tokenizer,
    #     device,
    #     metric_type="raw",
    #     problem_type="word_problems",
    #     use_system_message=False,
    #     percentages=[1/3, 1/4, 1/5, 1/8, 1/10, 1/20],
    #     outfile="outputs/llama_removed_layers_word_problems.json",
    # )
    # layer_select(
    #     noun_agreement_data,
    #     model,
    #     tokenizer,
    #     device,
    #     metric_type="raw",
    #     problem_type="determiner_noun_agreement",
    #     use_system_message=False,
    #     percentages=[1/3, 1/4, 1/5, 1/10, 1/20],
    #     outfile="outputs/llama_removed_layers_determiner_noun_agreement.json",
    # )