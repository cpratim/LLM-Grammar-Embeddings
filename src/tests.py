# from module.head_select import head_select
from util.data import (
    load_combined_grammar_data,
    load_grammar_data_by_category,
    load_numerical_data,
)
from pprint import pprint
from module.head_select import head_select, head_select_batches, head_select_layers, head_select_layers_remove
from util.model import get_device_memory_report
from module.blocked_attention import load_altered_attention_model, get_model_dimensions
from random import shuffle
import torch
import json

if __name__ == "__main__" or True:
    with open("activated.json", "w") as f:
        json.dump([(0, 0)], f)

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    get_device_memory_report(device)
    model_name = "google/gemma-2-9b-it"
    model, tokenizer = load_altered_attention_model(model_name, device, model_type="gemma")
    numerical_data = load_numerical_data(margin=50)
    grammar_data = load_grammar_data_by_category(
        category="linguistics_term", value="argument_structure", sample_rate=1
    )
    shuffle(numerical_data)
    shuffle(grammar_data)
    # top_heads = head_select(
    #     numerical_data, model, tokenizer, device, metric_type="raw", n_iterations=20, grammar=False, outfile="activated_numerical.json",
    #     mini_batch_size=25,
    # )
    # top_heads = head_select(
    #     numerical_data,
    #     model,
    #     tokenizer,
    #     device,
    #     metric_type="raw",
    #     n_iterations=20,
    #     grammar=False,
    #     outfile="activated_numerical.json",
    #     mini_batch_size=25,
    #     use_system_message=False
    # )
    # top_heads = head_select_batches(
    #     numerical_data, model, tokenizer, device, metric_type="raw", n_iterations=20, grammar=False, outfile="activated_numerical.json",
    #     mini_batch_size=25,
    #     use_system_message=False
    # )
    top_layers = head_select_layers(
        numerical_data, model, tokenizer, device, metric_type="raw", n_iterations=20, grammar=False, outfile="activated_numerical.json",
        mini_batch_size=25,
        use_system_message=False
    )
    # top_layers_remove = head_select_layers_remove(
    #     numerical_data, model, tokenizer, device, metric_type="raw", n_iterations=20, grammar=False, outfile="activated_numerical_remove.json",
    #     mini_batch_size=25,
    #     use_system_message=False
    # )

# 3241853
# 3244778
