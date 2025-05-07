from module.heads import head_select, remove_high_entopy_heads
from module.blocked_attention import load_altered_attention_model
from util.data import load_numerical_data, load_grammar_data_by_category, load_word_problems_data
import torch
from random import shuffle


if __name__ == "__main__":
    numerical_data = load_numerical_data(margin=50)
    shuffle(numerical_data)
    numerical_data = numerical_data[:100]
    grammar_data = load_grammar_data_by_category(
        category="linguistics_term", value="argument_structure", sample_rate=1
    )
    shuffle(grammar_data)
    grammar_data = grammar_data[:100]
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_altered_attention_model(
        "google/gemma-2-9b-it", device, model_type="gemma"
    )
    word_problems_data = load_word_problems_data()

    target_layers = [0, 1, 2, 3, ]
    target_heads = {
        0: [2, 4, 5, 6, 9, 11, 13, 14],
        1: [1, 3, 4, 5, 8, 9, 10, 11, 12, 15],
        2: [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        3: [0, 1, 2, 3, 4, 5, 9, 11, 12],
        27: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        28: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15],
        29: [0, 1, 4, 5, 6, 10, 15],
        38: [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15],
        39: [1, 2, 4, 5, 6, 10, 11, 12, 13, 14]
    }
    target_heads = {
        l: [i for i in range(16)] for l in target_layers
    }
    
    remove_high_entopy_heads(
        word_problems_data,
        model,
        tokenizer,
        device,
        target_heads=target_heads,
        metric_type="raw",
        problem_type="word_problems",
        percentage_to_remove=1,
        use_system_message=False,
        head_batch_size=10,
        # outfile="removed_layers_numerical.json",
    )
    # attentions = get_head_attentions_dataset(
    #     numerical_data,
    #     model,
    #     tokenizer,
    #     device,
    #     types_=["raw"],
    #     use_tqdm=True,
    #     grammar=False,
    #     use_system_message=False,
    # )
    # print(attentions['raw'][0]['attentions'].shape)