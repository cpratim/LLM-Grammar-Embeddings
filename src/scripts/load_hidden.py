from util.hidden import load_hidden_states
from util.model import load_model, get_device_memory_report, clear_cuda_cache
import torch
import os
import pickle


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    get_device_memory_report(device)

    output_dir = '../data/blimp-hidden'

    files = [
        '../data/blimp/complex_NP_island.jsonl',
        '../data/blimp/determiner_noun_agreement_1.jsonl',
        '../data/blimp/determiner_noun_agreement_2.jsonl',
        '../data/blimp/adjunct_island.jsonl',
        '../data/blimp/anaphor_number_agreement.jsonl'
    ]

    model_ids = [
        # 'meta-llama/Llama-3.2-1B-Instruct',
        # 'meta-llama/Llama-3.2-3B-Instruct',
        # 'mistralai/Mistral-7B-Instruct-v0.3',
        # 'meta-llama/Llama-3.1-8B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct-1M'
    ]

    for model_id in model_ids:
        print(f'Loading model [{model_id}]')
        print('-' * 100)
        model_name = model_id.split('/')[-1]
        model, tokenizer = load_model(model_id, device, eager_load=True)

        if not os.path.exists(f'{output_dir}/{model_name}'):
            os.makedirs(f'{output_dir}/{model_name}')

        for file in files:
            print(f'Loading hidden states for [{file}]')
            grammar_name = file.split('/')[-1].split('.')[0]
            
            hidden_states = load_hidden_states(file, model, tokenizer, device)
            with open(f'{output_dir}/{model_name}/{grammar_name}.pkl', 'wb') as f:
                pickle.dump(hidden_states, f)

            print(f'Saved {grammar_name}.pkl')
            print('-' * 100)

            clear_cuda_cache()
