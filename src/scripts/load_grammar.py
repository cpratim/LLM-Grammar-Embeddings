from util.grammar import load_multiple_choice_dataset, load_binary_dataset, load_raw_embeddings_dataset
from util.model import load_model, get_device_memory_report
import torch
import pickle
import os
import json

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    get_device_memory_report(device)

    output_dir = '../data/blimp-output'

    files = [
        '../data/blimp/complex_NP_island.jsonl',
        # '../data/blimp/determiner_noun_agreement_1.jsonl',
        # '../data/blimp/determiner_noun_agreement_2.jsonl',
        # '../data/blimp/adjunct_island.jsonl',
        # '../data/blimp/anaphor_number_agreement.jsonl'
    ]

    model_ids = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct'
    ]

    n_shots = [
        0, 1, 5, 10
    ]

    LIMIT = None

    for model_id in model_ids:
        model_name = model_id.split('/')[-1]

        model, tokenizer = load_model(model_id, device)

        if not os.path.exists(f'{output_dir}/{model_name}'):
            os.makedirs(f'{output_dir}/{model_name}')

        tokenizer.pad_token = tokenizer.eos_token 
        for file in files:
            grammar_name = file.split('/')[-1].split('.')[0]

            if not os.path.exists(f'{output_dir}/{model_name}/{grammar_name}'):
                os.makedirs(f'{output_dir}/{model_name}/{grammar_name}')

            raw_embeddings = load_raw_embeddings_dataset(file, model, tokenizer, device, limit=LIMIT)  
            with open(f'{output_dir}/{model_name}/{grammar_name}/no_prompt_embeddings.pkl', 'wb') as f:
                pickle.dump(raw_embeddings, f)

            print(f'Saved {grammar_name}/no_prompt_embeddings.pkl')
            print('-' * 100)

            torch.cuda.empty_cache()
        
            for n_shot in n_shots:
                multiple_choice_dataset, multiple_choice_model_outputs = load_multiple_choice_dataset(file, model, tokenizer, device, n_shot, limit=LIMIT)
                pkl_output_file = f'{output_dir}/{model_name}/{grammar_name}/multiple_choice_{n_shot}_shot_embeddings.pkl'
                with open(pkl_output_file, 'wb') as f:
                    pickle.dump(multiple_choice_model_outputs, f)

                json_output_file = f'{output_dir}/{model_name}/{grammar_name}/multiple_choice_{n_shot}_shot_dataset.json'
                with open(json_output_file, 'w') as f:
                    json.dump(multiple_choice_dataset, f, indent=4)

                print(f'Saved {pkl_output_file}')
                print(f'Saved {json_output_file}')
                print('-' * 100)

                binary_dataset, binary_model_outputs = load_binary_dataset(file, model, tokenizer, device, n_shot, limit=LIMIT)
                pkl_output_file = f'{output_dir}/{model_name}/{grammar_name}/binary_{n_shot}_shot_embeddings.pkl'
                with open(pkl_output_file, 'wb') as f:
                    pickle.dump(binary_model_outputs, f)

                json_output_file = f'{output_dir}/{model_name}/{grammar_name}/binary_{n_shot}_shot_dataset.json'
                with open(json_output_file, 'w') as f:
                    json.dump(binary_dataset, f, indent=4)

                print(f'Saved {pkl_output_file}')
                print(f'Saved {json_output_file}')
                print('-' * 100)
        
                torch.cuda.empty_cache()
            
            torch.cuda.empty_cache()
        
        torch.cuda.empty_cache()


