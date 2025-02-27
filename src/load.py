from util.data import load_multiple_choice_dataset, load_binary_dataset, load_raw_embeddings_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
import os
import json

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def get_device_memory_report(device):
    print(f'Device: {device} [{torch.cuda.get_device_name(device)}]')
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    
    free_memory_gb = free_memory / (1024 ** 3)
    total_memory_gb = total_memory / (1024 ** 3)
    
    print(f"Free Memory: {free_memory_gb:.2f}/{total_memory_gb:.2f} GB [{free_memory / total_memory * 100:.2f}%]")


if __name__ == "__main__":
    get_device_memory_report(device)

    output_dir = '../data/blimp-output'

    files = [
        '../data/blimp/anaphor_number_agreement.jsonl'
    ]

    model_ids = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct'
    ]

    n_shots = [
        1, 5, 10, 20
    ]

    for model_id in model_ids:
        model_name = model_id.split('/')[-1]

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
        )
        tokenizer.pad_token = tokenizer.eos_token 
        for file in files:
            file_name = file.split('/')[-1].split('.')[0]

            raw_embeddings = load_raw_embeddings_dataset(file, model, tokenizer, device, limit=None)  
            with open(f'{output_dir}/{model_name}/{file_name}_raw_embeddings.pkl', 'wb') as f:
                pickle.dump(raw_embeddings, f)

            print(f'Saved {file_name}_raw_embeddings.pkl')
            print('-' * 100)

            torch.cuda.empty_cache()
        
            for n_shot in n_shots:
                multiple_choice_dataset, multiple_choice_model_outputs = load_multiple_choice_dataset(file, model, tokenizer, device, n_shot, limit=None)

                model_output_dir = f'{output_dir}/{model_name}/'
                if not os.path.exists(model_output_dir):
                    os.makedirs(model_output_dir)

                pkl_output_file = f'{model_output_dir}/{file_name}_multiple_choice_{n_shot}_shot_embeddings.pkl'
                with open(pkl_output_file, 'wb') as f:
                    pickle.dump(multiple_choice_model_outputs, f)

                json_output_file = f'{model_output_dir}/{file_name}_multiple_choice_{n_shot}_shot_dataset.json'
                with open(json_output_file, 'w') as f:
                    json.dump(multiple_choice_dataset, f, indent=4)

                print(f'Saved {pkl_output_file}')
                print(f'Saved {json_output_file}')
                print('-' * 100)

                binary_dataset, binary_model_outputs = load_binary_dataset(file, model, tokenizer, device, n_shot, limit=None)
                
                pkl_output_file = f'{model_output_dir}/{file_name}_binary_{n_shot}_shot_embeddings.pkl'
                with open(pkl_output_file, 'wb') as f:
                    pickle.dump(binary_model_outputs, f)

                json_output_file = f'{model_output_dir}/{file_name}_binary_{n_shot}_shot_dataset.json'
                with open(json_output_file, 'w') as f:
                    json.dump(binary_dataset, f, indent=4)

                print(f'Saved {pkl_output_file}')
                print(f'Saved {json_output_file}')
                print('-' * 100)
        
                torch.cuda.empty_cache()
            
            torch.cuda.empty_cache()
        
        torch.cuda.empty_cache()


