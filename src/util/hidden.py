from util.data import load_grammar_data
from util.model import clear_cuda_cache
from tqdm import tqdm
import numpy as np

def load_hidden_states(file_path, model, tokenizer, device, batch_size=50):

    grammar_data = load_grammar_data(file_path)
    hidden_states = []
    
    for i in tqdm(range(0, len(grammar_data), batch_size)):

        batch = grammar_data[i:i+batch_size]
        good_sentences = [example['sentence_good'] for example in batch]
        bad_sentences = [example['sentence_bad'] for example in batch]

        good_tokens = tokenizer(good_sentences, return_tensors='pt', padding=True).to(device)
        bad_tokens = tokenizer(bad_sentences, return_tensors='pt', padding=True).to(device)

        good_tokens = {k: v.to(device) for k, v in good_tokens.items()}
        bad_tokens = {k: v.to(device) for k, v in bad_tokens.items()}

        good_outputs = model(
            **good_tokens, 
            output_hidden_states=True, 
            output_attentions=True,
            return_dict=True
        )
        bad_outputs = model(
            **bad_tokens, 
            output_hidden_states=True, 
            output_attentions=True, 
            return_dict=True
        )

        good_hidden_states = [v.detach().cpu().numpy() for v in good_outputs.hidden_states]
        bad_hidden_states = [v.detach().cpu().numpy() for v in bad_outputs.hidden_states]

        good_hidden_states = np.swapaxes(good_hidden_states, 0, 1)
        bad_hidden_states = np.swapaxes(bad_hidden_states, 0, 1)
        
        good_attentions = [v.detach().cpu().numpy() for v in good_outputs.attentions]
        bad_attentions = [v.detach().cpu().numpy() for v in bad_outputs.attentions]

        good_attentions = np.swapaxes(good_attentions, 0, 1)
        bad_attentions = np.swapaxes(bad_attentions, 0, 1)

        for i in range(len(batch)):
            hidden_states.append({
                'good_sentence': good_sentences[i],
                'bad_sentence': bad_sentences[i],
                'good_hidden_states': good_hidden_states[i],
                'bad_hidden_states': bad_hidden_states[i],
                'good_attention_states': good_attentions[i],
                'bad_attention_states': bad_attentions[i],
            })

        clear_cuda_cache()

    return hidden_states
