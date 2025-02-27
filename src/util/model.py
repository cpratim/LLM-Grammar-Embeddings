from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_id="meta-llama/Llama-3.2-3B"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
        device_map=device
    )
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
