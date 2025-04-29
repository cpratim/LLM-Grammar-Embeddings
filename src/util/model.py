from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Gemma2ForCausalLM
import torch
import gc


def load_model(model_id="meta-llama/Llama-3.2-3B", device=None, eager_load=False):

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16,
    )
    if eager_load:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_device_memory_report(device):
    print(f"Device: {device} [{torch.cuda.get_device_name(device)}]")
    free_memory, total_memory = torch.cuda.mem_get_info(device)

    free_memory_gb = free_memory / (1024**3)
    total_memory_gb = total_memory / (1024**3)

    print(
        f"Free Memory: {free_memory_gb:.2f}/{total_memory_gb:.2f} GB [{free_memory / total_memory * 100:.2f}%]"
    )


def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
