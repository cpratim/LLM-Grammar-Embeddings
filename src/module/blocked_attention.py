from module.llama import LlamaBlockedAttention
from module.gemma import GemmaBlockedAttention
from module.qwen import QwenBlockedAttention

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple, Dict, List
from util.model import clear_cuda_cache


def load_altered_attention_model(
    model_id: str,
    device: torch.device,
    disable_idx: Dict[int, List[int]] = {},
    disable_positional_encoding: bool = False,
    model_type: str = "llama",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        device_map=device,
    )
    tokenizer.pad_token = tokenizer.eos_token
    for idx, layer in enumerate(model.model.layers):
        if model_type == "llama":
            blocked_attn = LlamaBlockedAttention(
                layer.self_attn.config,
                idx,
                [] if idx not in disable_idx else disable_idx[idx],
                disable_positional_encoding=disable_positional_encoding,
            )
        elif model_type == "gemma":
            blocked_attn = GemmaBlockedAttention(
                layer.self_attn.config,
                idx,
                [] if idx not in disable_idx else disable_idx[idx],
                disable_positional_encoding=disable_positional_encoding,
            )
        elif model_type == "qwen":
            blocked_attn = QwenBlockedAttention(
                layer.self_attn.config,
                idx,
                [] if idx not in disable_idx else disable_idx[idx],
            )
        blocked_attn.load_state_dict(layer.self_attn.state_dict())
        del layer.self_attn
        layer.self_attn = blocked_attn
        clear_cuda_cache()
    model.to(device)
    return model, tokenizer


def disable_head(model: AutoModelForCausalLM, layer_idx: int, head_idx: int):
    model.model.layers[layer_idx].self_attn.disable_head(head_idx)


def enable_head(model: AutoModelForCausalLM, layer_idx: int, head_idx: int):
    model.model.layers[layer_idx].self_attn.enable_head(head_idx)


def get_model_dimensions(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device
):
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
    input_ids = {k: v.to(device) for k, v in input_ids.items()}
    model.forward(**input_ids, max_new_tokens=10)
    return (
        len(model.model.layers),
        model.model.layers[0].self_attn.get_model_dimensions(),
    )


def get_disabled_heads(model: AutoModelForCausalLM):
    disabled = {}
    for layer in range(len(model.model.layers)):
        disabled_idx = model.model.layers[layer].self_attn.disable_idx
        if len(disabled_idx) > 0:
            disabled[layer] = disabled_idx
    return disabled


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_idx = {
        0: [
            0,
            1,
            2,
            3,
        ]
    }
    model, tokenizer = load_altered_attention_model(model_id, device, disable_idx)
    print(model)
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
    input_ids = {k: v.to(device) for k, v in input_ids.items()}
    output = model.generate(**input_ids, max_new_tokens=10)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
