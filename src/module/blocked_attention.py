from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaAttention, 
    apply_rotary_pos_emb, 
    eager_attention_forward, 
    logger, 
    ALL_ATTENTION_FUNCTIONS
)

from typing import Tuple, Optional, Unpack, Callable, Dict, List
import torch

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


class CustomAttention(LlamaAttention):
    def __init__(self, config, layer_idx, disable_idx=[]):
        super().__init__(config, layer_idx)
        self.disable_idx = disable_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        
        if self.disable_idx:
            attn_output[:, :, self.disable_idx, :] = 0

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    

def load_blocked_attention_model(model_id: str, disable_idx: Dict[int, List[int]], device: torch.device) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn = CustomAttention(layer.self_attn.config, idx, [] if idx not in disable_idx else disable_idx[idx])
    model.to(device)
    return model, tokenizer

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_idx = {0: [0, 1, 2, 3,]}
    model, tokenizer = load_blocked_attention_model(model_id, disable_idx, device)
    print(model)