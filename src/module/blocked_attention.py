from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
    ALL_ATTENTION_FUNCTIONS,
    repeat_kv,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
)
import numpy as np
from typing import Tuple, Optional, Unpack, Callable, Dict, List, Set
import torch
import torch.nn as nn
from util.model import clear_cuda_cache

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


def print_matrix(matrix: np.ndarray):
    for row in matrix:
        for col in row:
            print(f"{col:.2f}", end=" ")
        print()


def increase_entropy(distribution, power=2, percentile=0.2):
    mean = torch.mean(distribution)
    deviation = distribution - mean

    over_mean = torch.clamp(torch.sign(deviation) * (deviation) ** power, 0, 1)
    under_mean = torch.clamp(torch.sign(deviation) * (deviation) ** power, -1, 0)

    over_sum = torch.sum(over_mean)
    under_sum = torch.sum(under_mean)

    sub = torch.zeros_like(distribution)
    add = torch.zeros_like(distribution)

    if over_sum > 0:
        sub = (over_mean / over_sum) * percentile

    if under_sum < 0:
        add = (under_mean / under_sum) * percentile

    new_dist = distribution - sub + add
    return new_dist


def increase_entropy_batch(attn_weights, power=2, percentile=0.1):
    # Extract the last row for all samples and heads
    last_row = attn_weights[:, :, -1, :]
    # Calculate deviation from mean (along the token dimension)
    mean = torch.mean(last_row, dim=-1, keepdim=True)
    deviation = last_row - mean
    # Calculate components above and below mean
    over_mean = torch.clamp(torch.sign(deviation) * (deviation) ** power, 0, 1)
    under_mean = torch.clamp(torch.sign(deviation) * (deviation) ** power, -1, 0)
    # Handle potential division by zero with safe division
    over_sum = torch.sum(over_mean, dim=-1, keepdim=True)
    under_sum = torch.sum(under_mean, dim=-1, keepdim=True)
    # Initialize sub and add tensors
    sub = torch.zeros_like(last_row)
    add = torch.zeros_like(last_row)
    # Safely divide using where operation instead of masking
    sub = torch.where(
        over_sum > 0,
        (over_mean / over_sum.clamp(min=1e-10)) * percentile,
        torch.zeros_like(over_mean),
    )
    add = torch.where(
        under_sum < 0,
        (under_mean / under_sum.clamp(max=-1e-10)) * percentile,
        torch.zeros_like(under_mean),
    )
    # Calculate new distribution
    new_last_row = last_row - sub + add
    # Create a copy of the original tensor and update the last row
    result = attn_weights.clone()
    result[:, :, -1, :] = new_last_row

    return result


def raise_entropy(attn_weights: torch.Tensor) -> torch.Tensor:

    attn_matrix = attn_weights[0, 0].detach().cpu().numpy()
    print_matrix(attn_matrix)
    print()
    return attn_weights


class GemmaBlockedAttention(Gemma2Attention):
    def __init__(
        self,
        config,
        layer_idx,
        disable_idx: Optional[Set[int]] = {},
        disable_positional_encoding: Optional[bool] = False,
    ):
        super().__init__(config, layer_idx)
        self.disable_idx = set(disable_idx)
        self.disable_positional_encoding = disable_positional_encoding
        self.save_model_dimensions = True if layer_idx == 0 else False
        self.model_dimensions = None

    def disable_head(self, head_idx: int):
        self.disable_idx.add(head_idx)

    def enable_head(self, head_idx: int):
        if head_idx in self.disable_idx:
            self.disable_idx.remove(head_idx)

    def get_model_dimensions(self):
        return self.model_dimensions

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
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if (
                attention_mask is not None
                and self.config._attn_implementation == "flash_attention_2"
            ):
                seq_len = attention_mask.shape[-1]
                key_states, value_states = (
                    key_states[:, :, :seq_len, :],
                    value_states[:, :, :seq_len, :],
                )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        if self.disable_idx:
            disable_idx = np.array(list(self.disable_idx))
            attn_output[:, :, disable_idx, :] = 0

        if self.save_model_dimensions:
            self.model_dimensions = attn_output.shape[-2]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaBlockedAttention(LlamaAttention):
    def __init__(
        self,
        config,
        layer_idx,
        disable_idx: Optional[Set[int]] = {},
        raise_entropy_idx: Optional[Set[int]] = {},
        lower_entropy_idx: Optional[Set[int]] = {},
        disable_positional_encoding: Optional[bool] = False,
    ):
        super().__init__(config, layer_idx)
        self.disable_idx = disable_idx
        self.raise_entropy_idx = raise_entropy_idx
        self.lower_entropy_idx = lower_entropy_idx
        self.disable_positional_encoding = disable_positional_encoding
        self.save_model_dimensions = True if layer_idx == 0 else False
        self.model_dimensions = None

    def disable_head(self, head_idx: int):
        self.disable_idx.add(head_idx)

    def enable_head(self, head_idx: int):
        if head_idx in self.disable_idx:
            self.disable_idx.remove(head_idx)

    def get_model_dimensions(self):
        return self.model_dimensions

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

        if not self.disable_positional_encoding:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        # if attention_mask is not None:
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_weights = increase_entropy_batch(attn_weights)
        # attn_output = torch.matmul(attn_weights, value_states)
        # attn_output = attn_output.transpose(1, 2).contiguous()
        attention_interface: Callable = eager_attention_forward
        # print(f"self.config._attn_implementation: {self.config._attn_implementation}")
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

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
            disable_idx = np.array(list(self.disable_idx))
            attn_output[:, :, disable_idx, :] = 0
        if self.save_model_dimensions:
            self.model_dimensions = attn_output.shape[-2]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


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
        blocked_attn.load_state_dict(layer.self_attn.state_dict())
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
