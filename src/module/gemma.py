from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    eager_attention_forward,
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
)
import torch
from typing import Tuple, Optional, Unpack, Callable, Dict, List, Set
import numpy as np
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging

logger = logging.get_logger(__name__)


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

        if len(self.disable_idx) > 0:
            disable_idx = np.array(list(self.disable_idx))
            attn_output[:, :, disable_idx, :] = 0

        if self.save_model_dimensions:
            self.model_dimensions = attn_output.shape[-2]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
