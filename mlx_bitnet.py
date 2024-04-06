# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

import argparse
import math
import warnings
import glob
import json
import time
import os
from time import perf_counter_ns
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from configuration_bitnet import BitnetConfig
from mlx.utils import tree_map, tree_unflatten
from sentencepiece import SentencePieceProcessor
from tokenization_bitnet import BitnetTokenizer



from dataclasses import dataclass

@dataclass
class MinimalBitnetConfig:
    # _name_or_path: str = "1bitLLM/bitnet_b1_58-xl"
    # architectures: list = ("BitnetForCausalLM",)
    attention_bias: bool = False
    # attention_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    # hidden_act: str = "silu"
    hidden_size: int = 2048
    # initializer_range: float = 0.02
    input_bits: int = 8
    intermediate_size: int = 5460
    max_position_embeddings: int = 2048
    # model_type: str = "llama"
    num_attention_heads: int = 32
    num_hidden_layers: int = 24
    num_key_value_heads: int = 32
    pad_token_id: int = 32000
    # pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    # rope_scaling: None = None
    rope_theta: float = 10000.0
    # tie_word_embeddings: bool = True
    # torch_dtype: str = "float16"
    # transformers_version: str = "4.39.0"
    use_cache: bool = True
    vocab_size: int = 32002
    weight_bits: int = 1
    # attn_implementation: str = "eager"
    output_hidden_states: bool = False
    output_attentions: bool = False
    use_return_dict: bool = True


def sanitize_config(_config: BitnetConfig) -> MinimalBitnetConfig:
    return MinimalBitnetConfig(
        attention_bias=_config.attention_bias,
        hidden_size=_config.hidden_size,
        input_bits=_config.input_bits,
        intermediate_size=_config.intermediate_size,
        max_position_embeddings=_config.max_position_embeddings,
        num_attention_heads=_config.num_attention_heads,
        num_key_value_heads=_config.num_key_value_heads,
        pad_token_id=_config.pad_token_id,
        rms_norm_eps=_config.rms_norm_eps,
        rope_theta=_config.rope_theta,
        weight_bits=_config.weight_bits,
        use_cache=_config.use_cache,
        output_hidden_states=_config.output_hidden_states,
        output_attentions=_config.output_attentions,
        use_return_dict=_config.use_return_dict,
    )

def clamp(arr, min=None, max=None):
    if not min:
        return mx.minimum(arr, max)
    if not max:
        return mx.maximum(arr, min)

    return mx.minimum(mx.maximum(arr, min), max)

def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.astype(mx.float32)
    s =  1 / clamp(weight.abs().mean(), min=1e-5)
    result = clamp((weight * s).round(), min=-1, max=1) / s
    return result.astype(dtype)


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.astype(mx.float32)
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / mx.maximum(x.abs().max(axis=-1, keepdims=True), 1e-5)
    # print("Qn", Qn)
    # print("Qp", Qp)
    # print("s", s)
    # print("x*s", x*s)
    result = clamp((x * s).round(), min=Qn, max=Qp) / s
    return result.astype(dtype)


class BitLinear(nn.Linear):

    def __init__(self,
            *kargs,
            weight_bits=1,
            input_bits=8,
            **kwargs
        ):
        super(BitLinear, self).__init__(*kargs, **kwargs)
        """
        RMSNorm is placed outside BitLinear
        """
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, input):
        # print("[mlx] input", input)
        quant_input = activation_quant(input, self.input_bits)
        # print("[mlx] quant_input", quant_input)
        quant_weight = weight_quant(self.weight, self.weight_bits)
        # print("[mlx] quant_weight", quant_weight)

        out = quant_input @ quant_weight.T
        # print("[mlx] out", out)
        if hasattr(self, 'bias') and self.bias is not None:
            out += mx.broadcast_to(self.bias.reshape((1, -1)), out.shape)
        # print("[mlx] out after bias", out)

        return out


class BitnetRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BitnetRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(mx.float32)
        variance = hidden_states.square().mean(axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)
        return self.weight * hidden_states



class BitnetRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.int64).astype(mx.float32) / self.dim))
        self.inv_freq = inv_freq
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = mx.arange(self.max_seq_len_cached, dtype=mx.int64).astype(self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = mx.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self._cos_cached = emb.cos().astype(mx.float32)
        self._sin_cached = emb.sin().astype(mx.float32)

    @property
    def sin_cached(self):
        print(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `BitnetAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        print(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `BitnetAttention` class"
        )
        return self._cos_cached

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq = self.inv_freq[None, :, None].astype(mx.float32)
        inv_freq_expanded = mx.broadcast_to(inv_freq, (position_ids.shape[0], inv_freq.shape[1], 1))
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.astype(mx.float32) @ position_ids_expanded.astype(mx.float32)).transpose((0, 2, 1))
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`mx.array`): The query tensor.
        k (`mx.array`): The key tensor.
        cos (`mx.array`): The cosine part of the rotary embedding.
        sin (`mx.array`): The sine part of the rotary embedding.
        position_ids (`mx.array`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(mx.array)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BitnetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = BitLinear(
            self.hidden_size, self.intermediate_size, bias=False, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.up_proj = BitLinear(
            self.hidden_size, self.intermediate_size, bias=False, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.down_proj = BitLinear(
            self.intermediate_size, self.hidden_size, bias=False, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.act_fn = nn.silu
        self.ffn_layernorm = BitnetRMSNorm(self.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x):
        x = self.act_fn(self.gate_proj.forward(x)) * self.up_proj.forward(x)
        x = self.ffn_layernorm.forward(x)
        # print("[mlx] mlp x_2", x)
        x = self.down_proj.forward(x)
        # print("[mlx] mlp x_3", x)
        return x


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = mx.expand_dims(hidden_states, axis=2)
    hidden_states = mx.broadcast_to(hidden_states, shape=(batch, num_key_value_heads, n_rep, slen, head_dim))
    hidden_states = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    return hidden_states


class BitnetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MinimalBitnetConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = BitLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.k_proj = BitLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.v_proj = BitLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.o_proj = BitLinear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self._init_rope()
        self.inner_attn_ln = BitnetRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def _init_rope(self):
        self.rotary_emb = BitnetRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        output_attentions: bool = False,
        cache_position: Optional[mx.array] = None,
        **kwargs,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[Tuple[mx.array]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj.forward(hidden_states)
        key_states = self.k_proj.forward(hidden_states)
        value_states = self.v_proj.forward(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb.forward(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_cache, value_cache = past_key_value
            key_states = mx.concatenate((key_cache, key_states), axis=2)
            value_states = mx.concatenate((value_cache, value_states), axis=2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = mx.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.softmax(attn_weights, axis=-1).astype(query_states.dtype)
        attn_output = mx.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of shape {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        # attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.swapaxes(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.inner_attn_ln.forward(attn_output)
        attn_output = self.o_proj.forward(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value





class BitnetDecoderLayer(nn.Module):
    def __init__(self, config: MinimalBitnetConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = BitnetAttention(config=config, layer_idx=layer_idx)

        self.mlp = BitnetMLP(config)
        self.input_layernorm = BitnetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BitnetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[mx.array] = None,
        **kwargs,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Args:
            hidden_states (`mx.array`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mx.array`, *optional*):
                attention mask of size `(batch_size, 1, query_sequence_length, key_sequence_length)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(mx.array)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm.forward(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        # print("[mlx] hidden_states_2", hidden_states)
        hidden_states = residual + hidden_states
        # print("[mlx] hidden_states_3", hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        # print("[mlx] hidden_states_4", hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        # print("[mlx] hidden_states_5", hidden_states)
        hidden_states = residual + hidden_states
        # print("[mlx] hidden_states_6", hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BitnetModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`BitnetDecoderLayer`]

    Args:
        config: BitnetConfig
    """

    def __init__(self, config: BitnetConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            BitnetDecoderLayer(sanitize_config(config), layer_idx) for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = BitnetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: mx.array = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[List[mx.array]] = None,
        inputs_embeds: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mx.array] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        # if use_cache:  # kept for BC (cache positions)
        #     if not isinstance(past_key_values, StaticCache):
        #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        #         past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            # if isinstance(past_key_values, StaticCache):
            #     raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = mx.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )

        if position_ids is None:
            position_ids = mx.expand_dims(cache_position, axis=0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # print("before hidden_states", hidden_states)
            # print("before attention_mask", causal_mask)
            # print("before position_ids", position_ids)
            # print("before past_key_value", past_key_values)
            # print("before output_attentions", output_attentions)
            # print("before use_cache", use_cache)
            # print("before cache_position", cache_position)

            layer_outputs = decoder_layer.forward(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            # print("layer_outputs", layer_outputs)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm.forward(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def generate(self, x, temp=1.0):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embed_tokens.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.embed_tokens(x)
        for l in self.layers:
            x, c = l.forward(x, mask=mask)
            # We store the per layer cache in a simple python list
            cache.append(c)
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.output(x[:, -1])
        y = sample(y)

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.embed_tokens(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = sample(self.output(x[:, -1]))

            yield y


    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        dtype = input_tensor.dtype
        # min_dtype = mx.finfo(dtype).min
        # TODO: not sure how to get min of a dtype in mlx
        min_dtype = 0
        sequence_length = input_tensor.shape[1]
        if hasattr(self.layers[0].self_attn, "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, mx.array) else cache_position[-1] + 1
            )

        causal_mask = mx.full((sequence_length, target_length), min_dtype, dtype=dtype)
        if sequence_length != 1:
            causal_mask = mx.triu(causal_mask, 1)
        causal_mask *= mx.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (input_tensor.shape[0], 1) + causal_mask.shape[-2:])
        if attention_mask is not None:
            # causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.ndim == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = mx.equal(causal_mask[..., :mask_length], 0.0) * mx.equal(attention_mask[:, None, None, :], 0.0)
                causal_mask[..., :mask_length] = mx.where(padding_mask, mx.array(min_dtype, dtype=causal_mask.dtype), causal_mask[..., :mask_length])
            elif attention_mask.ndim == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = mx.equal(attention_mask, 0.0).astype(dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        return causal_mask


class BitnetForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BitnetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: mx.array = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[List[mx.array]] = None,
        inputs_embeds: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mx.array] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
        Returns:

        Example:

        ```python
        >>> from transformers import LlamaTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Bitnet-2-7b-hf")
        >>> tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Bitnet-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(self.model.layers[0].self_attn, "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = mx.where(attention_mask == 0, mx.ones_like(position_ids), position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = mx.arange(past_length, past_length + input_length)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"

def generate(model, tokenizer, args):
    # input("Press enter to start generation")
    # print("------")
    print(args.prompt)
    x = mx.array([[tokenizer.bos_token_id] + tokenizer.encode(args.prompt)])
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(x, args.temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= args.max_tokens:
            break

        elif (len(tokens) % args.write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)

def load_model(model_name: str, dtype: str = "float16"):
    config = BitnetConfig.from_pretrained(model_name)
    dtype = getattr(mx, dtype)
    model = BitnetModel(sanitize_config(config))
    file_name = model_name.replace("/", "-")
    weights = mx.load(f"{file_name}.npz")
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: p.astype(dtype), weights)
    model.update(weights)
    mx.eval(model.parameters())
    return model, BitnetTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5 Inference script")
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the bitnet model.",
        default="1bitLLM/bitnet_b1_58-large",
    )
    parser.add_argument(
        "--prompt",
        help="",
        default="translate English to German: That is good.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dtype",
        help="The model data type.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )

    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model, dtype=args.dtype)

    print("[INFO] Generating with Bitnet...", flush=True)
    print("Input: ", args.prompt, flush=True)

    # generate(model, tokenizer, args)
