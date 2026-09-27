# Copyright 2026 Naive AI.
# Copyright 2026 The HuggingFace Inc. team.
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

from copy import copy

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts, DeepseekV3TopkRouter
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP as NaiveN05FlashMLP
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding as NaiveN05FlashRotaryEmbedding
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaRMSNorm as NaiveN05FlashRMSNorm
from transformers.models.llama.modeling_llama import repeat_kv

from .configuration_naive_n05_flash import NaiveN05FlashConfig

# Fuse experts weights
# mlp.experts.{i}.{gate,up,down}_proj.weight[_scale_inv] -> mlp.experts.{gate,up,down}_proj.weight[_scale_inv]
register_checkpoint_conversion_mapping("naive_n05_flash", get_checkpoint_conversion_mapping("deepseek_v3"), overwrite=True)


def round_indexer_fp8(states):
    states = states.float()
    scale = states.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    return (states / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


class NaiveN05FlashMoE(nn.Module):
    """DeepSeek-V3-style routed experts without a shared expert."""

    def __init__(self, config):
        super().__init__()
        self.experts = DeepseekV3Experts(config)
        self.gate = DeepseekV3TopkRouter(config)

    def forward(self, states):
        original_shape = states.shape
        _, topk_weights, topk_indices = self.gate(states)
        states = states.view(-1, states.shape[-1])
        states = self.experts(states, topk_indices, topk_weights)
        return states.view(*original_shape)


class NaiveN05FlashIndexer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.wq = nn.Linear(config.hidden_size, config.index_n_heads * config.index_head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(config.index_head_dim, eps=1e-5)
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)

    def forward(self, states, position_embeddings, past_key_values=None):
        c = self.config
        batch, length, _ = states.shape
        query = self.wq(states).view(batch, length, c.index_n_heads, c.index_head_dim).transpose(1, 2)
        key = self.k_norm(self.wk(states)).unsqueeze(1)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if c.indexer_activation_dtype == "fp8_e4m3":
            query, key = round_indexer_fp8(query), round_indexer_fp8(key)
        if past_key_values is not None:
            key = past_key_values.update_indexer(key.squeeze(1), self.layer_idx).unsqueeze(1)
        scores = (query.float() @ key.float().transpose(-1, -2)).relu()
        weights = self.weights_proj(states) * c.index_n_heads**-0.5
        return (scores * weights.transpose(1, 2).unsqueeze(-1).float()).sum(dim=1)


class NaiveN05FlashAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_swa = bool(config.hybrid_layer_pattern[layer_idx])
        prefix = "swa_" if self.is_swa else ""
        self.heads = getattr(config, prefix + "num_attention_heads")
        self.kv_heads = getattr(config, prefix + "num_key_value_heads")
        self.head_dim = getattr(config, prefix + "head_dim")
        self.v_dim = getattr(config, prefix + "v_head_dim")
        rope_config = copy(config)
        rope_config.head_dim = self.head_dim
        rope_config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(config, prefix + "rope_theta"),
            "partial_rotary_factor": config.partial_rotary_factor,
        }
        self.rotary_emb = NaiveN05FlashRotaryEmbedding(rope_config)
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.v_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.v_dim, config.hidden_size, bias=False)
        self.indexer = None if self.is_swa else NaiveN05FlashIndexer(config, layer_idx)
        sink = config.add_swa_attention_sink_bias if self.is_swa else config.add_full_attention_sink_bias
        self.attention_sink_bias = nn.Parameter(torch.zeros(self.heads), requires_grad=False) if sink else None

    def forward(self, states, positions, allowed, past_key_values=None):
        batch, length, _ = states.shape
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length(self.layer_idx)
        key_offset = 0 if past_key_values is None else past_key_values.get_mask_sizes(length, self.layer_idx)[1]
        query = self.q_proj(states).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(states).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(states).view(batch, length, self.kv_heads, self.v_dim).transpose(1, 2)
        position_embeddings = self.rotary_emb(states, positions)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        key_length = key.shape[-2]
        allowed = allowed[..., key_offset : key_offset + key_length]
        if self.is_swa:
            query_offsets = past_length + torch.arange(length, device=states.device)
            key_offsets = key_offset + torch.arange(key_length, device=states.device)
            allowed = allowed & (query_offsets[:, None] - key_offsets[None, :] < self.config.sliding_window)
        else:
            scores = self.indexer(states, position_embeddings, past_key_values).masked_fill(~allowed, -torch.inf)
            # Stable ties keep padding from changing which equal-score keys win.
            selected = scores.argsort(dim=-1, descending=True, stable=True)[..., : self.config.index_top_k]
            allowed = allowed & torch.zeros_like(scores, dtype=torch.bool).scatter(-1, selected, True)
        key = repeat_kv(key, self.heads // self.kv_heads)
        value = repeat_kv(value, self.heads // self.kv_heads)
        if self.config.attention_value_scale is not None:
            value = value * self.config.attention_value_scale
        logits = (query @ key.transpose(-1, -2)) * self.head_dim**-0.5
        logits = logits.masked_fill(~allowed[:, None], -torch.inf)
        if self.attention_sink_bias is not None:
            sink = self.attention_sink_bias[None, :, None, None].expand(batch, -1, length, -1)
            logits = torch.cat((logits, sink), dim=-1)
        probabilities = logits.float().softmax(-1).nan_to_num(0.0)[..., :key_length].to(value.dtype)
        output = (probabilities @ value).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class NaiveN05FlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = NaiveN05FlashAttention(config, layer_idx)
        self.mlp = NaiveN05FlashMoE(config) if config.moe_layer_freq[layer_idx] else NaiveN05FlashMLP(config)
        self.input_layernorm = NaiveN05FlashRMSNorm(config.hidden_size, config.layernorm_epsilon)
        self.post_attention_layernorm = NaiveN05FlashRMSNorm(config.hidden_size, config.layernorm_epsilon)

    def forward(self, states, positions, allowed, past_key_values=None):
        states = states + self.self_attn(self.input_layernorm(states), positions, allowed, past_key_values)
        return states + self.mlp(self.post_attention_layernorm(states))


class NaiveN05FlashModel(PreTrainedModel):
    config_class = NaiveN05FlashConfig
    base_model_prefix = "model"
    _can_set_experts_implementation_cached_value = True

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NaiveN05FlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = NaiveN05FlashRMSNorm(config.hidden_size, config.layernorm_epsilon)

    def forward(self, states, positions, allowed, past_key_values=None):
        for layer in self.layers:
            states = layer(states, positions, allowed, past_key_values)
        return self.norm(states)


class NaiveN05FlashForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NaiveN05FlashConfig
    base_model_prefix = "model"
    _no_split_modules = ["NaiveN05FlashDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules_strict = ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
    _can_set_experts_implementation_cached_value = True

    def __init__(self, config):
        config.validate()
        super().__init__(config)
        self.model = NaiveN05FlashModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.eval().requires_grad_(False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    @torch.inference_mode()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not use_cache:
            raise ValueError("Pass use_cache=True when providing past_key_values")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("NaiveN05Flash supports DynamicCache only")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        states = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = states.shape
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        total_length = past_length + length
        if attention_mask is None:
            attention_mask = torch.ones(batch, total_length, device=states.device, dtype=torch.bool)
        if attention_mask.shape != (batch, total_length):
            raise ValueError("Expected a 2-D attention_mask covering the cached prefix and new tokens")
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -length:]
        query_offsets = past_length + torch.arange(length, device=states.device)
        key_offsets = torch.arange(total_length, device=states.device)
        allowed = query_offsets[:, None] >= key_offsets[None, :]
        allowed = allowed[None] & attention_mask[:, None].bool()
        logits = self.lm_head(self.model(states, position_ids, allowed, past_key_values))
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


__all__ = ["NaiveN05FlashForCausalLM", "NaiveN05FlashModel"]
