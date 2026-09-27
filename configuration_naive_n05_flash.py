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

from transformers.configuration_utils import PretrainedConfig


class NaiveN05FlashConfig(PretrainedConfig):
    model_type = "naive_n05_flash"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 152576,
        hidden_size: int = 4096,
        intermediate_size: int = 16384,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 4,
        head_dim: int = 192,
        v_head_dim: int = 128,
        swa_num_attention_heads: int = 64,
        swa_num_key_value_heads: int = 8,
        swa_head_dim: int = 192,
        swa_v_head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        layernorm_epsilon: float = 1e-5,
        rope_theta: float = 10000000.0,
        swa_rope_theta: float = 10000.0,
        partial_rotary_factor: float = 0.334,
        sliding_window: int = 128,
        hybrid_layer_pattern: list[int] | None = None,
        attention_bias: bool = False,
        attention_projection_layout: str = "split",
        attention_value_scale: float | None = 0.707,
        add_full_attention_sink_bias: bool = False,
        add_swa_attention_sink_bias: bool = True,
        n_routed_experts: int = 256,
        moe_intermediate_size: int = 2048,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float | None = 1.0,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        moe_layer_freq: list[int] | None = None,
        enable_dsa: bool = True,
        index_top_k: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 16,
        index_n_kv_heads: int = 1,
        indexer_activation_dtype: str = "fp8_e4m3",
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = 151643,
        eos_token_id: int | list[int] | None = 151645,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.rope_theta = rope_theta
        self.swa_rope_theta = swa_rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.sliding_window = sliding_window
        self.hybrid_layer_pattern = hybrid_layer_pattern or [
            int(i != 0 and i % 6 != 5) for i in range(num_hidden_layers)
        ]
        self.attention_bias = attention_bias
        self.attention_projection_layout = attention_projection_layout
        self.attention_value_scale = attention_value_scale
        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor or 1.0
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.moe_layer_freq = moe_layer_freq or ([0] + [1] * (num_hidden_layers - 1))
        self.enable_dsa = enable_dsa
        self.index_top_k = index_top_k
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_n_kv_heads = index_n_kv_heads
        self.indexer_activation_dtype = indexer_activation_dtype
        self.use_cache = use_cache
        super().__init__(
            tie_word_embeddings=tie_word_embeddings, pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs
        )
        self.validate()

    def validate(self):
        if self.attention_projection_layout != "split" or self.index_n_kv_heads != 1:
            raise ValueError("NaiveN05Flash requires split Q/K/V and one indexer KV head")
        if not self.enable_dsa or self.index_top_k <= 0:
            raise ValueError("NaiveN05Flash requires DSA with a positive index_top_k")
        if self.indexer_activation_dtype not in ("bf16", "fp8_e4m3"):
            raise ValueError("indexer_activation_dtype must be bf16 or fp8_e4m3")
        if self.scoring_func != "sigmoid" or self.topk_method != "noaux_tc":
            raise ValueError("NaiveN05Flash requires sigmoid routing with correction bias")
        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError("NaiveN05Flash uses one expert group")
        for prefix in ("", "swa_"):
            heads = getattr(self, prefix + "num_attention_heads")
            kv_heads = getattr(self, prefix + "num_key_value_heads")
            head_dim = getattr(self, prefix + "head_dim")
            rotary_dim = int(head_dim * self.partial_rotary_factor)
            if kv_heads <= 0 or heads % kv_heads or rotary_dim % 2 or not 0 < rotary_dim <= head_dim:
                raise ValueError("Invalid attention head or rotary dimensions")
        if int(self.head_dim * self.partial_rotary_factor) > self.index_head_dim:
            raise ValueError("Indexer head dimension must cover the rotary dimension")
        if self._attn_implementation not in (None, "eager"):
            raise ValueError("NaiveN05Flash supports eager attention only")
        if len(self.hybrid_layer_pattern) != self.num_hidden_layers:
            raise ValueError("hybrid_layer_pattern must contain one entry per layer")
        if len(self.moe_layer_freq) != self.num_hidden_layers:
            raise ValueError("moe_layer_freq must contain one entry per layer")
        self.layer_types = [
            "sliding_attention" if is_swa else "deepseek_sparse_attention" for is_swa in self.hybrid_layer_pattern
        ]


__all__ = ["NaiveN05FlashConfig"]
