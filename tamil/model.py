"""Qwen3-style dense decoder (owner choice 2026-11-02): QK-norm keeps fp16 training stable on T4."""
from typing import Literal

import torch
from transformers import AttentionInterface, Qwen3Config, Qwen3ForCausalLM
from transformers.integrations.sdpa_attention import repeat_kv

# token ids follow tamil.tokenizer_train.SPECIAL_TOKENS order
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

SIZES = {
    "350m": dict(hidden_size=1024, intermediate_size=3072, num_hidden_layers=20,
                 num_attention_heads=16, num_key_value_heads=8, head_dim=128),
    "30m": dict(hidden_size=384, intermediate_size=1152, num_hidden_layers=6,
                num_attention_heads=6, num_key_value_heads=3, head_dim=64),  # ablation fallback
    "tiny": dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=16),  # CPU tests
}


def _sdpa_expand_kv(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """SDPA with K/V heads expanded up front. HF's "sdpa" passes enable_gqa=True when there is no mask
    (all of pretraining); per the T4 speed test that path ran the math kernel, which stores a full
    (seq x seq) score grid per head and ran out of memory. Without the flag the memory-efficient kernel can run."""
    n = module.num_key_value_groups
    key, value = repeat_kv(key, n), repeat_kv(value, n)
    is_causal = query.shape[2] > 1 and attention_mask is None
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                                           dropout_p=dropout, scale=scaling, is_causal=is_causal)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa_expand_kv", _sdpa_expand_kv)


def make_config(vocab_size: int, size: str = "350m") -> Qwen3Config:
    return Qwen3Config(vocab_size=vocab_size, max_position_embeddings=4096, rope_theta=1_000_000.0,
                       tie_word_embeddings=True, pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                       **SIZES[size])


def make_model(vocab_size: int, size: Literal["350m", "30m", "tiny"] = "350m") -> Qwen3ForCausalLM:
    cfg = make_config(vocab_size, size)
    cfg._attn_implementation = "sdpa_expand_kv"
    return Qwen3ForCausalLM(cfg)
