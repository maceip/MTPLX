import math
import mlx.core as mx
import pytest
from mtplx.models.qwen4_exp import ModelArgs, Model, _build_rope_inv_freq


def test_qwen4_exp_yarn_rope_frequency_scaling():
    """Verify YaRN RoPE scales inv_freq and mscale for long context."""
    dim = 64
    base = 10_000_000.0

    # Default without scaling
    inv_freq_default, mscale_default = _build_rope_inv_freq(dim, base, None)
    assert mscale_default == 1.0
    assert inv_freq_default.shape == (32,)

    # YaRN scaling with factor 4.0
    yarn_params = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262144,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    inv_freq_yarn, mscale_yarn = _build_rope_inv_freq(dim, base, yarn_params)
    assert mscale_yarn > 1.0
    assert inv_freq_yarn.shape == (32,)
    # Frequencies should be modulated compared to default
    assert not mx.allclose(inv_freq_yarn, inv_freq_default)


def test_qwen4_exp_yarn_rope_attention_factor_override():
    """Verify explicit attention_factor or mscale in rope_parameters takes precedence."""
    dim = 64
    base = 10_000_000.0

    yarn_params_custom = {
        "rope_type": "yarn",
        "factor": 4.0,
        "attention_factor": 1.25,
    }
    _, mscale = _build_rope_inv_freq(dim, base, yarn_params_custom)
    assert mscale == 1.25

    yarn_params_mscale = {
        "rope_type": "yarn",
        "factor": 4.0,
        "mscale": 1.35,
    }
    _, mscale_alt = _build_rope_inv_freq(dim, base, yarn_params_mscale)
    assert mscale_alt == 1.35


def test_qwen4_exp_yarn_rope_fallback_max_position():
    """Verify fallback max_position_embeddings is used when omitted from rope_parameters."""
    dim = 64
    base = 10_000_000.0

    yarn_params = {
        "rope_type": "yarn",
        "factor": 4.0,
    }
    inv_freq_32k, _ = _build_rope_inv_freq(
        dim, base, yarn_params, max_position_embeddings=32768
    )
    inv_freq_256k, _ = _build_rope_inv_freq(
        dim, base, yarn_params, max_position_embeddings=262144
    )
    assert not mx.allclose(inv_freq_32k, inv_freq_256k)


def test_qwen4_exp_model_with_yarn_config():
    """Verify Qwen4Exp model initializes Attention and QSAIndexer with YaRN RoPE."""
    args = ModelArgs(
        model_type="qwen4_exp",
        text_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 1000,
            "max_position_embeddings": 131072,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "attention_factor": 1.18,
            },
            "layer_types": ["linear_attention", "full_attention"],
        },
    )
    model = Model(args)
    layer = model.layers[1]
    assert hasattr(layer, "self_attn")
    assert layer.self_attn._mscale == 1.18
    assert layer.self_attn.indexer._mscale == 1.18
