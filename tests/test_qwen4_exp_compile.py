import mlx.core as mx
import pytest
from mtplx.models.qwen4_exp import ModelArgs, Model


def test_qwen4_exp_compile_flag_support(monkeypatch):
    """Verify that MTPLX_QWEN4EXP_COMPILE flag enables the compiled GDN decode path."""
    monkeypatch.setenv("MTPLX_QWEN4EXP_COMPILE", "1")
    args = ModelArgs(
        model_type="qwen4_exp",
        text_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 1000,
            "layer_types": ["linear_attention", "linear_attention"],
        },
    )
    model = Model(args)
    text_model = model.language_model.model
    assert text_model._gdn_compiled_env is True
