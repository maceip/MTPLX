"""Static-YaRN contract tests for Qwen4-Exp's million-token extension."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.qwen4_exp import (
    Attention,
    TextArgs,
    _apply_partial_rope,
    _rope_cos_sin,
    _rope_inv_freq_and_scaling,
)


def _args(factor: float) -> TextArgs:
    return TextArgs(
        max_position_embeddings=int(262_144 * factor),
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "factor": factor,
            "original_max_position_embeddings": 262_144,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    )


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_static_yarn_matches_qwen_frequency_ramp_and_mscale(factor: float):
    args = _args(factor)
    inv_freq, attention_scaling = _rope_inv_freq_and_scaling(args)
    mx.eval(inv_freq)

    assert tuple(inv_freq.shape) == (32,)
    assert attention_scaling == pytest.approx(1.0 + 0.1 * math.log(factor))

    base = 10_000_000.0
    default = [1.0 / (base ** ((2 * index) / 64)) for index in range(32)]
    values = inv_freq.tolist()
    # For this exact Qwen geometry, Transformers' truncated correction range
    # is [14,22]. Frequencies before it extrapolate unchanged; frequencies
    # after it interpolate by the full static factor.
    assert values[0] == pytest.approx(default[0], rel=1e-6)
    assert values[14] == pytest.approx(default[14], rel=1e-6)
    assert values[18] == pytest.approx(
        0.5 * default[18] + 0.5 * default[18] / factor,
        rel=1e-6,
    )
    assert values[22] == pytest.approx(default[22] / factor, rel=1e-6)
    assert values[31] == pytest.approx(default[31] / factor, rel=1e-6)


def test_production_factor_four_inv_freq_is_torch_bit_identical():
    """Pin the 32-value buffer emitted by installed Transformers/Torch CPU."""

    inv_freq, _ = _rope_inv_freq_and_scaling(_args(4.0))
    mx.eval(inv_freq)

    expected_float32_bits = [
        1065353216,
        1058714411,
        1052440603,
        1046607925,
        1040747895,
        1034226006,
        1028093503,
        1022431607,
        1016180025,
        1009782866,
        1003801109,
        998282877,
        991652109,
        985388013,
        979567074,
        972596068,
        965129534,
        957351307,
        949443597,
        941325830,
        932889448,
        923989741,
        914435499,
        907812804,
        901558465,
        895749318,
        889835338,
        883330636,
        877218903,
        871582109,
        865272974,
        858894148,
    ]
    actual_bits = np.asarray(inv_freq, dtype=np.float32).view(np.uint32).tolist()
    assert actual_bits == expected_float32_bits


@pytest.mark.parametrize("factor,position", [(2.0, 524_287), (4.0, 1_048_575)])
def test_static_yarn_scales_only_the_rotary_prefix(
    factor: float,
    position: int,
):
    args = _args(factor)
    inv_freq, attention_scaling = _rope_inv_freq_and_scaling(args)
    positions = mx.array([position], dtype=mx.int32)
    cosine, sine = _rope_cos_sin(positions, inv_freq, attention_scaling)
    values = mx.arange(256, dtype=mx.float32).reshape(1, 1, 1, 256) / 257.0
    actual = _apply_partial_rope(values, cosine, sine)
    unscaled_cosine, unscaled_sine = _rope_cos_sin(positions, inv_freq)
    unscaled = _apply_partial_rope(values, unscaled_cosine, unscaled_sine)
    mx.eval(actual, unscaled)

    rotary_dim = args.rotary_dim
    assert bool(
        mx.allclose(
            actual[..., :rotary_dim],
            unscaled[..., :rotary_dim] * attention_scaling,
            rtol=1e-5,
            atol=1e-6,
        ).item()
    )
    assert bool(
        mx.array_equal(
            actual[..., rotary_dim:],
            values[..., rotary_dim:],
        ).item()
    )


def test_unknown_qwen_rope_type_fails_closed():
    args = TextArgs(
        rope_parameters={
            "rope_type": "not-a-real-rope",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        }
    )
    with pytest.raises(ValueError, match="supports rope_type"):
        _rope_inv_freq_and_scaling(args)


def test_attention_indexer_and_compiled_core_share_static_yarn_spec():
    """Every QSA lane must consume the main attention layer's YaRN contract.

    Keep construction on MLX CPU and stop before asking the compiled core for
    an entry: this is an ownership/wiring test, not a Metal or graph-dispatch
    test.  The deliberately tiny projection widths avoid allocating the full
    model while retaining the shipped 256/128 head and 64-dim RoPE geometry.
    """

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        args = TextArgs(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=256,
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2_048,
            indexer_compress_ratio=4,
            max_position_embeddings=1_048_576,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "factor": 4.0,
                "original_max_position_embeddings": 262_144,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
        )
        attention = Attention(args)
        indexer = attention.indexer
        assert indexer is not None
        mx.eval(attention._inv_freq, indexer._inv_freq)

        assert tuple(attention._inv_freq.shape) == (32,)
        assert bool(mx.array_equal(attention._inv_freq, indexer._inv_freq).item())
        assert (
            attention._rope_attention_scaling
            == indexer._rope_attention_scaling
            == 1.0 + 0.1 * math.log(4.0)
        )

        core = indexer._get_compiled_indexer_core()
        assert core._inv_freq is indexer._inv_freq
        assert core._rope_attention_scaling == indexer._rope_attention_scaling
        assert core.to_dict()["entry_count"] == 0
    finally:
        mx.set_default_device(previous_device)
