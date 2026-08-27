"""qwen4_exp blocked-prefill wiring: eligibility + step-path parity."""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Metal
    pytest.skip("Metal required", allow_module_level=True)

from mtplx.kernels.gdn_blocked_prefill import (  # noqa: E402
    blocked_prefill_eligible,
    blocked_prefill_ineligibility_reason,
    gated_delta_blocked_prefill,
)
import mlx_lm.models.gated_delta as gd  # noqa: E402


def _max_abs(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_tiny_smoke_geometry_is_ineligible():
    q = mx.zeros((1, 8, 2, 16), dtype=mx.float32)
    v = mx.zeros((1, 8, 4, 16), dtype=mx.float32)
    g = mx.ones((1, 8, 4), dtype=mx.float32)
    reason = blocked_prefill_ineligibility_reason(q, v, g, None, None)
    assert reason is not None
    assert "Dk=16" in reason
    assert not blocked_prefill_eligible(q, v, g, None, None)


def test_real_qwen4_exp_geometry_is_eligible():
    # config.json of /Users/mac/models/Qwen3.8-Flash-Next-MLX-4bit
    q = mx.zeros((1, 32, 16, 128), dtype=mx.bfloat16)
    v = mx.zeros((1, 32, 48, 128), dtype=mx.bfloat16)
    g = mx.ones((1, 32, 48), dtype=mx.float32)
    state = mx.zeros((1, 48, 128, 128), dtype=mx.float32)
    assert blocked_prefill_ineligibility_reason(q, v, g, None, state) is None
    assert blocked_prefill_eligible(q, v, g, None, state)


def test_qwen4_exp_shapes_blocked_matches_stock_kernel():
    B, T, Hk, Hv, Dk, Dv = 1, 33, 16, 48, 128, 128
    mx.random.seed(11)
    q = (mx.random.normal((B, T, Hk, Dk)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((B, T, Hk, Dk)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((B, T, Hv, Dv)) * 0.5).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.float32) * 0.98
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.float32)
    state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.1).astype(mx.float32)
    y_ref, s_ref = gd.gated_delta_kernel(q, k, v, g, beta, state)
    y_new, s_new = gated_delta_blocked_prefill(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_new, s_new)
    assert _max_abs(y_new, y_ref) <= 0.05
    assert _max_abs(s_new, s_ref) <= 0.05


def _eligible_tiny_config() -> dict:
    hidden = 64
    return {
        "model_type": "qwen4_exp",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": hidden,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 128,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "output_gate_type": "sigmoid",
            "hc_count": 2,
            "hc_lowrank": 16,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 16,
            "indexer_budget": 32,
            "indexer_compress_ratio": 4,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 128,
            "make_ngram_vocab_size_divisible_by": 8,
            "split_ngram_parts": 2,
            "ple_embed_dim": hidden,
            "ple_layer_ids": [],
            "ple_conv_kernel_size": 4,
            "eos_token_id": 0,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
            "mtp": {
                "hybrid": True,
                "layer_types": ["full_attention"],
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1,
                "rope_theta": 10000,
            },
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }


def _build_model(config: dict):
    from mtplx.models.qwen4_exp import Model, ModelArgs

    args = ModelArgs.from_dict(
        {"model_type": "qwen4_exp", "text_config": config["text_config"]}
    )
    mx.random.seed(0)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _gdn_states(cache):
    states = []
    for entry in cache:
        if entry is None or hasattr(entry, "keys"):
            continue
        try:
            conv = entry[0]
            rec = entry[1]
        except Exception:
            continue
        if rec is not None:
            states.append((conv, rec))
    return states


def _forward_blocked(model, prompt):
    cache = model.make_cache()
    logits, hidden = model(prompt, cache=cache, return_hidden=True)
    mx.eval(logits, hidden)
    return logits, hidden, cache


def _forward_step(model, prompt):
    cache = model.make_cache()
    logit_steps = []
    hidden_steps = []
    for t in range(int(prompt.shape[1])):
        logits, hidden = model(
            prompt[:, t : t + 1], cache=cache, return_hidden=True
        )
        logit_steps.append(logits)
        hidden_steps.append(hidden)
    logits = mx.concatenate(logit_steps, axis=1)
    hidden = mx.concatenate(hidden_steps, axis=1)
    mx.eval(logits, hidden)
    return logits, hidden, cache


def _compare_pair(label: str, a, b, atol: float) -> tuple[bool, float]:
    max_abs = _max_abs(a, b)
    ok = bool(mx.allclose(a, b, atol=atol, rtol=1e-3).item())
    return ok, max_abs


def test_eligible_tiny_model_blocked_matches_step_path():
    model = _build_model(_eligible_tiny_config())
    prompt = mx.array([list(range(1, 17))], dtype=mx.int32)
    os.environ.pop("MTPLX_GDN_BLOCKED_PREFILL_FORCE_STOCK", None)
    _b_logits, b_hidden, b_cache = _forward_blocked(model, prompt)
    os.environ["MTPLX_GDN_BLOCKED_PREFILL_FORCE_STOCK"] = "1"
    try:
        _s_logits, s_hidden, s_cache = _forward_step(model, prompt)
    finally:
        os.environ.pop("MTPLX_GDN_BLOCKED_PREFILL_FORCE_STOCK", None)

    atol = 5e-3
    ok_h, _ = _compare_pair("HIDDEN", b_hidden, s_hidden, atol)
    assert ok_h
    for (b_conv, b_rec), (s_conv, s_rec) in zip(
        _gdn_states(b_cache), _gdn_states(s_cache)
    ):
        ok_c, _ = _compare_pair("CONV", b_conv, s_conv, atol)
        ok_r, _ = _compare_pair("STATE", b_rec, s_rec, atol)
        assert ok_c and ok_r
