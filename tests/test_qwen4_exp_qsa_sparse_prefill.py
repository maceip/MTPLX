"""QSA prefill must gather a budget of tokens, not materialize [S, T] masks."""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Metal
    pytest.skip("Metal required", allow_module_level=True)

from mtplx.models.qwen4_exp import (  # noqa: E402
    Attention,
    QSASelection,
    TextArgs,
    _AttnCache,
    _indexer_score_chunk,
    selection_to_dense_add_mask,
)


def _attn_args(**overrides) -> TextArgs:
    cfg = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=32,
        indexer_compress_ratio=4,
        partial_rotary_factor=0.25,
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 10_000.0, "partial_rotary_factor": 0.25},
    )
    cfg.update(overrides)
    return TextArgs.from_dict(cfg)


def _max_abs(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_legacy_dense_mask_at_131k_is_68gib():
    """The bug: a [T, T] fp16 additive mask at 131072 is ~64 GiB of values.

    131072^2 * 2 bytes = 32 GiB for the bool keep plus 32 GiB for the fp16
    additive mask — the report's 68 GiB counts the fp32-score-sized neighbour.
    A single fp16 [T,T] array alone is 32 GiB; fp32 scores would be 64 GiB.
    """
    t = 131072
    fp16_bytes = t * t * 2
    fp32_bytes = t * t * 4
    assert fp16_bytes == 34_359_738_368  # 32 GiB
    assert fp32_bytes == 68_719_476_736  # 64 GiB
    # Product QSA keep is K=2048+r per query, not T:
    k = 2048 + 4
    gather_idx_bytes = t * k * 4  # int32 indices
    assert gather_idx_bytes < 2 * 1024**3


def test_indexer_score_chunk_caps_s_by_n_blocks():
    # 131K tokens / r=4 -> 32768 blocks. 128 MiB / (32768*4) = 1024 rows.
    assert _indexer_score_chunk(32768, 2048) == 1024
    assert _indexer_score_chunk(16, 2048) == 2048


def test_selection_is_budget_indices_not_st_mask():
    args = _attn_args()
    attn = Attention(args)
    mx.eval(attn.parameters())
    t = 128
    x = mx.random.normal((1, t, args.hidden_size)).astype(mx.float16)
    cache = _AttnCache()
    sel = attn.indexer(x, attn.rope, cache.indexer, 0)
    assert isinstance(sel, QSASelection)
    assert sel.token_idx.shape[:2] == (1, t)
    k = int(sel.token_idx.shape[-1])
    assert k == args.indexer_budget + args.indexer_compress_ratio
    assert k < t
    mx.eval(sel.token_idx, sel.valid)
    n_valid = int(mx.sum(sel.valid.astype(mx.int32)).item())
    assert n_valid <= t * k
    # Per-query kept tokens cannot exceed budget + r (complete blocks + tail).
    per_row = mx.sum(sel.valid.astype(mx.int32), axis=-1)
    assert int(mx.max(per_row).item()) <= k


def test_gather_matches_dense_mask_path(monkeypatch):
    args = _attn_args()
    attn = Attention(args)
    mx.eval(attn.parameters())
    t = 96
    mx.random.seed(7)
    x = (mx.random.normal((1, t, args.hidden_size)) * 0.2).astype(mx.float16)

    monkeypatch.delenv("MTPLX_QSA_DENSE_MASK", raising=False)
    cache_a = _AttnCache()
    out_a = attn(x, mask="causal", cache=cache_a)
    mx.eval(out_a)

    monkeypatch.setenv("MTPLX_QSA_DENSE_MASK", "1")
    cache_b = _AttnCache()
    out_b = attn(x, mask="causal", cache=cache_b)
    mx.eval(out_b)

    assert out_a.shape == out_b.shape == (1, t, args.hidden_size)
    assert _max_abs(out_a, out_b) < 2e-2


def test_dense_add_mask_shape_is_the_bug_surface():
    t, k = 48, 36
    idx = mx.broadcast_to(mx.arange(k)[None, None, :], (1, t, k)).astype(mx.int32)
    valid = mx.ones((1, t, k), dtype=mx.bool_)
    sel = QSASelection(token_idx=idx, valid=valid)
    add = selection_to_dense_add_mask(sel, kv_len=t, dtype=mx.float16)
    mx.eval(add)
    assert tuple(add.shape) == (1, 1, t, t)
