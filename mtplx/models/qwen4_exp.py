# Copyright © 2024 Apple Inc. / MTPLX authors
# Native MLX port of Qwen3.8-Flash-Next (architecture: qwen4_exp / qwen4_exp_text).
# Day-0 support for:
#   - GatedDeltaNet (hybrid linear-attention with split QKV/Z/A/B projections & 1D conv)
#   - QSA (Query-Selected Attention / Sparse Attention with QSAIndexer)
#   - GatedResidual (Hyper-Connections with low-rank MLP mixing & injection weights)
#   - PLELayer (Per-Layer Embedding: sharded n-gram hash table with short conv)
#   - Fine-grained MoE (SwitchGLU routed experts + gated shared expert)

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, _BaseCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_update
from mlx_lm.models.switch_layers import (
    SwitchGLU,
    SwitchLinear,
    SwiGLU,
    _gather_sort,
    _scatter_unsort,
)

from ..kernels.gdn_blocked_prefill import (
    blocked_prefill_eligible,
    blocked_prefill_ineligibility_reason,
    gated_delta_blocked_prefill,
)


def _compile_enabled() -> bool:
    return (
        str(os.environ.get("MTPLX_QWEN4EXP_COMPILE", "")).strip().lower()
        in {"1", "true", "yes", "on"}
        or str(os.environ.get("MTPLX_COMPILE_AR_FORWARD", "")).strip().lower()
        in {"1", "true", "yes", "on"}
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    full_attention_interval: int = 4
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    # Gated DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # Hyper-Connections (residual)
    hc_count: int = 4
    hc_lowrank: int = 320
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # N-gram / PLE
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_embed_dim: int = 2560
    ple_layer_ids: list = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    seed: int = 0
    eos_token_id: Any = 248044
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(default_factory=dict)
    rope_scaling: dict = field(default_factory=dict)
    rope_theta: float = 10_000_000.0
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144
    # Multi-Token Prediction (MTP) metadata
    mtp_num_hidden_layers: int = 0
    mtp: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict):
        params = dict(params)
        rope_scaling = params.get("rope_scaling")
        rope_params = params.get("rope_parameters")
        if isinstance(rope_scaling, dict) and not rope_params:
            rp = dict(rope_scaling)
            if "type" in rp and "rope_type" not in rp:
                rp["rope_type"] = rp["type"]
            params["rope_parameters"] = rp
        elif isinstance(rope_params, dict) and isinstance(rope_scaling, dict):
            merged = dict(rope_scaling)
            merged.update(rope_params)
            if "type" in merged and "rope_type" not in merged:
                merged["rope_type"] = merged["type"]
            params["rope_parameters"] = merged
        elif isinstance(rope_params, dict):
            rp = dict(rope_params)
            if "type" in rp and "rope_type" not in rp:
                rp["rope_type"] = rp["type"]
            params["rope_parameters"] = rp
        if "norm_topk_prob" in params:
            params["norm_topk_prob"] = bool(params["norm_topk_prob"])
        for k in (
            "routed_scaling_factor",
            "moe_routed_scaling_factor",
            "router_scaling_factor",
        ):
            if k in params:
                params["routed_scaling_factor"] = float(params[k])
                break
        return super().from_dict(params)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None
    language_model_only: bool = False

    @classmethod
    def from_dict(cls, params: dict):
        tcfg = params.get("text_config")
        if not tcfg or not isinstance(tcfg, dict):
            fields = {"model_type", "vision_config", "quantization", "language_model_only"}
            args = {k: v for k, v in params.items() if k in fields}
            args["text_config"] = dict(params)
            return cls(**args)
        return super().from_dict(params)

    def __post_init__(self):
        cfg = dict(self.text_config) if self.text_config else {}
        if not cfg:
            # Fallback if flat dictionary was passed
            cfg = {
                k: getattr(self, k)
                for k in dir(self)
                if not k.startswith("_")
                and k not in ("text_config", "vision_config", "text", "from_dict")
            }
        self.text = TextArgs.from_dict(cfg)
        rp = self.text.rope_parameters or {}
        self.text.rope_theta = float(rp.get("rope_theta", self.text.rope_theta))
        self.text.partial_rotary_factor = float(
            rp.get("partial_rotary_factor", self.text.partial_rotary_factor)
        )
        if not self.text.layer_types:
            n, k = self.text.num_hidden_layers, self.text.full_attention_interval
            self.text.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(n)
            ]


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm, normalized per group when group_size is given.

    Hyper-connections normalize each of the hc_count streams separately, hence the
    reshape: one weight of size hc_count*hidden, but one statistic per stream.
    """

    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"dim {dim} not divisible by group_size {group_size}")

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps).reshape(shape)
        return x * self.weight


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        g = act(gate.astype(mx.float32))
        return (g * out.astype(mx.float32)).astype(x.dtype)


# ---------------------------------------------------------------------------
# RoPE / Rotary embeddings
# ---------------------------------------------------------------------------


def _rope_partial(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rope to the first `rotary_dim` dimensions only."""
    d = cos.shape[-1]
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    xr, xp = x[..., :d], x[..., d:]
    half = d // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


class RotaryEmbedding:
    def __init__(self, dim: int, base: float, rope_parameters: Optional[dict] = None):
        self.dim = dim
        self.mscale = 1.0
        inv_freq = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        rp = rope_parameters or {}
        rope_type = str(rp.get("rope_type") or rp.get("type", "default")).lower()
        if rope_type == "yarn":
            # Canonical yarn (mirrors mlx-lm rope_utils.YarnRoPE): blend
            # interpolated and extrapolated periods across a correction range,
            # and scale rotation amplitude by mscale.
            factor = float(rp.get("factor", 1.0))
            orig = float(
                rp.get("original_max_position_embeddings")
                or rp.get("original_max_position", 262144)
            )
            beta_fast = float(rp.get("beta_fast", 32))
            beta_slow = float(rp.get("beta_slow", 1))

            def _corr_dim(num_rotations: float) -> float:
                return (
                    dim * math.log(orig / (num_rotations * 2 * math.pi))
                ) / (2 * math.log(base))

            low = max(math.floor(_corr_dim(beta_fast)), 0)
            high = min(math.ceil(_corr_dim(beta_slow)), dim - 1)
            if low == high:
                high += 0.001
            ramp = mx.clip(
                (mx.arange(dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1
            )
            freq_mask = 1.0 - ramp
            freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
            freq_inter = factor * freq_extra
            periods = (freq_inter * freq_extra) / (
                freq_inter * freq_mask + freq_extra * (1.0 - freq_mask)
            )
            inv_freq = 1.0 / periods
            if factor > 1:
                self.mscale = 0.1 * math.log(factor) + 1.0
        self.inv_freq = inv_freq
        # mx.fast.rope(freqs=) uses theta = pos / freqs, so this is 1/inv_freq.
        # Yarn (or any other) variants that rewrite inv_freq stay correct.
        self.freqs = 1.0 / self.inv_freq

    def __call__(self, positions: mx.array):
        # positions: (B, T) -> cos/sin (B, T, dim); mscale folds the yarn
        # amplitude into the rotation (passthrough dims stay unscaled).
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb) * self.mscale, mx.sin(emb) * self.mscale


def _cache_slot(cache: Any, idx: int) -> Any:
    if cache is None:
        return None
    try:
        if hasattr(cache, "__len__") and len(cache) <= idx:
            return None
        return cache[idx]
    except (IndexError, TypeError, KeyError):
        return None


def _set_cache_slot(cache: Any, idx: int, value: Any) -> None:
    if cache is None:
        return
    try:
        cache[idx] = value
    except (IndexError, TypeError, KeyError):
        pass


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    inv_norm = mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * inv_norm


def _blocked_prefill_force_stock() -> bool:
    return str(os.environ.get("MTPLX_GDN_BLOCKED_PREFILL_FORCE_STOCK", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_BLOCKED_PREFILL_LOGGED = False
_BLOCKED_PREFILL_FALLBACK_LOGGED = False


def _log_blocked_prefill_once(q: mx.array) -> None:
    global _BLOCKED_PREFILL_LOGGED
    if _BLOCKED_PREFILL_LOGGED:
        return
    _BLOCKED_PREFILL_LOGGED = True
    try:
        print(
            "[qwen4_exp] gdn-blocked-prefill routed "
            f"T={q.shape[1]} q={tuple(q.shape)} dtype={q.dtype}",
            flush=True,
        )
    except Exception:
        pass


def _log_blocked_prefill_fallback_once(reason: str) -> None:
    global _BLOCKED_PREFILL_FALLBACK_LOGGED
    if _BLOCKED_PREFILL_FALLBACK_LOGGED:
        return
    _BLOCKED_PREFILL_FALLBACK_LOGGED = True
    try:
        print(
            "[qwen4_exp] gdn-blocked-prefill fallback to step path: "
            f"{reason}",
            flush=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# QSA (Query-Selected Attention)
# ---------------------------------------------------------------------------
#
# Spec (tech report pp. 6-9): token budget K=2048, compression r=4, top-512
# complete micro-blocks per query, plus the per-query incomplete tail. Core
# attention must run on that gathered set — never on a dense [S, kv_len]
# score/mask. The mlx-lm PR #1788 port (and this file, previously) converted
# the selection into a boolean keep of shape (B, 1, S, kv_len) and fed it to
# dense SDPA: at S=kv_len=131072 that is a 68 GiB fp16 [T,T] tensor, and even
# under 2048-token generation chunking each chunk still attended the FULL
# growing KV (O(T^2 / chunk) total). Gather + query-chunked indexer scoring
# is the product path; MTPLX_QSA_DENSE_MASK=1 restores the old mask path for
# A/B micro-benches.


def _qsa_dense_mask_enabled() -> bool:
    return str(os.environ.get("MTPLX_QSA_DENSE_MASK", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _qsa_query_chunk() -> int:
    raw = os.environ.get("MTPLX_QSA_QUERY_CHUNK", "2048").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2048


def _qsa_neg_inf(dtype) -> mx.array:
    if hasattr(mx, "finfo"):
        try:
            return mx.array(mx.finfo(dtype).min, dtype=dtype)
        except Exception:
            pass
    return mx.array(-1e4 if dtype in (mx.float16, mx.bfloat16) else -1e9, dtype=dtype)


def _indexer_score_chunk(n_blocks: int, default: int) -> int:
    """Keep the (S, n_blocks) score tile under ~128 MiB float32."""
    cap = max(1, (128 * 1024 * 1024) // max(int(n_blocks) * 4, 1))
    return max(1, min(int(default), cap))


@dataclass
class QSASelection:
    """Per-query token indices for sparse core attention. Never a [S, T] mask."""

    token_idx: mx.array  # (B, S, K) int32
    valid: mx.array  # (B, S, K) bool


@dataclass
class QSAPrep:
    """Indexer state after key pooling; queries are scored in chunks."""

    q: mx.array  # (B, S, H_idx, D_idx)
    pooled: mx.array  # (B, n_blocks, D_idx)
    q_pos: mx.array  # (S,)
    kv_len: int
    left_padding: Optional[mx.array] = None


_QSA_PATH_LOGGED = False


def _log_qsa_path_once(msg: str) -> None:
    global _QSA_PATH_LOGGED
    if _QSA_PATH_LOGGED:
        return
    _QSA_PATH_LOGGED = True
    try:
        print(f"[qwen4_exp] {msg}", flush=True)
    except Exception:
        pass


def _gather_kv_tokens(kv: mx.array, token_idx: mx.array) -> mx.array:
    """Gather kv [B, H, T, D] at token_idx [B, S, K] -> [B, H, S, K, D].

    B == 1 (every serving path here) gathers along axis 2 directly: no
    transpose/reshape of the cache, so history is never copied per step.
    """
    B, H, T, D = kv.shape
    _, S, K = token_idx.shape
    if B == 1:
        gathered = mx.take(kv, token_idx.reshape(-1), axis=2)  # (1, H, S*K, D)
        return gathered.reshape(1, H, S, K, D)
    kv_flat = kv.transpose(0, 2, 1, 3).reshape(B * T, H, D)
    batch_off = (mx.arange(B, dtype=token_idx.dtype) * T).reshape(B, 1, 1)
    gathered = mx.take(kv_flat, (token_idx + batch_off).reshape(-1), axis=0)
    return gathered.reshape(B, S, K, H, D).transpose(0, 3, 1, 2, 4)


def _qsa_gather_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    sel: QSASelection,
    scale: float,
    mask: Optional[Any] = None,
) -> mx.array:
    """SDPA over gathered K=budget tokens per query. q/k/v are [B, H, S, D].

    Keys differ per query, so fused SDPA over a shared KV sequence cannot
    be used. Keep S in-graph and GEMM the last dim (K), never flatten to
    ``B*S`` independent q_len=1 calls. Broadcasts over GQA head groups to
    avoid allocating 12x repeated K/V working sets in memory.
    """
    B, H, S, D = q.shape
    T = int(k.shape[2])
    H_kv = int(k.shape[1])
    safe = mx.clip(mx.where(sel.valid, sel.token_idx, 0), 0, max(T - 1, 0))
    k_sel = _gather_kv_tokens(k, safe)  # (B, H_kv, S, K, D)
    v_sel = _gather_kv_tokens(v, safe)

    mask_gathered = None
    if isinstance(mask, mx.array):
        m = mask
        while m.ndim < 4:
            m = m[None]
        if m.shape[0] != B and m.shape[0] == 1:
            m = mx.broadcast_to(m, (B, m.shape[1], m.shape[2], m.shape[3]))
        if m.shape[2] != S and m.shape[2] == 1:
            m = mx.broadcast_to(m, (m.shape[0], m.shape[1], S, m.shape[3]))
        safe_mask = mx.clip(safe, 0, max(int(m.shape[-1]) - 1, 0))
        idx = safe_mask[:, None, :, :]
        if m.shape[1] > 1:
            idx = mx.broadcast_to(idx, (B, m.shape[1], S, safe_mask.shape[-1]))
        mask_gathered = mx.take_along_axis(m, idx, axis=-1)

    if H != H_kv:
        rep = H // H_kv
        # Reshape q to (B, H_kv, rep, S, 1, D) and broadcast against (B, H_kv, 1, S, D, K)
        # to avoid materializing 12x repeated k_sel/v_sel working sets in memory.
        q_view = q.reshape(B, H_kv, rep, S, 1, D)
        k_view = k_sel.swapaxes(-1, -2).reshape(B, H_kv, 1, S, D, -1)
        scores = (mx.matmul(q_view, k_view) * scale).squeeze(-2)  # (B, H_kv, rep, S, K)
        scores = scores.reshape(B, H, S, -1)  # (B, H, S, K)
        if mask_gathered is not None:
            if mask_gathered.dtype == mx.bool_:
                scores = mx.where(mask_gathered, scores, _qsa_neg_inf(q.dtype))
            else:
                scores = scores + mask_gathered
        scores = mx.where(sel.valid[:, None, :, :], scores, _qsa_neg_inf(q.dtype))
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        probs_view = probs.reshape(B, H_kv, rep, S, 1, -1)  # (B, H_kv, rep, S, 1, K)
        v_view = v_sel.reshape(B, H_kv, 1, S, -1, D)        # (B, H_kv, 1, S, K, D)
        out = mx.matmul(probs_view, v_view).squeeze(-2)     # (B, H_kv, rep, S, D)
        return out.reshape(B, H, S, D)
    # (B, H, S, 1, D) @ (B, H, S, D, K) -> (B, H, S, 1, K)
    scores = mx.matmul(q[..., None, :], k_sel.swapaxes(-1, -2)).squeeze(-2) * scale
    if mask_gathered is not None:
        if mask_gathered.dtype == mx.bool_:
            scores = mx.where(mask_gathered, scores, _qsa_neg_inf(q.dtype))
        else:
            scores = scores + mask_gathered
    scores = mx.where(sel.valid[:, None, :, :], scores, _qsa_neg_inf(q.dtype))
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    # (B, H, S, 1, K) @ (B, H, S, K, D) -> (B, H, S, D)
    return mx.matmul(probs[..., None, :], v_sel).squeeze(-2)


def selection_to_dense_add_mask(sel: QSASelection, kv_len: int, dtype) -> mx.array:
    """Legacy [B, 1, S, kv_len] additive mask — the O(S·T) materialization."""
    B, S, _ = sel.token_idx.shape
    idx = mx.where(sel.valid, sel.token_idx, kv_len)
    keep = mx.zeros((B, S, kv_len + 1), dtype=mx.bool_)
    keep = mx.put_along_axis(keep, idx, mx.array(True), axis=-1)[..., :kv_len]
    return mx.where(
        keep[:, None],
        mx.array(0, dtype=dtype),
        _qsa_neg_inf(dtype),
    )


class QSAIndexer(nn.Module):
    """Select, per query, a budget of compressed key blocks.

    Scoring is O(S · n_blocks) as in the spec (n_blocks = T/r), but the
    (S, n_blocks) tile is produced in query chunks and immediately reduced
    to (S, K) token indices. Nothing of shape [S, T] is allocated.
    """

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def prepare(self, x: mx.array, rope: Any, cache: Any, offset: int) -> Optional[QSAPrep]:
        B, S, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        if cache is not None:
            raw_k = cache.update(raw_k)
        kv_len = int(raw_k.shape[1])

        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // self.compress_ratio
        if n_blocks <= 0:
            return None

        r = self.compress_ratio
        left_padding = getattr(cache, "left_padding", None)
        if left_padding is None and hasattr(cache, "indexer"):
            left_padding = getattr(cache.indexer, "left_padding", None)

        if cache is not None:
            cached_pooled = getattr(cache, "pooled", None)
            n_cached = int(cached_pooled.shape[1]) if cached_pooled is not None else 0
            if n_cached > n_blocks:
                cached_pooled = cached_pooled[:, :n_blocks, :] if n_blocks > 0 else None
                n_cached = n_blocks if n_blocks > 0 else 0

            if n_cached < n_blocks:
                new_raw_k = raw_k[:, n_cached * r : n_blocks * r, :]
                n_new = n_blocks - n_cached
                if left_padding is not None:
                    tok_pos = mx.arange(n_cached * r, n_blocks * r)
                    is_valid = tok_pos[None, :] >= left_padding[:, None]
                    new_raw_k = mx.where(is_valid[..., None], new_raw_k, 0.0)
                    valid_counts = mx.maximum(
                        is_valid.reshape(B, n_new, r).sum(axis=-1, keepdims=True), 1
                    )
                    new_pooled = (
                        new_raw_k.reshape(B, n_new, r, self.head_dim)
                        .astype(mx.float32)
                        .sum(axis=2)
                        / valid_counts
                    )
                else:
                    new_pooled = (
                        new_raw_k.reshape(B, n_new, r, self.head_dim)
                        .astype(mx.float32)
                        .mean(axis=2)
                    )
                new_pooled = self.k_layernorm(new_pooled.astype(raw_k.dtype))
                new_starts = mx.arange(n_cached, n_blocks) * r
                cos_k, sin_k = rope(new_starts[None, :])
                new_pooled = _rope_partial(new_pooled, cos_k, sin_k)
                pooled = (
                    new_pooled
                    if cached_pooled is None or n_cached == 0
                    else mx.concatenate([cached_pooled, new_pooled], axis=1)
                )
                cache.pooled = pooled
            else:
                pooled = cached_pooled
        else:
            pooled_raw = raw_k[:, : n_blocks * r]
            if left_padding is not None:
                tok_pos = mx.arange(n_blocks * r)
                is_valid = tok_pos[None, :] >= left_padding[:, None]
                pooled_raw = mx.where(is_valid[..., None], pooled_raw, 0.0)
                valid_counts = mx.maximum(
                    is_valid.reshape(B, n_blocks, r).sum(axis=-1, keepdims=True), 1
                )
                pooled = (
                    pooled_raw.reshape(B, n_blocks, r, self.head_dim)
                    .astype(mx.float32)
                    .sum(axis=2)
                    / valid_counts
                )
            else:
                pooled = (
                    pooled_raw.reshape(B, n_blocks, r, self.head_dim)
                    .astype(mx.float32)
                    .mean(axis=2)
                )
            pooled = self.k_layernorm(pooled.astype(raw_k.dtype))
            block_starts = mx.arange(n_blocks) * r
            cos_k, sin_k = rope(block_starts[None, :])
            pooled = _rope_partial(pooled, cos_k, sin_k)

        q_pos = mx.arange(offset, offset + S)
        cos_q, sin_q = rope(q_pos[None, :])
        q = self.q_layernorm(q)
        q = _rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])
        return QSAPrep(
            q=q,
            pooled=pooled,
            q_pos=q_pos,
            kv_len=kv_len,
            left_padding=left_padding,
        )

    def select(
        self,
        q: mx.array,
        q_pos: mx.array,
        pooled: mx.array,
        kv_len: int,
        left_padding: Optional[mx.array] = None,
    ) -> QSASelection:
        """Score this query slice against every compressed block; return token idx."""
        B, S, _, _ = q.shape
        n_blocks = int(pooled.shape[1])
        r = self.compress_ratio
        n_complete = (q_pos + 1) // r
        is_complete = mx.arange(n_blocks)[None, :] < n_complete[:, None]  # (S, n_blocks)

        if left_padding is not None:
            block_starts = mx.arange(n_blocks) * r  # (n_blocks,)
            is_not_pad = block_starts[None, None, :] >= left_padding[:, None, None]  # (B, 1, n_blocks)
            valid_block = is_complete[None, :, :] & is_not_pad  # (B, S, n_blocks)
        else:
            valid_block = mx.broadcast_to(is_complete[None, :, :], (B, S, n_blocks))

        # Sum_h ReLU(<q_h, k_b>) without materializing the 4-head score tensor.
        acc = None
        pooled_t = pooled.astype(mx.float32).swapaxes(-1, -2)  # (B, D, n_blocks)
        for h in range(self.n_heads):
            s_h = q[:, :, h].astype(mx.float32) @ pooled_t  # (B, S, n_blocks)
            s_h = mx.maximum(s_h, 0)
            acc = s_h if acc is None else acc + s_h
        scores = acc / math.sqrt(self.head_dim)
        masked = mx.where(valid_block, scores, -mx.inf)

        k_blk = min(self.block_topk, n_blocks)
        top = mx.argpartition(-masked, k_blk - 1, axis=-1)[..., :k_blk]
        is_top_complete = mx.take_along_axis(valid_block, top, axis=-1)
        top = mx.where(is_top_complete, top, 0)

        tok = (top[..., None] * r + mx.arange(r)).reshape(B, S, k_blk * r)
        valid_blk = mx.broadcast_to(
            is_top_complete[..., None], (*is_top_complete.shape, r)
        ).reshape(B, S, k_blk * r) & (tok < kv_len)
        if left_padding is not None:
            valid_blk = valid_blk & (tok >= left_padding[:, None, None])

        tail_start = n_complete * r
        tail = tail_start[:, None] + mx.arange(r)  # (S, r)
        tail_valid = (tail <= q_pos[:, None]) & (tail < kv_len)
        tail = mx.broadcast_to(tail[None], (B, S, r))
        tail_valid = mx.broadcast_to(tail_valid[None], (B, S, r))
        if left_padding is not None:
            tail_valid = tail_valid & (tail >= left_padding[:, None, None])

        return QSASelection(
            token_idx=mx.concatenate([tok, tail], axis=-1).astype(mx.int32),
            valid=mx.concatenate([valid_blk, tail_valid], axis=-1),
        )

    def __call__(
        self, x: mx.array, rope: Any, cache: Any, offset: int
    ) -> Optional[QSASelection]:
        prep = self.prepare(x, rope, cache, offset)
        if prep is None:
            return None
        S = int(prep.q.shape[1])
        n_blocks = int(prep.pooled.shape[1])
        chunk = _indexer_score_chunk(n_blocks, _qsa_query_chunk())
        if S <= chunk:
            return self.select(
                prep.q, prep.q_pos, prep.pooled, prep.kv_len, prep.left_padding
            )
        parts: List[QSASelection] = []
        for s0 in range(0, S, chunk):
            s1 = min(s0 + chunk, S)
            parts.append(
                self.select(
                    prep.q[:, s0:s1],
                    prep.q_pos[s0:s1],
                    prep.pooled,
                    prep.kv_len,
                    prep.left_padding,
                )
            )
        return QSASelection(
            token_idx=mx.concatenate([p.token_idx for p in parts], axis=1),
            valid=mx.concatenate([p.valid for p in parts], axis=1),
        )


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        # q_proj also carries the output gate: n_heads * head_dim * 2
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)
        # House attention is (x, mask, cache). RoPE is a module attr (shared
        # with the trunk RotaryEmbedding); the QSA indexer cache lives on
        # cache.indexer. Runtime split-attention hooks assume this shape.
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, args.rope_theta, args.rope_parameters)

    def _core_attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Any,
        cache: Any,
        sel: Optional[QSASelection],
    ) -> mx.array:
        if sel is None:
            return scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
        if _qsa_dense_mask_enabled():
            kv_len = int(k.shape[2])
            add = selection_to_dense_add_mask(sel, kv_len, q.dtype)
            mask = (
                add
                if mask is None
                else (mask + add if not isinstance(mask, str) else add)
            )
            return scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
        return _qsa_gather_attention(q, k, v, sel, self.scale, mask=mask)

    def __call__(
        self,
        x: mx.array,
        mask: Any = None,
        cache: Any = None,
    ) -> mx.array:
        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0
        idx_cache = getattr(cache, "indexer", None) if cache is not None else None
        rope = self.rope

        q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Contiguous positions: fused kernel. QSAIndexer keeps _rope_partial
        # (block_starts are strided — that path is owned by the other lane).
        # MTP _ShiftedRope exposes .dim/.freqs/.position_shift so the same
        # kernel applies with the draft position offset.
        freqs = getattr(rope, "freqs", None)
        dims = getattr(rope, "dim", None)
        # mx.fast.rope has no amplitude scale; yarn (mscale != 1) takes the
        # cos/sin path where mscale is folded into the rotation.
        if (
            freqs is not None
            and dims is not None
            and float(getattr(rope, "mscale", 1.0) or 1.0) == 1.0
        ):
            rope_offset = offset + int(getattr(rope, "position_shift", 0) or 0)
            q = mx.fast.rope(
                q,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=rope_offset,
                freqs=freqs,
            )
            k = mx.fast.rope(
                k,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=rope_offset,
                freqs=freqs,
            )
        else:
            cos, sin = rope(mx.arange(offset, offset + S)[None])
            cos, sin = cos[:, None], sin[:, None]
            q, k = _rope_partial(q, cos, sin), _rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        indexer = self.indexer
        if indexer is None:
            out = self._core_attend(q, k, v, mask, cache, None)
        # Decode / MTP reuse wrapper: keep the S==1 __call__ contract.
        elif S == 1 or not hasattr(indexer, "prepare"):
            sel = indexer(x, rope, idx_cache, offset)
            if sel is not None and not isinstance(sel, QSASelection) and hasattr(sel, "ndim"):
                # Dense keep mask from an older wrapper; convert to additive.
                add = mx.where(sel, mx.array(0, q.dtype), _qsa_neg_inf(q.dtype))
                mask = (
                    add
                    if mask is None
                    else (mask + add if not isinstance(mask, str) else add)
                )
                sel = None
            if sel is not None:
                _log_qsa_path_once(
                    "qsa-gather "
                    f"S={S} kv={k.shape[2]} K={sel.token_idx.shape[-1]} "
                    f"dense_mask={_qsa_dense_mask_enabled()}"
                )
            out = self._core_attend(q, k, v, mask, cache, sel)
        else:
            prep = indexer.prepare(x, rope, idx_cache, offset)
            if prep is None:
                out = self._core_attend(q, k, v, mask, cache, None)
            else:
                kv_len = prep.kv_len
                n_blocks = int(prep.pooled.shape[1])
                score_chunk = _indexer_score_chunk(n_blocks, _qsa_query_chunk())
                gather_chunk = _qsa_query_chunk()
                chunk = min(score_chunk, gather_chunk)
                _log_qsa_path_once(
                    "qsa-gather "
                    f"S={S} kv={kv_len} n_blocks={n_blocks} chunk={chunk} "
                    f"K~{self.indexer.token_budget} dense_mask={_qsa_dense_mask_enabled()}"
                )
                n_chunks = (S + chunk - 1) // chunk
                pieces: List[mx.array] = []
                for s0 in range(0, S, chunk):
                    s1 = min(s0 + chunk, S)
                    sel = indexer.select(
                        prep.q[:, s0:s1],
                        prep.q_pos[s0:s1],
                        prep.pooled,
                        prep.kv_len,
                        prep.left_padding,
                    )
                    mask_c = mask
                    if mask_c is not None and not isinstance(mask_c, str):
                        mask_c = mask_c[..., s0:s1, : kv_len]
                    piece = self._core_attend(
                        q[:, :, s0:s1], k, v, mask_c, cache, sel
                    )
                    if n_chunks > 1:
                        mx.eval(piece)
                    pieces.append(piece)
                out = mx.concatenate(pieces, axis=2)

        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# ---------------------------------------------------------------------------
# Gated DeltaNet (linear attention)
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_v = args.linear_num_value_heads
        self.n_k = args.linear_num_key_heads
        self.dk = args.linear_key_head_dim
        self.dv = args.linear_value_head_dim
        self.key_dim = self.dk * self.n_k
        self.value_dim = self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        # Load-time fused projections (old split keys are remapped in sanitize).
        self.in_proj_qkvz = nn.Linear(d, self.conv_dim + self.value_dim, bias=False)
        self.in_proj_ba = nn.Linear(d, 2 * self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(
            self.dv, eps=args.rms_norm_eps, activation=args.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)
        # L2(x, eps) == rms_norm(x, weight=None, eps=eps/D) * (D**-0.5).
        # Avoid allocating/passing weight array every step.
        object.__setattr__(self, "_l2_eps", 1e-6 / float(self.dk))
        inv_scale = self.dk**-0.5
        object.__setattr__(self, "_inv_scale", inv_scale)
        object.__setattr__(self, "_inv_scale_sq", inv_scale**2)

    def __call__(self, x: mx.array, mask: Any, cache: Any) -> mx.array:
        B, S, _ = x.shape
        qkvz = self.in_proj_qkvz(x)
        mixed_qkv, z = mx.split(qkvz, [self.conv_dim], axis=-1)
        z = z.reshape(B, S, self.n_v, self.dv)
        b, a = mx.split(self.in_proj_ba(x), [self.n_v], axis=-1)

        slot0 = _cache_slot(cache, 0) if cache is not None else None
        conv_state = (
            slot0
            if slot0 is not None
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        K_minus_1 = self.conv_kernel_size - 1
        if mask is None:
            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            if cache is not None:
                _set_cache_slot(
                    cache,
                    0,
                    mx.contiguous(conv_input[:, -K_minus_1:, :]),
                )
            conv_out = nn.silu(self.conv1d(conv_input))
        elif S == 1:
            row_active = (
                mask.any(axis=-1, keepdims=True)
                if mask.ndim > 1
                else mask[:, None]
            )
            mixed_qkv_active = mx.where(mask[..., None], mixed_qkv, 0)
            conv_input = mx.concatenate([conv_state, mixed_qkv_active], axis=1)
            if cache is not None:
                new_slot0 = mx.contiguous(conv_input[:, -K_minus_1:, :])
                new_slot0 = mx.where(row_active[..., None], new_slot0, conv_state)
                _set_cache_slot(cache, 0, new_slot0)
            conv_out = nn.silu(self.conv1d(conv_input))
            conv_out = mx.where(mask[..., None], conv_out, 0)
        else:
            conv_out_rows = []
            new_slot0_rows = []
            for i in range(B):
                m_i = mask[i] if mask.ndim > 1 else mask
                row_qkv = mixed_qkv[i]
                c_state_i = conv_state[i : i + 1]
                m_list = m_i.tolist() if hasattr(m_i, "tolist") else list(m_i)
                valid_idx = [idx for idx, v in enumerate(m_list) if v]
                n_valid = len(valid_idx)
                if n_valid == 0:
                    conv_out_rows.append(mx.zeros((1, S, self.conv_dim), dtype=x.dtype))
                    new_slot0_rows.append(c_state_i)
                elif n_valid == S:
                    c_inp = mx.concatenate([c_state_i, row_qkv[None, ...]], axis=1)
                    new_slot0_rows.append(mx.contiguous(c_inp[:, -K_minus_1:, :]))
                    conv_out_rows.append(nn.silu(self.conv1d(c_inp)))
                else:
                    valid_qkv = row_qkv[valid_idx][None, ...]
                    c_inp = mx.concatenate([c_state_i, valid_qkv], axis=1)
                    new_slot0_rows.append(mx.contiguous(c_inp[:, -K_minus_1:, :]))
                    v_out = nn.silu(self.conv1d(c_inp))
                    full_out = mx.zeros((1, S, self.conv_dim), dtype=x.dtype)
                    full_out[:, valid_idx] = v_out
                    conv_out_rows.append(full_out)
            if cache is not None:
                _set_cache_slot(cache, 0, mx.concatenate(new_slot0_rows, axis=0))
            conv_out = mx.concatenate(conv_out_rows, axis=0)

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        q = mx.fast.rms_norm(q, None, self._l2_eps) * self._inv_scale_sq
        k = mx.fast.rms_norm(k, None, self._l2_eps) * self._inv_scale

        state = _cache_slot(cache, 1) if cache is not None else None
        out, state = self._gated_delta_update(q, k, v, a, b, state, mask)
        if cache is not None:
            _set_cache_slot(cache, 1, state)
            if hasattr(cache, "advance"):
                cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))

    def _gated_delta_update(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        a: mx.array,
        b: mx.array,
        state: Any,
        mask: Any,
    ) -> Tuple[mx.array, mx.array]:
        """Prefill (T>1) uses the blocked-sequential kernel when eligible.

        Maps in_proj_qkv/z/b/a + conv + L2-normed q/k onto the kernel's
        (q, k, v, g, beta, state) contract. Decode (T==1), training, CPU,
        FORCE_STOCK, and any failed structural check stay on
        ``gated_delta_update``.
        """
        if q.shape[1] == 1:
            return gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )
        if (
            q.shape[1] > 1
            and not self.training
            and mx.default_device() == mx.gpu
            and not _blocked_prefill_force_stock()
        ):
            beta = mx.sigmoid(b)
            g = compute_g(self.A_log, a, self.dt_bias)
            if blocked_prefill_eligible(q, v, g, mask, state):
                if state is None:
                    state = mx.zeros(
                        (q.shape[0], self.n_v, self.dv, self.dk), dtype=mx.float32
                    )
                _log_blocked_prefill_once(q)
                return gated_delta_blocked_prefill(q, k, v, g, beta, state)
            reason = blocked_prefill_ineligibility_reason(q, v, g, mask, state)
            if reason is not None:
                _log_blocked_prefill_fallback_once(reason)
        return gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )


# ---------------------------------------------------------------------------
# MoE & Feed-Forward blocks
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.gate_up_proj = nn.Linear(dim, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), [self.hidden], axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class FusedSwitchGLU(SwitchGLU):
    """SwitchGLU with gate/up packed into one gathered projection.

    mlx-lm's SwitchGLU owns two SwitchLinear modules. Subclassing keeps the
    gather/sort contract; load-time remap concatenates the old split weights.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=None,
        bias: bool = False,
    ):
        nn.Module.__init__(self)
        self.hidden_dims = hidden_dims
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = SwiGLU() if activation is None else activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        packed = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x_gate, x_up = mx.split(packed, [self.hidden_dims], axis=-1)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = FusedSwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)
        self.norm_topk_prob = getattr(args, "norm_topk_prob", True)
        self.routed_scaling_factor = float(getattr(args, "routed_scaling_factor", 1.0) or 1.0)

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        if self.norm_topk_prob:
            w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        else:
            scores = mx.softmax(logits, axis=-1, precise=True)
            w = mx.take_along_axis(scores, idx, axis=-1)
        if self.routed_scaling_factor != 1.0:
            w = w * self.routed_scaling_factor
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


# ---------------------------------------------------------------------------
# Hyper-Connections (gated residual)
# ---------------------------------------------------------------------------


class GatedResidual(nn.Module):
    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_dim, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dim, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_dim, self.hc, bias=False) if use_combine else None
        )

    def __call__(self, hyper: mx.array):
        normed = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject


# ---------------------------------------------------------------------------
# N-gram & PLE (Per-Layer Embedding)
# ---------------------------------------------------------------------------


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class _ShardedEmbedding(nn.Module):
    """One contiguous embedding table, addressed by global index.

    Checkpoints stay sharded on disk (`shard_i.weight` / scales / biases).
    :func:`sanitize` concatenates those shards along axis 0 into ``shard_0``
    so ``__call__`` is a single gather. The child is named ``shard_0`` so
    quantized configs keyed at ``...ngram_embedding.shard_0`` (group_size=32)
    still match. ``gid`` is already ``shard * rows + row``.
    """

    def __init__(self, n_shards: int, rows: int, dim: int):
        super().__init__()
        self.n_shards = n_shards
        self.rows = rows
        self.dim = dim
        self.shard_0 = nn.Embedding(n_shards * rows, dim)

    def __call__(self, gid: mx.array) -> mx.array:
        return self.shard_0(gid)


class NGramEmbedding(nn.Module):
    """N-gram hash table sharded into `split_ngram_parts` sub-embeddings."""

    def __init__(self, args: TextArgs, embed_dim: int, ple_layer_index: int = 0):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + h
            s = _nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes

        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        self.n_shards = args.split_ngram_parts
        self.rows_per_shard = math.ceil(padded / self.n_shards)
        self.ngram_embedding = _ShardedEmbedding(
            self.n_shards, self.rows_per_shard, head_dim
        )

        mults = []
        max_long = (1 << 63) - 1
        half = max(1, (max_long // max(args.vocab_size, 1)) // 2)
        base_seed = args.seed + _PRIME_1 * ple_layer_index
        for i in range(self.ngram_size):
            mults.append(
                2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1
            )
        self.layer_multipliers = mx.array(mults, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self._mults = mx.array(mults, dtype=mx.int64)
        self._sizes = mx.array(sizes, dtype=mx.int64)
        self._offsets = mx.array(offsets, dtype=mx.int64)

    def _shift_right(self, ids: mx.array, shift: int) -> mx.array:
        """Shift right by `shift`, without crossing an EOS boundary."""
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = mx.arange(T)
        eos_pos = mx.where(ids == self.eos_token_id, pos, -1)
        prev_incl = mx.cummax(eos_pos, axis=1)
        prev = mx.concatenate(
            [mx.full((B, 1), -1, dtype=prev_incl.dtype), prev_incl[:, :-1]], axis=1
        )
        in_segment = pos[None] - (prev + 1)
        src = pos - shift
        gathered = mx.take_along_axis(
            ids, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1
        )
        ok = (in_segment >= shift) & (src[None] >= 0)
        return mx.where(ok, gathered, self.eos_token_id)

    def __call__(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        n_new = ids.shape[1]
        history = mx.concatenate([prev_context, ids], axis=1).astype(mx.int64)
        shifted = [self._shift_right(history, s) for s in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self._mults[0]
            for p in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self._mults[p])
            gid = mixed[..., None] % self._sizes[lo:hi].reshape(1, 1, -1)
            blocks.append(gid + self._offsets[lo:hi].reshape(1, 1, -1))

        gid = mx.concatenate(blocks, axis=-1)[:, -n_new:]
        return self.ngram_embedding(gid).reshape(*gid.shape[:2], -1)


class PLELayer(nn.Module):
    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.d = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = NGramEmbedding(args, args.ple_embed_dim, ple_layer_index)
        k = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.short_conv_state_len = (k - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_dim,
            hc_dim,
            kernel_size=k,
            groups=hc_dim,
            dilation=self.dilation,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache: Any, mask: Any = None) -> mx.array:
        B, S, _ = x.shape
        n = self.short_conv_state_len
        slot = _cache_slot(cache, 2)
        state = (
            slot
            if slot is not None
            else mx.zeros((B, n, x.shape[-1]), dtype=x.dtype)
        )
        if mask is None:
            full = mx.concatenate([state, x], axis=1)
            if cache is not None:
                _set_cache_slot(cache, 2, mx.contiguous(full[:, -n:, :]))
            out = nn.silu(self.conv1d(full[:, -(n + S) :, :]))
            return out
        elif S == 1:
            row_active = (
                mask.any(axis=-1, keepdims=True)
                if mask.ndim > 1
                else mask[:, None]
            )
            x_active = mx.where(mask[..., None], x, 0)
            full = mx.concatenate([state, x_active], axis=1)
            if cache is not None:
                new_slot2 = mx.contiguous(full[:, -n:, :])
                new_slot2 = mx.where(row_active[..., None], new_slot2, state)
                _set_cache_slot(cache, 2, new_slot2)
            out = nn.silu(self.conv1d(full[:, -(n + S) :, :]))
            out = mx.where(mask[..., None], out, 0)
            return out
        else:
            out_rows = []
            new_slot2_rows = []
            for i in range(B):
                m_i = mask[i] if mask.ndim > 1 else mask
                row_x = x[i]
                s_i = state[i : i + 1]
                m_list = m_i.tolist() if hasattr(m_i, "tolist") else list(m_i)
                valid_idx = [idx for idx, v in enumerate(m_list) if v]
                n_valid = len(valid_idx)
                if n_valid == 0:
                    out_rows.append(mx.zeros((1, S, x.shape[-1]), dtype=x.dtype))
                    new_slot2_rows.append(s_i)
                elif n_valid == S:
                    full_i = mx.concatenate([s_i, row_x[None, ...]], axis=1)
                    new_slot2_rows.append(mx.contiguous(full_i[:, -n:, :]))
                    out_rows.append(nn.silu(self.conv1d(full_i[:, -(n + S) :, :])))
                else:
                    valid_x = row_x[valid_idx][None, ...]
                    full_i = mx.concatenate([s_i, valid_x], axis=1)
                    new_slot2_rows.append(mx.contiguous(full_i[:, -n:, :]))
                    v_out = nn.silu(self.conv1d(full_i[:, -(n + n_valid) :, :]))
                    full_out = mx.zeros((1, S, x.shape[-1]), dtype=x.dtype)
                    full_out[:, valid_idx] = v_out
                    out_rows.append(full_out)

            if cache is not None:
                _set_cache_slot(cache, 2, mx.concatenate(new_slot2_rows, axis=0))
            return mx.concatenate(out_rows, axis=0)

    def __call__(
        self,
        hidden: mx.array,
        ids: mx.array,
        prev_ctx: mx.array,
        cache: Any,
        mask: Any = None,
    ) -> mx.array:
        emb = self.ple_embedding(ids, prev_ctx).astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc, self.d)

        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0)
        out = gated + self._short_conv(self.norm_conv(gated), cache, mask=mask)
        if mask is not None:
            out = mx.where(mask[..., None], out, 0)
        return out


# ---------------------------------------------------------------------------
# Decoder layer & Model trunk
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        ple_idx = (
            args.ple_layer_ids.index(layer_idx + 1)
            if (layer_idx + 1) in args.ple_layer_ids
            else None
        )
        self.ple = PLELayer(args, ple_idx) if ple_idx is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(
        self,
        h: mx.array,
        rope: Any,
        mask: Any,
        conv_mask: Any,
        cache: Any,
        idx_cache: Any,
        ids: mx.array,
        prev_ctx: Any,
    ) -> mx.array:
        del idx_cache  # QSA indexer cache is cache.indexer; house attn is (x, mask, cache)
        if self.ple is not None:
            h = h + self.ple(h, ids, prev_ctx, cache, conv_mask)

        x, hyper, inject = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            # House call: (x, mask, cache). rope/idx_cache are attrs — see Attention.
            if getattr(self.self_attn, "rope", None) is None:
                self.self_attn.rope = rope
            x = self.self_attn(x, mask, cache)
        h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)

        x, hyper, inject = self.mlp_hyper_connection(h)
        x = self.mlp(x)
        return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


def _build_ple_tail_context(
    prev_ctx: mx.array,
    ids: mx.array,
    ctx_len: int,
    eos: int,
    left_padding: Optional[Any] = None,
) -> mx.array:
    if left_padding is None:
        return mx.concatenate([prev_ctx, ids], axis=1)[:, -ctx_len:]

    B, S = ids.shape
    lp_list = (
        left_padding.tolist()
        if hasattr(left_padding, "tolist")
        else list(left_padding)
    )
    rows = []
    for i in range(B):
        lp = int(lp_list[i]) if i < len(lp_list) else 0
        v_len = max(0, S - lp)
        if v_len >= ctx_len:
            rows.append(ids[i : i + 1, S - ctx_len : S])
        elif v_len > 0:
            needed = ctx_len - v_len
            prefix = prev_ctx[i : i + 1, -needed:]
            suffix = ids[i : i + 1, lp:S]
            rows.append(mx.concatenate([prefix, suffix], axis=1))
        else:
            rows.append(prev_ctx[i : i + 1])
    return mx.concatenate(rows, axis=0)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, args.rope_theta, args.rope_parameters)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.rope = self.rope
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.first_full_attn_idx = next(
            (i for i, l in enumerate(self.layers) if l.layer_type == "full_attention"),
            None,
        )
        self.first_linear_attn_idx = next(
            (i for i, l in enumerate(self.layers) if l.layer_type == "linear_attention"),
            None,
        )
        self.ple_layers = [
            i for i in range(args.num_hidden_layers) if (i + 1) in args.ple_layer_ids
        ]

    def __call__(
        self,
        ids: mx.array | None = None,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hyper: bool = False,
    ) -> mx.array | Tuple[mx.array, mx.array]:
        if ids is not None and ids.ndim == 1:
            ids = ids[None]
        if input_embeddings is not None and input_embeddings.ndim == 2:
            input_embeddings = input_embeddings[None]

        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        attn_cache = (
            cache[self.first_full_attn_idx]
            if (
                self.first_full_attn_idx is not None
                and len(cache) > self.first_full_attn_idx
                and cache[self.first_full_attn_idx] is not None
            )
            else None
        )
        mask = create_attention_mask(
            h, [attn_cache] if attn_cache is not None else None
        )
        linear_cache = (
            cache[self.first_linear_attn_idx]
            if (
                self.first_linear_attn_idx is not None
                and len(cache) > self.first_linear_attn_idx
                and cache[self.first_linear_attn_idx] is not None
            )
            else None
        )
        if linear_cache is None:
            linear_cache = next(
                (c for c in cache if c is not None and hasattr(c, "make_mask")),
                None,
            )
        conv_mask = create_ssm_mask(h, linear_cache)

        left_padding = getattr(linear_cache, "left_padding", None)
        if left_padding is None:
            for c in cache:
                if c is not None:
                    if getattr(c, "left_padding", None) is not None:
                        left_padding = c.left_padding
                        break
                    if (
                        hasattr(c, "indexer")
                        and getattr(c.indexer, "left_padding", None) is not None
                    ):
                        left_padding = c.indexer.left_padding
                        break

        prev_ctx = None
        if self.ple_layers and ids is not None:
            ctx_len = self.args.ngram_size - 1
            eos = self.args.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            pc = cache[self.ple_layers[0]] if len(cache) > self.ple_layers[0] else None
            prev = _cache_slot(pc, 3)
            prev_ctx = (
                prev
                if prev is not None
                else mx.full((ids.shape[0], ctx_len), eos, ids.dtype)
            )
            if pc is not None:
                tail = _build_ple_tail_context(
                    prev_ctx, ids, ctx_len, eos, left_padding
                )
                _set_cache_slot(pc, 3, tail)

        h = mx.tile(h, (1, 1, self.hc))
        if _compile_enabled() and (
            (ids is not None and ids.shape[1] == 1)
            or (input_embeddings is not None and input_embeddings.shape[1] == 1)
        ):
            from .qwen4_exp_compiled import run_compiled_layers

            h = run_compiled_layers(
                self, h, self.rope, mask, conv_mask, cache, ids, prev_ctx
            )
        else:
            for layer, c in zip(self.layers, cache):
                idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
                h = layer(h, self.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        mixed = self.hyper_connection_mixer(h)
        if return_hyper:
            return mixed, h
        return mixed


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


class _IndexerCache(_BaseCache):
    """Holds the indexer raw keys and cached pooled block representations."""

    _GROW = 256

    def __init__(self):
        self._buf = None
        self._len = 0
        self.pooled = None
        self.left_padding = None

    @property
    def keys(self):
        if self._buf is None:
            return None
        return self._buf[:, : self._len, :]

    @keys.setter
    def keys(self, v):
        # Restore/trim path (MTP rollback): adopt the array as-is; growth
        # resumes by chunked reallocation on the next update().
        self._buf = v
        self._len = 0 if v is None else int(v.shape[1])
        self.pooled = None

    def update(self, k: mx.array) -> mx.array:
        # Chunked growth (mlx-lm KVCache style): amortized O(1) appends
        # instead of a full copy of history per decode step.
        n = int(k.shape[1])
        if self._buf is None:
            pad = (-n) % self._GROW
            if pad:
                B, _, D = k.shape
                self._buf = mx.concatenate(
                    [k, mx.zeros((B, pad, D), dtype=k.dtype)], axis=1
                )
            else:
                self._buf = k
            self._len = n
            return self.keys
        cap = int(self._buf.shape[1])
        if self._len + n > cap:
            grow = ((self._len + n - cap + self._GROW - 1) // self._GROW) * self._GROW
            B, _, D = self._buf.shape
            self._buf = mx.concatenate(
                [self._buf, mx.zeros((B, grow, D), dtype=self._buf.dtype)], axis=1
            )
        self._buf[:, self._len : self._len + n, :] = k
        self._len += n
        return self.keys

    @property
    def batch_size(self) -> int:
        if self._buf is not None:
            return int(self._buf.shape[0])
        return 1

    def empty(self) -> bool:
        return self._buf is None or self._len == 0

    def filter(self, batch_indices: Any) -> None:
        if self._buf is not None:
            self._buf = self._buf[batch_indices]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        self.pooled = None

    def extend(self, other: Any) -> None:
        if other is None:
            return
        self_k = self.keys
        other_k = getattr(other, "keys", None)
        if self_k is None and other_k is None:
            self._buf = None
            self._len = 0
            self.pooled = None
            return

        a_batch = self.batch_size
        b_batch = getattr(other, "batch_size", 1)
        L1 = int(self_k.shape[1]) if self_k is not None else 0
        L2 = int(other_k.shape[1]) if other_k is not None else 0
        max_L = max(L1, L2)
        D = (
            self_k.shape[-1]
            if self_k is not None
            else (other_k.shape[-1] if other_k is not None else 128)
        )
        dt = (
            self_k.dtype
            if self_k is not None
            else (other_k.dtype if other_k is not None else mx.float32)
        )

        def pad_right_aligned(k, batch, cur_len):
            if k is None:
                return mx.zeros((batch, max_L, D), dtype=dt)
            if cur_len < max_L:
                pad = max_L - cur_len
                return mx.concatenate([mx.zeros((batch, pad, D), dtype=dt), k], axis=1)
            return k

        a_pad = pad_right_aligned(self_k, a_batch, L1)
        b_pad = pad_right_aligned(other_k, b_batch, L2)
        self._buf = mx.concatenate([a_pad, b_pad], axis=0)
        self._len = max_L
        other_lp = getattr(other, "left_padding", None)
        if self.left_padding is not None or other_lp is not None:
            a_lp = (
                self.left_padding
                if self.left_padding is not None
                else mx.zeros((a_batch,), dtype=mx.int32)
            )
            b_lp = (
                other_lp
                if other_lp is not None
                else mx.zeros((b_batch,), dtype=mx.int32)
            )
            self.left_padding = mx.concatenate([a_lp, b_lp], axis=0)
        self.pooled = None

    def extract(self, idx: int) -> "_IndexerCache":
        cache = _IndexerCache()
        if self._buf is not None and self._len > 0:
            k = self._buf[idx : idx + 1, : self._len, :]
            cache.keys = mx.contiguous(k)
        if self.left_padding is not None:
            cache.left_padding = self.left_padding[idx : idx + 1]
        return cache

    @classmethod
    def merge(cls, caches: list[Any]) -> "_IndexerCache":
        cache = cls()
        if not caches or all(
            c is None
            or (hasattr(c, "empty") and c.empty())
            or getattr(c, "_buf", None) is None
            for c in caches
        ):
            return cache
        keys_list = [getattr(c, "keys", None) for c in caches]
        lengths = [int(k.shape[1]) if k is not None else 0 for k in keys_list]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            return cache
        B = len(caches)
        sample = next(k for k in keys_list if k is not None)
        D = sample.shape[-1]
        dt = sample.dtype
        buf = mx.zeros((B, max_len, D), dtype=dt)
        for i, (l, k) in enumerate(zip(lengths, keys_list)):
            if k is not None and l > 0:
                pad = max_len - l
                buf[i : i + 1, pad:max_len, :] = k
        cache._buf = buf
        cache._len = max_len
        lps = [getattr(c, "left_padding", None) for c in caches]
        if any(lp is not None for lp in lps):
            lp_list = [
                lp
                if lp is not None
                else mx.zeros((getattr(c, "batch_size", 1),), dtype=mx.int32)
                for lp, c in zip(lps, caches)
            ]
            cache.left_padding = mx.concatenate(lp_list, axis=0)
        cache.pooled = None
        return cache

    def trim(self, n: int) -> None:
        if self._buf is not None and self._len > 0:
            keep = self._len - int(n)
            if keep <= 0:
                self._buf = None
                self._len = 0
            else:
                self._len = keep
        self.pooled = None

    @property
    def state(self):
        return self.keys

    @state.setter
    def state(self, v):
        self.keys = v


class _AttnCache(KVCache):
    def __init__(self):
        super().__init__()
        self.indexer = _IndexerCache()
        self.left_padding = None
        self._mtplx_indexer_trim = True

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        res = super().update_and_fetch(keys, values)
        if isinstance(res, tuple) and len(res) >= 2:
            return res[0], res[1]
        return res

    def trim(self, n: int) -> int:
        trimmed = super().trim(n)
        dropped = int(n if trimmed is None else trimmed)
        if hasattr(self, "indexer") and self.indexer is not None:
            if dropped:
                self.indexer.trim(dropped)
        reuse = getattr(self, "_sparse_index_reuse", None)
        if reuse is not None and hasattr(reuse, "reset"):
            reuse.reset()
        return trimmed

    def filter(self, batch_indices: Any) -> None:
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        if getattr(self, "left_padding", None) is not None:
            self.left_padding = self.left_padding[batch_indices]
        if getattr(self, "offset", None) is not None and isinstance(self.offset, mx.array):
            self.offset = self.offset[batch_indices]
        if getattr(self, "lengths", None) is not None:
            self.lengths = self.lengths[batch_indices]
        if getattr(self, "_lengths", None) is not None:
            self._lengths = self._lengths[batch_indices]
        if (
            hasattr(self, "indexer")
            and self.indexer is not None
            and hasattr(self.indexer, "filter")
        ):
            self.indexer.filter(batch_indices)
            if getattr(self, "left_padding", None) is not None:
                self.indexer.left_padding = self.left_padding

    def extend(self, other: Any) -> None:
        if other is None:
            return
        a_batch = self.keys.shape[0] if self.keys is not None else 1
        b_batch = other.keys.shape[0] if getattr(other, "keys", None) is not None else 1
        if self.keys is None and getattr(other, "keys", None) is None:
            pass
        elif self.keys is None:
            self.keys = other.keys
            self.values = other.values
            self.offset = getattr(other, "offset", 0)
        elif getattr(other, "keys", None) is not None:
            self.keys = mx.concatenate([self.keys, other.keys], axis=0)
            self.values = mx.concatenate([self.values, other.values], axis=0)
        other_lp = getattr(other, "left_padding", None)
        if getattr(self, "left_padding", None) is not None or other_lp is not None:
            a_lp = (
                self.left_padding
                if getattr(self, "left_padding", None) is not None
                else mx.zeros((a_batch,), dtype=mx.int32)
            )
            b_lp = (
                other_lp
                if other_lp is not None
                else mx.zeros((b_batch,), dtype=mx.int32)
            )
            self.left_padding = mx.concatenate([a_lp, b_lp], axis=0)
        other_len = getattr(other, "lengths", None)
        if getattr(self, "lengths", None) is not None or other_len is not None:
            a_len = (
                self.lengths
                if getattr(self, "lengths", None) is not None
                else mx.zeros(
                    (a_batch,),
                    dtype=mx.int32,
                )
            )
            b_len = (
                other_len
                if other_len is not None
                else mx.zeros(
                    (b_batch,),
                    dtype=mx.int32,
                )
            )
            self.lengths = mx.concatenate([a_len, b_len], axis=0)
        if (
            hasattr(self, "indexer")
            and self.indexer is not None
            and hasattr(self.indexer, "extend")
        ):
            self.indexer.extend(getattr(other, "indexer", None))

    def extract(self, idx: int) -> "_AttnCache":
        cache = _AttnCache()
        if self.keys is not None:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1])
            cache.values = mx.contiguous(self.values[idx : idx + 1])
            cache.offset = cache.keys.shape[2]
        if getattr(self, "left_padding", None) is not None:
            cache.left_padding = self.left_padding[idx : idx + 1]
        if getattr(self, "lengths", None) is not None:
            cache.lengths = self.lengths[idx : idx + 1]
        if getattr(self, "_lengths", None) is not None:
            cache._lengths = self._lengths[idx : idx + 1]
        if (
            hasattr(self, "indexer")
            and self.indexer is not None
            and hasattr(self.indexer, "extract")
        ):
            cache.indexer = self.indexer.extract(idx)
        return cache

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        if getattr(self, "left_padding", None) is not None:
            from mlx_lm.models.cache import create_causal_mask

            offset = getattr(self, "_idx", getattr(self, "offset", 0))
            window_size = kwargs.pop("window_size", None)
            return create_causal_mask(
                N,
                offset=offset,
                left_padding=self.left_padding,
                window_size=window_size,
                **kwargs,
            )
        return super().make_mask(N, return_array=return_array, **kwargs)

    @classmethod
    def merge(cls, caches: list[Any]):
        merged = super().merge(caches)
        indexers = [getattr(c, "indexer", None) for c in caches]
        merged.indexer = _IndexerCache.merge(indexers)
        lps = [getattr(c, "left_padding", None) for c in caches]
        if any(lp is not None for lp in lps):
            lengths = [
                getattr(c, "size", lambda: getattr(c, "offset", 0))()
                for c in caches
            ]
            max_length = max(lengths) if lengths else 0
            lp_list = []
            for c, l in zip(caches, lengths):
                pad_offset = max_length - l
                c_lp = getattr(c, "left_padding", None)
                if c_lp is not None:
                    lp_list.append(c_lp + pad_offset)
                else:
                    b = (
                        c.keys.shape[0]
                        if getattr(c, "keys", None) is not None
                        else 1
                    )
                    lp_list.append(mx.full((b,), pad_offset, dtype=mx.int32))
            merged.left_padding = mx.concatenate(lp_list, axis=0)
        if getattr(merged, "left_padding", None) is not None:
            merged.indexer.left_padding = merged.left_padding
        from mtplx.qwen4_exp_mtp_patch import _install_indexer_aware_trim

        _install_indexer_aware_trim(merged)
        return merged

    @property
    def state(self):
        indexer_keys = getattr(self.indexer, "keys", None)
        if self.keys is None or self.values is None:
            return self.keys, self.values, indexer_keys
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values, indexer_keys
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
            indexer_keys,
        )

    @state.setter
    def state(self, v):
        if v is None:
            self.keys = self.values = None
            self.offset = 0
            if hasattr(self, "indexer") and self.indexer is not None:
                self.indexer.keys = None
                self.indexer._len = 0
            return
        if len(v) == 3:
            keys, values, indexer_keys = v
            self.keys = keys
            self.values = values
            self.offset = 0 if self.keys is None else int(self.keys.shape[2])
            if hasattr(self, "indexer") and self.indexer is not None:
                self.indexer.keys = indexer_keys
                self.indexer._len = (
                    0 if indexer_keys is None else int(indexer_keys.shape[1])
                )
        else:
            keys, values = v
            self.keys = keys
            self.values = values
            self.offset = 0 if self.keys is None else int(self.keys.shape[2])


# ---------------------------------------------------------------------------
# Top-level Model & Weight sanitization
# ---------------------------------------------------------------------------

_NGRAM_SHARD_MARKER = ".ngram_embedding.shard_"
_NGRAM_SHARD_LEAVES = ("weight", "scales", "biases")


def _concat_sharded_embedding_tables(weights: Dict[str, Any]) -> None:
    """Fuse on-disk ngram shards into ``shard_0`` along axis 0, in-place.

    Quantized shards concat ``weight`` / ``scales`` / ``biases`` independently
    (rows are the concat axis, so quant groups stay valid). Unquantized shards
    concat ``weight`` only. Destination is allocated first and each source
    shard is ``mx.eval``'d then deleted so peak extra memory is one table plus
    the remaining unfused shards, not a second full copy of the model.
    """
    n_shards_for: Dict[str, int] = {}
    for key in weights:
        idx = key.find(_NGRAM_SHARD_MARKER)
        if idx < 0:
            continue
        rest = key[idx + len(_NGRAM_SHARD_MARKER) :]
        shard_str, _, leaf = rest.partition(".")
        if not shard_str.isdigit() or leaf not in _NGRAM_SHARD_LEAVES:
            continue
        prefix = key[: idx + len(".ngram_embedding")]
        n = int(shard_str) + 1
        if n > n_shards_for.get(prefix, 0):
            n_shards_for[prefix] = n

    for prefix, n_shards in n_shards_for.items():
        if n_shards <= 1:
            continue
        for leaf in _NGRAM_SHARD_LEAVES:
            k0 = f"{prefix}.shard_0.{leaf}"
            if k0 not in weights:
                continue
            row_counts = [
                int(weights[f"{prefix}.shard_{i}.{leaf}"].shape[0])
                for i in range(n_shards)
            ]
            sample = weights[k0]
            dest = mx.zeros((sum(row_counts), *sample.shape[1:]), dtype=sample.dtype)
            mx.eval(dest)
            del sample
            offset = 0
            for i, n_rows in enumerate(row_counts):
                src = weights.pop(f"{prefix}.shard_{i}.{leaf}")
                dest = mx.slice_update(
                    dest, src, mx.array(offset, dtype=mx.int32), (0,)
                )
                mx.eval(dest)
                offset += n_rows
                del src
            weights[k0] = dest
            gc.collect()


_PROJ_LEAVES = ("weight", "scales", "biases", "bias")


def _as_mx(value: Any) -> mx.array:
    return value if isinstance(value, mx.array) else mx.array(value)


def _proj_leaves(weights: Dict[str, Any], stem: str) -> Dict[str, Any]:
    leaves: Dict[str, Any] = {}
    for leaf in _PROJ_LEAVES:
        key = f"{stem}.{leaf}"
        if key in weights:
            leaves[leaf] = weights[key]
    return leaves


def _quant_status(leaves: Dict[str, Any]) -> str:
    if not leaves:
        return "missing"
    return "quantized" if "scales" in leaves else "dense"


def _output_axis(weight: Any) -> int:
    # Linear: (out, in). SwitchLinear: (experts, out, in). Packed along last dim.
    return 1 if getattr(weight, "ndim", 0) >= 3 else 0


def _concat_proj_leaves(
    left: Dict[str, Any], right: Dict[str, Any], axis: int
) -> Dict[str, Any]:
    fused: Dict[str, Any] = {}
    for leaf in _PROJ_LEAVES:
        if leaf in left or leaf in right:
            if leaf not in left or leaf not in right:
                raise ValueError(
                    f"cannot concat projection leaf {leaf!r}: present on only one side"
                )
            fused[leaf] = mx.concatenate(
                [_as_mx(left[leaf]), _as_mx(right[leaf])], axis=axis
            )
    return fused


def _pop_proj(weights: Dict[str, Any], stem: str) -> None:
    for leaf in _PROJ_LEAVES:
        weights.pop(f"{stem}.{leaf}", None)


def _write_proj(weights: Dict[str, Any], stem: str, leaves: Dict[str, Any]) -> None:
    for leaf, value in leaves.items():
        weights[f"{stem}.{leaf}"] = value


def _fuse_proj_pair(
    weights: Dict[str, Any], left_stem: str, right_stem: str, dest_stem: str
) -> bool:
    """Concat two same-status projections along the output axis. Exact, no requant.

    Returns True when a fusion was written. Mixed quantization status is left
    untouched (never change a weight's quant status).
    """
    if any(f"{dest_stem}.{leaf}" in weights for leaf in _PROJ_LEAVES):
        if any(f"{left_stem}.{leaf}" in weights for leaf in _PROJ_LEAVES):
            # Dest already present (HF packed gate_up / already remapped).
            _pop_proj(weights, left_stem)
            _pop_proj(weights, right_stem)
        return False
    left = _proj_leaves(weights, left_stem)
    right = _proj_leaves(weights, right_stem)
    if not left or not right:
        return False
    left_status = _quant_status(left)
    right_status = _quant_status(right)
    if left_status != right_status or left_status == "missing":
        return False
    axis = _output_axis(left["weight"])
    _write_proj(weights, dest_stem, _concat_proj_leaves(left, right, axis))
    _pop_proj(weights, left_stem)
    _pop_proj(weights, right_stem)
    return True


def _rewrite_packed_experts(weights: Dict[str, Any]) -> None:
    """Keep HF packed experts.gate_up_proj and experts.down_proj as switch_mlp.* (no split).

    Remaps all quantization leaves (weight, scales, biases, bias) so prequantized
    sidecars load fully onto FusedSwitchGLU.
    """
    for key in list(weights):
        for proj in ("gate_up_proj", "down_proj"):
            if key.endswith(f".experts.{proj}") or key == f"experts.{proj}":
                dest = key[: -len(f"experts.{proj}")] + f"switch_mlp.{proj}.weight"
                weights[dest] = weights.pop(key)
                break
            matched = False
            for leaf in _PROJ_LEAVES:
                target_suffix = f".experts.{proj}.{leaf}"
                if key.endswith(target_suffix):
                    dest = key[: -len(target_suffix)] + f".switch_mlp.{proj}.{leaf}"
                    weights[dest] = weights.pop(key)
                    matched = True
                    break
                elif key == f"experts.{proj}.{leaf}":
                    dest = f"switch_mlp.{proj}.{leaf}"
                    weights[dest] = weights.pop(key)
                    matched = True
                    break
            if matched:
                break


def remap_fused_projections(weights: Dict[str, Any]) -> Dict[str, Any]:
    """Map old split GDN/MoE projection keys onto the fused module names.

    On-disk layout is unchanged; this is load-time surgery only. Quantized
    tensors are concatenated along the output axis so each row keeps its
    original groups (exact). Mixed-status pairs are not fused.
    """
    out = dict(weights)
    _rewrite_packed_experts(out)

    qkv_stems = [
        key[: -len(".weight")]
        for key in list(out)
        if key.endswith(".linear_attn.in_proj_qkv.weight")
    ]
    for qkv in qkv_stems:
        prefix = qkv[: -len("in_proj_qkv")]
        _fuse_proj_pair(out, qkv, prefix + "in_proj_z", prefix + "in_proj_qkvz")

    b_stems = [
        key[: -len(".weight")]
        for key in list(out)
        if key.endswith(".linear_attn.in_proj_b.weight")
    ]
    for b in b_stems:
        prefix = b[: -len("in_proj_b")]
        _fuse_proj_pair(out, b, prefix + "in_proj_a", prefix + "in_proj_ba")

    gate_stems = [
        key[: -len(".weight")]
        for key in list(out)
        if key.endswith(".switch_mlp.gate_proj.weight")
        or key.endswith(".shared_expert.gate_proj.weight")
    ]
    for gate in gate_stems:
        dest = gate[: -len("gate_proj")] + "gate_up_proj"
        _fuse_proj_pair(out, gate, gate[: -len("gate_proj")] + "up_proj", dest)

    return out


_QWEN4_TRUNK_NORM_SUFFIXES = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
    "norm.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
)


def _shift_trunk_gemma_norms(weights: Dict[str, Any]) -> Dict[str, Any]:
    """Restore MLX-absolute RMSNorm gains on raw HF qwen4_exp trunk weights."""
    targets = [
        (k, v)
        for k, v in weights.items()
        if not k.startswith("mtp.")
        and getattr(v, "ndim", None) == 1
        and any(str(k).endswith(suffix) for suffix in _QWEN4_TRUNK_NORM_SUFFIXES)
    ]
    if not targets:
        return weights
    try:
        means = [float(v.mean().item()) for _, v in targets]
        if sum(means) / len(means) > 0.5:
            return weights
    except Exception:
        return weights

    out = dict(weights)
    for k, v in targets:
        out[k] = v + 1.0
    return out


def sanitize(weights: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize checkpoint weights to match the MTPLX module hierarchy.

    Key mappings performed:
      - Strips `model.language_model.` and `language_model.` prefixes from Hugging Face checkpoints.
      - Discards vision tower weights (`vision_tower.` / `model.visual.` / `visual.`).
      - Transposes 1D conv weights `(C, 1, K)` -> `(C, K, 1)` for MLX.
      - Stacks numbered MoE expert tensors into `switch_mlp` if unstacked.
      - Concatenates sharded n-gram embedding tables into ``shard_0`` (post-load;
        on-disk layout is unchanged).
      - Converts raw zero-centered Gemma trunk norms to MLX absolute convention.
      - Fuses GDN in_proj_qkv+z / in_proj_b+a and MoE gate+up at load time.
      - Retains all `mtp.*` tensors for downstream speculative/MTP lanes.
    """
    out: Dict[str, Any] = {}
    for k, v in weights.items():
        if k.startswith("model.language_model."):
            suffix = k[len("model.language_model.") :]
            if (
                suffix.startswith("model.")
                or suffix.startswith("lm_head.")
                or suffix.startswith("mtp.")
            ):
                k = suffix
            else:
                k = "model." + suffix
        elif k.startswith("language_model."):
            suffix = k[len("language_model.") :]
            if (
                suffix.startswith("model.")
                or suffix.startswith("lm_head.")
                or suffix.startswith("mtp.")
            ):
                k = suffix
            else:
                k = "model." + suffix
        if (
            k.startswith("vision_tower.")
            or k.startswith("model.visual.")
            or k.startswith("visual.")
        ):
            continue
        if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
            if v.shape[1] == 1:
                v = v.transpose(0, 2, 1)
        out[k] = v

    # Check for unstacked numbered experts: model.layers.{l}.mlp.experts.{e}.{m}.{k}
    expert_keys = [k for k in out if ".mlp.experts." in k and not k.startswith("mtp.")]
    if expert_keys:
        layer_indices = set()
        for k in expert_keys:
            parts = k.split(".")
            if "layers" in parts:
                idx = parts[parts.index("layers") + 1]
                if idx.isdigit():
                    layer_indices.add(int(idx))
        for l in sorted(layer_indices):
            prefix = f"model.layers.{l}.mlp"
            for m in ("gate_proj", "down_proj", "up_proj", "gate_up_proj"):
                for leaf in ("weight", "scales", "biases"):
                    sample_key = f"{prefix}.experts.0.{m}.{leaf}"
                    if sample_key in out:
                        e = 0
                        stacked = []
                        while f"{prefix}.experts.{e}.{m}.{leaf}" in out:
                            stacked.append(out.pop(f"{prefix}.experts.{e}.{m}.{leaf}"))
                            e += 1
                        if stacked:
                            dest_m = "gate_up_proj" if m == "gate_up_proj" else m
                            out[f"{prefix}.switch_mlp.{dest_m}.{leaf}"] = mx.stack(
                                stacked
                            )
                    elif f"{prefix}.experts.{m}.{leaf}" in out:
                        dest_m = "gate_up_proj" if m == "gate_up_proj" else m
                        out[f"{prefix}.switch_mlp.{dest_m}.{leaf}"] = out.pop(
                            f"{prefix}.experts.{m}.{leaf}"
                        )

    _concat_sharded_embedding_tables(out)
    out = _shift_trunk_gemma_norms(out)
    return remap_fused_projections(out)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )
        self._mtp_weights: Dict[str, Any] = {}

    @property
    def mtp_weights(self) -> Dict[str, Any]:
        return getattr(self, "_mtp_weights", {})

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        emit_logits: bool = True,
        logits_keep: Optional[int] = None,
        return_hidden: bool = False,
        **kwargs: Any,
    ) -> Union[mx.array, Tuple[Optional[mx.array], mx.array], None]:
        result = self.model(
            inputs,
            cache,
            input_embeddings,
            return_hyper=bool(return_hidden),
        )
        if return_hidden:
            mixed, hyper = result
        else:
            mixed = result
            hyper = mixed
        if not emit_logits:
            return (None, hyper) if return_hidden else None

        out = mixed
        if logits_keep is not None:
            out = out[:, -max(1, int(logits_keep)) :, :]

        if self.args.text.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)

        return (logits, hyper) if return_hidden else logits

    @property
    def layers(self) -> List[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> List[Any]:
        caches = []
        for i, t in enumerate(self.args.text.layer_types):
            if t == "full_attention":
                caches.append(_AttnCache())
            else:
                # 0: deltanet conv, 1: ssm state, 2: PLE conv, 3: n-gram context
                caches.append(ArraysCache(4))
        return caches

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = sanitize(weights)
        # Retain MTP tensors accessible via private side-channel for MTP lane,
        # but exclude them from strict trunk loading and module parameter tree.
        self._mtp_weights = {
            k: v for k, v in sanitized.items() if k.startswith("mtp.")
        }
        return {k: v for k, v in sanitized.items() if not k.startswith("mtp.")}

    @property
    def quant_predicate(self) -> Callable[[str, Any], bool]:
        def fn(path: str, module: Any) -> bool:
            # Keep MoE router in full precision; norms and convs are skipped anyway
            return not path.endswith("mlp.gate")

        return fn
