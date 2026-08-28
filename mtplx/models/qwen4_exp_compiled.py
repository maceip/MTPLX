# Copyright © 2024 Apple Inc. / MTPLX authors
# Native MLX compilation wrappers for Qwen3.8-Flash-Next (qwen4_exp).
#
# Stage 1: Shape-stable single-token decode subgraphs (GDN, GatedResidual, MoE).
# Keeps layer class bodies in qwen4_exp.py untouched to avoid merge conflicts
# with parallel lanes.

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


def is_compile_enabled() -> bool:
    """Check if the compiled decode path is enabled via environment variable."""
    return (
        str(os.environ.get("MTPLX_QWEN4EXP_COMPILE", "")).strip().lower()
        in {"1", "true", "yes", "on"}
        or str(os.environ.get("MTPLX_COMPILE_AR_FORWARD", "")).strip().lower()
        in {"1", "true", "yes", "on"}
    )


# ---------------------------------------------------------------------------
# Trace-safe carrier for GatedDeltaNet state
# ---------------------------------------------------------------------------


class GDNCarrier:
    """Throwaway carrier holding GDN conv and recurrent states during graph tracing.

    Constructed inside the compiled wrapper; object creation during tracing is
    transparent to MLX tracing while array inputs and outputs are recorded.
    """

    def __init__(self, conv_state: mx.array, rec_state: mx.array):
        self.conv_state = conv_state
        self.rec_state = rec_state

    def __getitem__(self, idx: int) -> mx.array:
        if idx == 0:
            return self.conv_state
        elif idx == 1:
            return self.rec_state
        raise IndexError(f"GDNCarrier index out of range: {idx}")

    def __setitem__(self, idx: int, val: mx.array) -> None:
        if idx == 0:
            self.conv_state = val
        elif idx == 1:
            self.rec_state = val
        else:
            raise IndexError(f"GDNCarrier index out of range: {idx}")

    def advance(self, S: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Stage 1: Individual Subgraph Compilers
# ---------------------------------------------------------------------------


def compile_gdn_step(
    layer: Any,
) -> Callable[[mx.array, mx.array, mx.array], Tuple[mx.array, mx.array, mx.array]]:
    """Compile single-token GatedDeltaNet step (S=1).

    Args:
        layer: GatedDeltaNet module instance.

    Returns:
        Compiled function taking (x, conv_state, recurrent_state) and returning
        (out, new_conv_state, new_recurrent_state).
    """

    def gdn_step(
        x: mx.array, conv_state: mx.array, rec_state: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        carrier = GDNCarrier(conv_state, rec_state)
        out = layer(x, None, carrier)
        return out, carrier.conv_state, carrier.rec_state

    return mx.compile(gdn_step)


def compile_gated_residual(
    hc_layer: Any,
) -> Callable[[mx.array], Union[Tuple[mx.array, mx.array, mx.array], mx.array]]:
    """Compile GatedResidual hyper-connection mixing/injection step.

    Args:
        hc_layer: GatedResidual module instance.

    Returns:
        Compiled function taking (h) and returning (mixed, hyper, inject) or mixed.
    """

    def hc_step(h: mx.array):
        return hc_layer(h)

    return mx.compile(hc_step)


def compile_moe_step(moe_layer: Any) -> Callable[[mx.array], mx.array]:
    """Compile single-token SparseMoeBlock (S=1).

    Args:
        moe_layer: SparseMoeBlock module instance.

    Returns:
        Compiled function taking (x) and returning out.
    """

    def moe_step(x: mx.array) -> mx.array:
        return moe_layer(x)

    return mx.compile(moe_step)


def compile_linear_layer_step(
    layer: Any,
) -> Callable[[mx.array, Any, mx.array, mx.array], Tuple[mx.array, mx.array, mx.array]]:
    """Compile full single-token step for a linear attention decoder layer.

    Fuses attn_hyper_connection -> GatedDeltaNet -> residual ->
    mlp_hyper_connection -> SparseMoeBlock -> residual into a single
    compiled Metal execution graph.

    Args:
        layer: DecoderLayer instance where layer_type == "linear_attention".

    Returns:
        Compiled function taking (h, conv_mask, conv_state, rec_state) and returning
        (h_out, new_conv_state, new_rec_state).
    """

    def linear_step(
        h: mx.array, conv_mask: Any, conv_state: mx.array, rec_state: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        x, hyper, inject = layer.attn_hyper_connection(h)
        carrier = GDNCarrier(conv_state, rec_state)
        x = layer.linear_attn(x, conv_mask, carrier)
        h_mid = hyper + (x[..., None, :] * inject[..., None]).reshape(
            *x.shape[:-1], -1
        )
        x_mlp, hyper_mlp, inject_mlp = layer.mlp_hyper_connection(h_mid)
        x_moe = layer.mlp(x_mlp)
        h_out = hyper_mlp + (x_moe[..., None, :] * inject_mlp[..., None]).reshape(
            *x_moe.shape[:-1], -1
        )
        return h_out, carrier.conv_state, carrier.rec_state

    return mx.compile(linear_step)


def compile_attn_pre_step(
    layer: Any,
) -> Callable[[mx.array], Tuple[mx.array, mx.array, mx.array]]:
    """Compile pre-attention hyper-connection for a full attention layer."""

    def attn_pre_step(h: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        return layer.attn_hyper_connection(h)

    return mx.compile(attn_pre_step)


def compile_attn_post_step(
    layer: Any,
) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    """Compile post-attention residual, MLP hyper-connection, MoE, and final residual."""

    def attn_post_step(x: mx.array, hyper: mx.array, inject: mx.array) -> mx.array:
        h_mid = hyper + (x[..., None, :] * inject[..., None]).reshape(
            *x.shape[:-1], -1
        )
        x_mlp, hyper_mlp, inject_mlp = layer.mlp_hyper_connection(h_mid)
        x_moe = layer.mlp(x_mlp)
        return hyper_mlp + (x_moe[..., None, :] * inject_mlp[..., None]).reshape(
            *x_moe.shape[:-1], -1
        )

    return mx.compile(attn_post_step)


def compile_hyper_mixer(mixer: Any) -> Callable[[mx.array], mx.array]:
    """Compile final hyper-connection mixer."""

    def mixer_step(h: mx.array) -> mx.array:
        return mixer(h)

    return mx.compile(mixer_step)


# ---------------------------------------------------------------------------
# Compiled Model Execution Manager
# ---------------------------------------------------------------------------


class CompiledLayerRunner:
    """Manages compiled step functions across model layers for a model instance."""

    def __init__(self, model: Any):
        self.model = model
        self.layer_runners: List[Dict[str, Any]] = []

        for layer in model.layers:
            if layer.layer_type == "linear_attention":
                self.layer_runners.append(
                    {
                        "type": "linear",
                        "step": compile_linear_layer_step(layer),
                        "gdn": layer.linear_attn,
                        "layer": layer,
                    }
                )
            else:
                self.layer_runners.append(
                    {
                        "type": "full_attention",
                        "pre": compile_attn_pre_step(layer),
                        "post": compile_attn_post_step(layer),
                        "layer": layer,
                    }
                )

        self.mixer_step = compile_hyper_mixer(model.hyper_connection_mixer)

    def run(
        self,
        h: mx.array,
        rope: Any,
        mask: Any,
        conv_mask: Any,
        cache: Optional[List[Any]],
        ids: Optional[mx.array],
        prev_ctx: Optional[mx.array],
    ) -> mx.array:
        B = h.shape[0]
        if cache is None:
            cache = [None] * len(self.layer_runners)

        for i, (runner, c) in enumerate(zip(self.layer_runners, cache)):
            layer = runner["layer"]

            # 1. PLE step: stays eager because of NumPy host-sync in _ShardedEmbedding.
            # INTEGRATION POINT (Stage 2): When Lane A/B replaces numpy indexing with
            # purely native MLX ops, ple can be absorbed into the compiled graph.
            if layer.ple is not None:
                h = h + layer.ple(h, ids, prev_ctx, c, conv_mask)

            if runner["type"] == "linear":
                gdn = runner["gdn"]
                # Zero-initialized states for shape stability (no None retrace hazards)
                conv_state = (
                    c[0]
                    if (c is not None and c[0] is not None)
                    else mx.zeros(
                        (B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=h.dtype
                    )
                )
                rec_state = (
                    c[1]
                    if (c is not None and c[1] is not None)
                    else mx.zeros(
                        (B, gdn.n_v, gdn.dv, gdn.dk), dtype=mx.float32
                    )
                )

                h, new_c, new_r = runner["step"](h, conv_mask, conv_state, rec_state)

                if c is not None:
                    c[0] = new_c
                    c[1] = new_r
                    if hasattr(c, "advance"):
                        c.advance(1)

            else:
                # Full attention layer: attention stays eager (offset / KVCache / rotary)
                x_in, hyper, inject = runner["pre"](h)
                if getattr(layer.self_attn, "rope", None) is None:
                    layer.self_attn.rope = rope
                x_out = layer.self_attn(x_in, mask, c)
                h = runner["post"](x_out, hyper, inject)

        return h


# Cache runner instance directly on the model instance to avoid global
# dictionary retention and memory leaks across reload cycles.
def clear_compiled_runner_cache(model: Any | None = None) -> None:
    """Explicitly drop cached compiled runners to free memory upon model unload."""
    if model is not None and hasattr(model, "_compiled_runner"):
        try:
            delattr(model, "_compiled_runner")
        except Exception:
            pass


def get_compiled_runner(model: Any) -> CompiledLayerRunner:
    runner = getattr(model, "_compiled_runner", None)
    if runner is None or getattr(runner, "model", None) is not model:
        runner = CompiledLayerRunner(model)
        try:
            model._compiled_runner = runner
        except Exception:
            pass
    return runner


def run_compiled_layers(
    model: Any,
    h: mx.array,
    rope: Any,
    mask: Any,
    conv_mask: Any,
    cache: Optional[List[Any]],
    ids: Optional[mx.array],
    prev_ctx: Optional[mx.array],
) -> mx.array:
    """Entry point called from Qwen4ExpModel.__call__ when compiled path is active."""
    runner = get_compiled_runner(model)
    return runner.run(h, rope, mask, conv_mask, cache, ids, prev_ctx)
