from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from mtplx.backends.descriptors import (
    MLX_LM_AR_DESCRIPTOR,
    descriptor_for_architecture_id,
    descriptor_for_backend_id,
    descriptor_from_inspection,
)
from mtplx.backends.registry import (
    _passes_family_runtime_gate,
    _passes_qwen4_exp_gate,
    _passes_qwen4_exp_mtp_gate,
)
from mtplx.cache_state import (
    BlockOwnedKVCache,
    TailOwnedKVCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
)
from mtplx.models.qwen4_exp import Model, ModelArgs, _AttnCache, _IndexerCache
from mtplx.server import openai


def test_qsa_indexer_preserved_during_cache_repaging():
    attn_cache = _AttnCache()
    attn_cache.indexer.update(mx.ones((1, 8, 128)))
    assert attn_cache.indexer.keys is not None
    assert attn_cache.indexer.keys.shape == (1, 8, 128)

    # 1. VllmMetalPagedKVCache
    paged = VllmMetalPagedKVCache.from_cache(attn_cache, block_size=16, num_blocks=4)
    assert hasattr(paged, "indexer")
    assert paged.indexer is attn_cache.indexer
    assert paged.indexer.keys.shape == (1, 8, 128)

    # 2. TailOwnedKVCache
    tail = TailOwnedKVCache.from_cache(attn_cache, mode="eval_only")
    assert hasattr(tail, "indexer")
    assert tail.indexer is attn_cache.indexer

    # 3. BlockOwnedKVCache
    block = BlockOwnedKVCache.from_cache(attn_cache, mode="eval_only")
    assert hasattr(block, "indexer")
    assert block.indexer is attn_cache.indexer

    # 4. Promotion & Demotion
    paged.key_cache = mx.zeros((4, 16, 2, 64))
    paged.value_cache = mx.zeros((4, 16, 2, 64))
    paged.offset = 8
    promoted = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    assert hasattr(promoted, "indexer")
    assert promoted.indexer is attn_cache.indexer

    demoted = promoted.to_paged_cache()
    assert hasattr(demoted, "indexer")
    assert demoted.indexer is attn_cache.indexer

    # 5. Trim indexer keys on paged cache
    paged.trim(2)
    assert paged.indexer.keys.shape == (1, 6, 128)


def test_quant_predicate_accepts_two_arguments():
    model = Model(ModelArgs())
    pred = model.quant_predicate
    assert callable(pred)
    # Must accept 2 arguments without error
    assert pred("model.layers.0.mlp.gate", None) is False
    assert pred("model.layers.0.mlp.down_proj", None) is True


def test_sanitize_excludes_mtp_tensors_from_trunk_dict():
    model = Model(ModelArgs())
    weights = {
        "model.embed_tokens.weight": mx.zeros((10, 10)),
        "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
        "mtp.layers.0.mlp.gate_proj.weight": mx.zeros((10, 10)),
    }
    sanitized = model.sanitize(weights)
    assert "mtp.layers.0.self_attn.q_proj.weight" not in sanitized
    assert "mtp.layers.0.mlp.gate_proj.weight" not in sanitized
    assert "model.embed_tokens.weight" in sanitized
    assert "mtp.layers.0.self_attn.q_proj.weight" in model.mtp_weights


def test_qwen4_exp_mtp_family_runtime_gate(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"dummy")

    inspection = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=str(tmp_path),
        mtp_num_hidden_layers=1,
        weight_keys=("mtp.layers.0.self_attn.q_proj.weight",),
    )

    assert _passes_qwen4_exp_mtp_gate(inspection, tensor_gate=True) is True
    assert _passes_family_runtime_gate("qwen4-exp-mtp", inspection, tensor_gate=True) is True
    assert _passes_family_runtime_gate("qwen4-exp", inspection, tensor_gate=False) is True


def test_qwen4_exp_backend_descriptor_registered():
    assert descriptor_for_backend_id("qwen4_exp") == MLX_LM_AR_DESCRIPTOR
    assert descriptor_for_architecture_id("qwen4-exp") == MLX_LM_AR_DESCRIPTOR
    assert (
        descriptor_from_inspection({"recommended_backend": "qwen4_exp"})
        == MLX_LM_AR_DESCRIPTOR
    )


def test_select_backend_context_window_clamps_to_model_max():
    desc = MLX_LM_AR_DESCRIPTOR

    # 32K checkpoint requested with 1M context must clamp to 32,768
    result = openai._select_backend_context_window(
        desc,
        model_max=32_768,
        requested=1_048_576,
    )
    assert result == 32_768

    # 1M checkpoint requested with 1M context returns 1,048,576
    result = openai._select_backend_context_window(
        desc,
        model_max=1_048_576,
        requested=1_048_576,
    )
    assert result == 1_048_576


def test_report_actual_model_in_memory_cap_failures(monkeypatch):
    args = openai.parse_args(["--model", "custom-org/my-qwen-model", "--backend-id", "qwen4_exp"])

    monkeypatch.setattr(
        openai,
        "_apply_metal_memory_caps",
        lambda **kwargs: {
            "applied": False,
            "reason": "insufficient_ram",
            "minimum_resident_bytes": 100 * 1024**3,
        },
    )

    with pytest.raises(RuntimeError, match="custom-org/my-qwen-model cannot load inside"):
        openai.ServerState(args)


def test_fail_preflight_when_model_cannot_fit_active_caps(monkeypatch):
    monkeypatch.setattr(
        openai,
        "_apply_metal_memory_caps",
        lambda **kwargs: {
            "applied": False,
            "reason": "insufficient_ram",
            "minimum_resident_bytes": 128 * 1024**3,
        },
    )

    with pytest.raises(RuntimeError, match="cannot load inside the available Metal memory budget"):
        openai.apply_memory_caps_preflight(
            entry="benchmark",
            model="Qwen/Qwen3.8-Flash-Next",
            contexts=[4096],
        )


def test_qwen4_exp_ar_gate_requires_model_shards(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    inspection_empty = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=str(empty_dir),
    )
    assert _passes_qwen4_exp_gate(inspection_empty) is False

    # Only MTP shard present (no trunk shard) -> must fail AR gate
    mtp_only_dir = tmp_path / "mtp_only"
    mtp_only_dir.mkdir()
    (mtp_only_dir / "mtp.safetensors").write_bytes(b"dummy")
    inspection_mtp_only = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=str(mtp_only_dir),
    )
    assert _passes_qwen4_exp_gate(inspection_mtp_only) is False

    # Trunk shard present -> passes
    trunk_dir = tmp_path / "trunk"
    trunk_dir.mkdir()
    (trunk_dir / "model.safetensors").write_bytes(b"dummy")
    inspection_trunk = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=str(trunk_dir),
    )
    assert _passes_qwen4_exp_gate(inspection_trunk) is True
