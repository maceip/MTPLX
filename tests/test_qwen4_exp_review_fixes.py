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


def test_sparse_index_reuse_reset_on_rollback_and_cycle_start():
    from mtplx.generation import _rollback_mtp_cache
    from mtplx.qwen4_exp_mtp_patch import (
        SparseIndexReuse,
        _install_indexer_aware_trim,
    )

    class DummyIndexer:
        def __init__(self):
            self.token_budget = 2048
            self.n_heads = 4
            self.head_dim = 128

        def __call__(self, x, rope, cache, offset: int):
            return mx.zeros((1, 1, 1))

    dummy_indexer = DummyIndexer()
    wrapper = SparseIndexReuse(dummy_indexer)
    wrapper.prev_keep = mx.zeros((1, 4, 128))

    attn_cache = _AttnCache()
    attn_cache.indexer = _IndexerCache()
    attn_cache.indexer.update(mx.ones((1, 8, 128)))
    attn_cache._sparse_index_reuse = wrapper
    _install_indexer_aware_trim(attn_cache)

    assert wrapper.prev_keep is not None
    # 1. trim() resets prev_keep
    attn_cache.trim(2)
    assert wrapper.prev_keep is None

    # 2. _rollback_mtp_cache resets prev_keep
    wrapper.prev_keep = mx.zeros((1, 4, 128))
    _rollback_mtp_cache([attn_cache], 0)
    assert wrapper.prev_keep is None


def test_flat_qwen4_exp_config_preserves_geometry():
    flat_cfg = {
        "model_type": "qwen4_exp",
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 4,
    }
    args = ModelArgs.from_dict(flat_cfg)
    assert args.text.hidden_size == 1024
    assert args.text.num_hidden_layers == 16
    assert args.text.num_attention_heads == 8
    assert args.text.num_key_value_heads == 2
    assert args.text.head_dim == 128
    assert args.text.num_experts == 64
    assert args.text.num_experts_per_tok == 4


def test_ple_state_preserved_with_recurrent_cache():
    from mtplx.models.qwen4_exp import PLELayer, TextArgs
    from mtplx.qwen4_exp_mtp_patch import Qwen4ExpRecurrentCache

    args = TextArgs(
        hidden_size=256,
        ple_embed_dim=256,
        ple_layer_ids=[1],
        hc_count=2,
        ngram_size=3,
        ple_conv_kernel_size=4,
    )
    ple = PLELayer(args, ple_layer_index=0)
    recurrent_cache = Qwen4ExpRecurrentCache(4)
    assert recurrent_cache[2] is None
    assert len(recurrent_cache) == 4

    x = mx.ones((1, 1, 256 * 2))
    out = ple._short_conv(x, recurrent_cache)
    assert out is not None
    assert recurrent_cache[2] is not None
    assert recurrent_cache[2].shape == (1, 9, 256 * 2)


def test_mtp_module_quantization_from_contract(tmp_path):
    from types import SimpleNamespace
    import mlx.nn as nn
    from mtplx.models.qwen4_exp import TextArgs
    from mtplx.mtp_patch import MTPContract
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "mtp_num_hidden_layers": 1,
        "hc_count": 2,
        "hc_lowrank": 128,
    }

    class DummyLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                layers=[
                    SimpleNamespace(layer_type="linear_attention"),
                    SimpleNamespace(layer_type="linear_attention"),
                    SimpleNamespace(layer_type="linear_attention"),
                    SimpleNamespace(layer_type="full_attention"),
                ],
                embed_tokens=nn.Embedding(100, 256),
            )
            self.args = TextArgs(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                hc_count=2,
                hc_lowrank=128,
                rope_parameters={},
                rope_theta=10000.0,
                partial_rotary_factor=0.25,
                layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                full_attention_interval=4,
            )

        def make_cache(self):
            return [None] * 4

    model = DummyLanguageModel()
    contract = MTPContract(
        mtp_quant_bits=4,
        mtp_quant_group_size=64,
        mtp_quant_policy="all",
    )

    success = inject_qwen4_exp_mtp_support(
        model,
        str(tmp_path),
        config,
        contract=contract,
        allow_random_init=True,
    )
    assert success is True
    assert hasattr(model, "mtp")
    # Check that linear layers in MTP are quantized
    quantized_layers = [
        m for _, m in model.mtp.named_modules() if isinstance(m, nn.QuantizedLinear)
    ]
    assert len(quantized_layers) > 0


def test_qsa_gather_attention_applies_array_mask():
    from mtplx.models.qwen4_exp import QSASelection, _qsa_gather_attention

    B, H, S, D = 1, 2, 2, 64
    T = 4
    K = 3
    q = mx.ones((B, H, S, D))
    k = mx.ones((B, H, T, D))
    v = mx.ones((B, H, T, D))

    token_idx = mx.array([[[0, 1, 2], [1, 2, 3]]])  # (1, 2, 3)
    valid = mx.array([[[True, True, True], [True, True, True]]])
    sel = QSASelection(token_idx=token_idx, valid=valid)

    # Boolean mask: key 0 and 1 are masked out (False = padding)
    # mask shape: (B, 1, S, T)
    mask = mx.array([[[[False, False, True, True], [False, False, True, True]]]])
    out = _qsa_gather_attention(q, k, v, sel, scale=1.0, mask=mask)
    assert out.shape == (B, H, S, D)

    # Floating additive mask: key 0 is -1e9
    mask_float = mx.array([[[[-1e9, 0.0, 0.0, 0.0], [-1e9, 0.0, 0.0, 0.0]]]])
    out_float = _qsa_gather_attention(q, k, v, sel, scale=1.0, mask=mask_float)
    assert out_float.shape == (B, H, S, D)


def test_qwen4_exp_mtp_gate_requires_mtp_tensors(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"dummy")

    # Config declares mtp_num_hidden_layers > 0, but no MTP weights exist and tensor_gate=False
    inspection_no_mtp_weights = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=str(tmp_path),
        mtp_num_hidden_layers=1,
        weight_keys=("model.embed_tokens.weight", "model.layers.0.mlp.gate_up_proj.weight"),
    )
    assert _passes_qwen4_exp_mtp_gate(inspection_no_mtp_weights, tensor_gate=False) is False

    # When tensor_gate=True or MTP weights exist -> passes
    inspection_with_mtp = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=str(tmp_path),
        mtp_num_hidden_layers=1,
        weight_keys=("mtp.layers.0.self_attn.q_proj.weight",),
    )
    assert _passes_qwen4_exp_mtp_gate(inspection_with_mtp, tensor_gate=False) is True
    assert _passes_qwen4_exp_mtp_gate(inspection_no_mtp_weights, tensor_gate=True) is True


def test_mtp_forward_routes_through_installed_draft_head(tmp_path):
    import mlx.nn as nn
    from mtplx.models.qwen4_exp import TextArgs
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "mtp_num_hidden_layers": 1,
        "hc_count": 2,
        "hc_lowrank": 64,
    }

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                layers=[
                    SimpleNamespace(layer_type="linear_attention"),
                    SimpleNamespace(layer_type="linear_attention"),
                    SimpleNamespace(layer_type="linear_attention"),
                    SimpleNamespace(layer_type="full_attention"),
                ],
                embed_tokens=nn.Embedding(100, 128),
            )
            self.lm_head = nn.Linear(128, 100, bias=False)
            self.args = TextArgs(
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                hc_count=2,
                hc_lowrank=64,
                rope_parameters={},
                rope_theta=10000.0,
                partial_rotary_factor=0.25,
                layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                full_attention_interval=4,
            )

        def make_cache(self):
            return [None] * 4

    model = DummyModel()
    inject_qwen4_exp_mtp_support(model, str(tmp_path), config, allow_random_init=True)

    # Attach custom draft head returning a distinct constant
    class CustomDraftHead(nn.Module):
        def __call__(self, x):
            return mx.full((*x.shape[:-1], 100), 42.0)

    model._mtplx_draft_lm_head = CustomDraftHead()

    h = mx.zeros((1, 1, 128 * 2))
    toks = mx.array([[0]], dtype=mx.int32)
    logits = model.mtp_forward(h, toks)
    assert float(logits[0, 0, 0].item()) == 42.0


def test_forward_logits_controls_through_qwen4_wrapper(tmp_path):
    import mlx.nn as nn
    from mtplx.models.qwen4_exp import Model, ModelArgs, TextArgs
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "mtp_num_hidden_layers": 1,
        "hc_count": 2,
        "hc_lowrank": 64,
        "vocab_size": 50,
    }

    model = Model(ModelArgs.from_dict(config))
    inject_qwen4_exp_mtp_support(model, str(tmp_path), config, allow_random_init=True)

    toks = mx.array([[1, 2, 3, 4]])
    cache = model.make_cache()

    # 1. emit_logits=False
    out = model(toks, cache=cache, emit_logits=False)
    assert out is None

    out_h = model(toks, cache=cache, emit_logits=False, return_hidden=True)
    assert out_h[0] is None
    assert out_h[1] is not None

    # 2. logits_keep=1
    out_k1 = model(toks, cache=cache, logits_keep=1)
    assert out_k1 is not None
    assert out_k1.shape == (1, 1, 50)

    out_k1_h = model(toks, cache=cache, logits_keep=1, return_hidden=True)
    assert out_k1_h[0] is not None
    assert out_k1_h[0].shape == (1, 1, 50)


def test_recurrent_mask_built_for_padded_qwen4_batches():
    from mtplx.models.qwen4_exp import Model, ModelArgs
    from mtplx.qwen4_exp_mtp_patch import Qwen4ExpRecurrentCache, wrap_qwen4_exp_cache

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 50,
    }

    model = Model(ModelArgs.from_dict(config))
    cache = model.make_cache()
    # Install left padding on recurrent cache (batch of 2 sequences, first has 2 pad tokens)
    left_padding = mx.array([2, 0])
    wrapped_cache = wrap_qwen4_exp_cache(cache, model.model)
    for c in wrapped_cache:
        if isinstance(c, Qwen4ExpRecurrentCache):
            c.left_padding = left_padding

    toks = mx.array([[0, 0, 5, 6], [7, 8, 9, 10]])
    logits = model(toks, cache=wrapped_cache)
    assert logits is not None
    assert logits.shape == (2, 4, 50)


def test_avoid_double_shifting_mlx_mtp_norms(tmp_path):
    from mtplx.qwen4_exp_mtp_patch import _load_mtp_weights, _shift_qwen4_gemma_mtp_norms

    # 1. Delta-encoded norm weights (mean around 0.0) -> should be shifted (+1.0)
    delta_weights = {
        "pre_fc_norm_embedding.weight": mx.zeros((128,)),
        "pre_fc_norm_hidden.weight": mx.zeros((128,)),
        "layers.0.self_attn.q_norm.weight": mx.zeros((32,)),
    }
    shifted = _shift_qwen4_gemma_mtp_norms(delta_weights)
    assert float(shifted["pre_fc_norm_embedding.weight"].mean().item()) == 1.0
    assert float(shifted["pre_fc_norm_hidden.weight"].mean().item()) == 1.0

    # 2. Already shifted / absolute norm weights (mean around 1.0) -> should NOT be shifted
    already_shifted = {
        "pre_fc_norm_embedding.weight": mx.ones((128,)),
        "pre_fc_norm_hidden.weight": mx.ones((128,)),
        "layers.0.self_attn.q_norm.weight": mx.ones((32,)),
    }
    not_double_shifted = _shift_qwen4_gemma_mtp_norms(already_shifted)
    assert float(not_double_shifted["pre_fc_norm_embedding.weight"].mean().item()) == 1.0

    # 3. Safetensors file saved with format='mlx' metadata -> should NOT be shifted
    sf_path = tmp_path / "mtp.safetensors"
    mx.save_safetensors(
        str(sf_path),
        {
            "mtp.pre_fc_norm_embedding.weight": mx.ones((128,)),
            "mtp.pre_fc_norm_hidden.weight": mx.ones((128,)),
        },
        metadata={"format": "mlx"},
    )
    loaded = _load_mtp_weights([sf_path])
    assert float(loaded["pre_fc_norm_embedding.weight"].mean().item()) == 1.0



