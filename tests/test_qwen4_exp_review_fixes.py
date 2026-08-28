from __future__ import annotations

import argparse
import math
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


def test_short_cache_and_custom_cache_slot_resilience():
    from mtplx.models.qwen4_exp import GatedDeltaNet, PLELayer, TextArgs

    args = TextArgs(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        hc_count=2,
        hc_lowrank=64,
        ple_embed_dim=128,
        ple_layer_ids=[1],
        ngram_size=3,
        ple_conv_kernel_size=4,
        layer_types=["linear_attention", "linear_attention"],
    )

    # 1. Test GatedDeltaNet with a plain 2-element list (no .advance method)
    gdn = GatedDeltaNet(args)
    plain_list_cache = [None, None]
    x = mx.ones((1, 1, 128))
    out = gdn(x, mask=None, cache=plain_list_cache)
    assert out is not None
    assert plain_list_cache[0] is not None
    assert plain_list_cache[1] is not None

    # 2. Test PLELayer with a 2-element list (shorter than index 2)
    ple = PLELayer(args, ple_layer_index=0)
    x_ple = mx.ones((1, 1, 256))
    out_ple = ple._short_conv(x_ple, plain_list_cache)
    assert out_ple is not None


def test_qsa_indexer_keys_preserved_in_cache_snapshots():
    from mtplx.cache_state import (
        BlockOwnedKVCache,
        TailOwnedKVCache,
        VllmMetalPagedKVCache,
        restore_cache,
        snapshot_cache,
        snapshot_cache_lazy_hybrid,
    )
    from mtplx.models.qwen4_exp import _AttnCache

    # 1. _AttnCache snapshot and restore
    attn_cache = _AttnCache()
    attn_cache.indexer.update(mx.ones((1, 8, 128)))
    attn_cache.keys = mx.zeros((1, 2, 8, 64))
    attn_cache.values = mx.zeros((1, 2, 8, 64))
    attn_cache.offset = 8

    snap = snapshot_cache([attn_cache])
    assert len(snap.states[0]) == 3
    assert snap.states[0][2] is not None
    assert snap.states[0][2].shape == (1, 8, 128)

    fresh_attn = _AttnCache()
    assert fresh_attn.indexer.keys is None
    restore_cache([fresh_attn], snap)
    assert fresh_attn.indexer.keys is not None
    assert fresh_attn.indexer.keys.shape == (1, 8, 128)
    assert fresh_attn.indexer._len == 8

    # 2. Lazy hybrid snapshot with _AttnCache
    lazy_snap = snapshot_cache_lazy_hybrid([attn_cache])
    fresh_lazy = _AttnCache()
    restore_cache([fresh_lazy], lazy_snap)
    assert fresh_lazy.indexer.keys is not None
    assert fresh_lazy.indexer.keys.shape == (1, 8, 128)

    # 3. Paged cache with indexer
    paged = VllmMetalPagedKVCache.from_cache(attn_cache, block_size=16, num_blocks=4)
    paged.key_cache = mx.zeros((4, 16, 2, 64))
    paged.value_cache = mx.zeros((4, 16, 2, 64))
    paged.offset = 8
    paged_snap = snapshot_cache([paged])
    assert len(paged_snap.states[0]) == 3
    assert paged_snap.states[0][2] is not None
    assert paged_snap.states[0][2].shape == (1, 8, 128)

    fresh_paged = VllmMetalPagedKVCache.from_cache(attn_cache, block_size=16, num_blocks=4)
    fresh_paged.indexer.keys = None
    restore_cache([fresh_paged], paged_snap)
    assert fresh_paged.indexer.keys is not None
    assert fresh_paged.indexer.keys.shape == (1, 8, 128)


def test_honor_qwen4_exp_declared_d1_default_in_server():
    from pydantic import BaseModel

    class DummyRequest(BaseModel):
        pass

    # 1. Unspecified depth: honors Qwen4 descriptor default (D1)
    args = openai.parse_args(["--model", "Qwen/Qwen3.8-Flash-Next"])
    assert args._explicit_depth is False

    state = SimpleNamespace(
        args=args,
        backend_descriptor=descriptor_for_backend_id("qwen4_exp"),
        runtime=SimpleNamespace(model=None),
    )
    req = DummyRequest()
    depth = openai._request_depth_for_generation(state, req, generation_mode="mtp")
    assert depth == 1

    # 2. Explicit --depth 2 on CLI: honors user flag
    args_explicit = openai.parse_args(["--model", "Qwen/Qwen3.8-Flash-Next", "--depth", "2"])
    assert args_explicit._explicit_depth is True

    state_explicit = SimpleNamespace(
        args=args_explicit,
        backend_descriptor=descriptor_for_backend_id("qwen4_exp"),
        runtime=SimpleNamespace(model=None),
    )
    depth_explicit = openai._request_depth_for_generation(state_explicit, req, generation_mode="mtp")
    assert depth_explicit == 2


def test_raw_hf_qwen4_exp_trunk_norm_gains_converted():
    from mtplx.models.qwen4_exp import sanitize

    # 1. Raw HF zero-centered Gemma norms (mean around 0.0) -> converted (+1.0)
    raw_hf_weights = {
        "model.embed_tokens.weight": mx.zeros((100, 128)),
        "model.layers.0.self_attn.q_norm.weight": mx.zeros((32,)),
        "model.layers.0.self_attn.k_norm.weight": mx.zeros((32,)),
        "model.layers.0.linear_attn.norm.weight": mx.zeros((128,)),
        "model.layers.0.attn_hyper_connection.hc_norm.weight": mx.zeros((256,)),
        "model.hyper_connection_mixer.hc_norm.weight": mx.zeros((256,)),
    }
    sanitized = sanitize(raw_hf_weights)
    assert float(sanitized["model.layers.0.self_attn.q_norm.weight"].mean().item()) == 1.0
    assert float(sanitized["model.layers.0.self_attn.k_norm.weight"].mean().item()) == 1.0
    assert float(sanitized["model.layers.0.linear_attn.norm.weight"].mean().item()) == 1.0
    assert float(sanitized["model.layers.0.attn_hyper_connection.hc_norm.weight"].mean().item()) == 1.0
    assert float(sanitized["model.hyper_connection_mixer.hc_norm.weight"].mean().item()) == 1.0

    # 2. Already converted MLX norms (mean around 1.0) -> not double-shifted
    mlx_weights = {
        "model.embed_tokens.weight": mx.zeros((100, 128)),
        "model.layers.0.self_attn.q_norm.weight": mx.ones((32,)),
        "model.layers.0.self_attn.k_norm.weight": mx.ones((32,)),
        "model.layers.0.linear_attn.norm.weight": mx.ones((128,)),
    }
    sanitized_mlx = sanitize(mlx_weights)
    assert float(sanitized_mlx["model.layers.0.self_attn.q_norm.weight"].mean().item()) == 1.0
    assert float(sanitized_mlx["model.layers.0.linear_attn.norm.weight"].mean().item()) == 1.0


def test_rope_scaling_normalized_into_rope_parameters():
    from mtplx.models.qwen4_exp import Model, ModelArgs, TextArgs

    # 1. Config using rope_scaling dictionary with type="yarn"
    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 50,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    }

    args = ModelArgs.from_dict(config)
    assert args.text.rope_parameters is not None
    assert args.text.rope_parameters.get("rope_type") == "yarn"
    assert float(args.text.rope_parameters.get("factor")) == 4.0

    model = Model(args)
    # Model's rotary embedding must have applied Yarn scaling (mscale > 1.0 for factor=4.0)
    assert model.model.rope.mscale > 1.0
    assert abs(model.model.rope.mscale - (0.1 * math.log(4.0) + 1.0)) < 1e-4


def test_qwen4_exp_recurrent_cache_batch_protocol():
    from mtplx.qwen4_exp_mtp_patch import Qwen4ExpRecurrentCache

    # 1. Initialization and batch_size
    c1 = Qwen4ExpRecurrentCache(4)
    c1[0] = mx.ones((2, 3, 128))
    c1[1] = mx.ones((2, 4, 32, 32))
    c1.left_padding = mx.array([2, 0])
    c1.lengths = mx.array([8, 10])
    assert c1.batch_size == 2

    # 2. Extract
    extracted = c1.extract(1)
    assert extracted.batch_size == 1
    assert extracted[0].shape == (1, 3, 128)
    assert extracted.left_padding[0].item() == 0
    assert extracted.lengths[0].item() == 10

    # 3. Filter
    c1.filter(mx.array([1]))
    assert c1.batch_size == 1
    assert c1[0].shape == (1, 3, 128)
    assert c1.left_padding[0].item() == 0
    assert c1.lengths[0].item() == 10

    # 4. Extend
    c2 = Qwen4ExpRecurrentCache(4)
    c2[0] = mx.full((1, 3, 128), 2.0)
    c2[1] = mx.full((1, 4, 32, 32), 2.0)
    c2.left_padding = mx.array([1])
    c2.lengths = mx.array([6])

    c1.extend(c2)
    assert c1.batch_size == 2
    assert c1[0].shape == (2, 3, 128)
    assert c1.left_padding.tolist() == [0, 1]
    assert c1.lengths.tolist() == [10, 6]

    # 5. Prepare & Finalize
    c1.prepare(lengths=[5, 5], left_padding=[0, 0])
    assert c1.lengths.tolist() == [5, 5]
    assert c1.left_padding.tolist() == [0, 0]
    c1.finalize()
    assert c1.lengths is None
    assert c1.left_padding is None


def test_compiled_linear_step_passes_recurrent_mask():
    from mtplx.models.qwen4_exp import DecoderLayer, ModelArgs, TextArgs
    from mtplx.models.qwen4_exp_compiled import (
        CompiledLayerRunner,
        compile_linear_layer_step,
    )

    args = TextArgs(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        hc_count=2,
        hc_lowrank=64,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
    )
    layer = DecoderLayer(args, layer_idx=0)
    step_fn = compile_linear_layer_step(layer)

    h = mx.ones((2, 1, 128 * 2))
    conv_mask = mx.array([[False], [True]])
    conv_state = mx.zeros((2, 3, layer.linear_attn.conv_dim))
    rec_state = mx.zeros((2, layer.linear_attn.n_v, layer.linear_attn.dv, layer.linear_attn.dk))

    h_out, new_conv, new_rec = step_fn(h, conv_mask, conv_state, rec_state)
    assert h_out.shape == (2, 1, 128 * 2)
    assert new_conv.shape == (2, 3, layer.linear_attn.conv_dim)
    assert new_rec.shape == (2, layer.linear_attn.n_v, layer.linear_attn.dv, layer.linear_attn.dk)


def test_remap_packed_expert_quantization_leaves():
    from mtplx.models.qwen4_exp import remap_fused_projections

    quant_weights = {
        "layers.0.mlp.experts.gate_up_proj.weight": mx.zeros((4, 64, 32)),
        "layers.0.mlp.experts.gate_up_proj.scales": mx.ones((4, 64, 1)),
        "layers.0.mlp.experts.gate_up_proj.biases": mx.zeros((4, 64, 1)),
        "layers.0.mlp.experts.down_proj.weight": mx.zeros((4, 32, 64)),
        "layers.0.mlp.experts.down_proj.scales": mx.ones((4, 32, 1)),
        "layers.0.mlp.experts.down_proj.biases": mx.zeros((4, 32, 1)),
    }
    remapped = remap_fused_projections(quant_weights)

    assert "layers.0.mlp.switch_mlp.gate_up_proj.weight" in remapped
    assert "layers.0.mlp.switch_mlp.gate_up_proj.scales" in remapped
    assert "layers.0.mlp.switch_mlp.gate_up_proj.biases" in remapped
    assert "layers.0.mlp.switch_mlp.down_proj.weight" in remapped
    assert "layers.0.mlp.switch_mlp.down_proj.scales" in remapped
    assert "layers.0.mlp.switch_mlp.down_proj.biases" in remapped
    assert not any("experts" in k for k in remapped)


def test_reject_incomplete_mtp_sidecar(tmp_path):
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
        "vocab_size": 50,
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
                embed_tokens=nn.Embedding(50, 128),
            )
            self.lm_head = nn.Linear(128, 50, bias=False)
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

    # 1. Truncated sidecar: save only 1 parameter
    sf_path = tmp_path / "mtp.safetensors"
    mx.save_safetensors(
        str(sf_path),
        {"mtp.pre_fc_norm_embedding.weight": mx.ones((128,))},
    )

    model = DummyModel()
    res = inject_qwen4_exp_mtp_support(model, str(tmp_path), config, allow_random_init=False)
    # Must fail because required MTP parameters are missing
    assert res is False


def test_qsa_indexer_cache_batch_protocol():
    from mtplx.models.qwen4_exp import _AttnCache, _IndexerCache

    # 1. _IndexerCache merge
    idx1 = _IndexerCache()
    idx1.update(mx.ones((1, 8, 128)))
    idx1.pooled = mx.ones((1, 2, 128))

    idx2 = _IndexerCache()
    idx2.update(mx.full((1, 12, 128), 2.0))
    idx2.pooled = mx.full((1, 3, 128), 2.0)

    merged = _IndexerCache.merge([idx1, idx2])
    assert merged.batch_size == 2
    assert merged.keys.shape == (2, 12, 128)
    assert merged.pooled is None  # Invalidated on merge

    # 2. _IndexerCache filter
    merged.filter(mx.array([1]))
    assert merged.batch_size == 1
    assert merged.keys.shape == (1, 12, 128)
    assert float(merged.keys[0, -1, 0].item()) == 2.0

    # 3. _IndexerCache extract
    extracted = merged.extract(0)
    assert extracted.batch_size == 1
    assert extracted.keys.shape == (1, 12, 128)

    # 4. _AttnCache merge, filter, extend, extract
    a1 = _AttnCache()
    a1.keys = mx.ones((1, 2, 8, 64))
    a1.values = mx.ones((1, 2, 8, 64))
    a1.offset = 8
    a1.indexer.update(mx.ones((1, 8, 128)))

    a2 = _AttnCache()
    a2.keys = mx.full((1, 2, 12, 64), 3.0)
    a2.values = mx.full((1, 2, 12, 64), 3.0)
    a2.offset = 12
    a2.indexer.update(mx.full((1, 12, 128), 3.0))

    a_merged = _AttnCache.merge([a1, a2])
    assert a_merged.keys.shape[0] == 2
    assert hasattr(a_merged, "indexer")
    assert a_merged.indexer.batch_size == 2
    assert a_merged.indexer.keys.shape == (2, 12, 128)

    # Filter a_merged
    a_merged.filter(mx.array([1]))
    assert a_merged.keys.shape[0] == 1
    assert a_merged.indexer.batch_size == 1
    assert float(a_merged.indexer.keys[0, -1, 0].item()) == 3.0

    # Extract from a_merged
    a_extracted = a_merged.extract(0)
    assert hasattr(a_extracted, "indexer")
    assert a_extracted.indexer.batch_size == 1
    assert a_extracted.indexer.keys.shape == (1, 12, 128)


def test_avoid_double_filtering_qsa_indexer():
    from mtplx.models.qwen4_exp import _AttnCache
    from mtplx.qwen4_exp_mtp_patch import _install_indexer_aware_trim

    a1 = _AttnCache()
    a1.keys = mx.ones((1, 2, 8, 64))
    a1.values = mx.ones((1, 2, 8, 64))
    a1.offset = 8
    a1.indexer.update(mx.ones((1, 8, 128)))

    a2 = _AttnCache()
    a2.keys = mx.full((1, 2, 8, 64), 2.0)
    a2.values = mx.full((1, 2, 8, 64), 2.0)
    a2.offset = 8
    a2.indexer.update(mx.full((1, 8, 128), 2.0))

    merged = _AttnCache.merge([a1, a2])
    assert merged.indexer.batch_size == 2

    # Filtering merged with index [1] must select row 1 and not fail with out-of-bounds
    merged.filter(mx.array([1]))
    assert merged.indexer.batch_size == 1
    assert float(merged.indexer.keys[0, 0, 0].item()) == 2.0


def test_mask_ple_updates_for_padded_batch_rows():
    from mtplx.models.qwen4_exp import PLELayer, TextArgs
    from mtplx.qwen4_exp_mtp_patch import Qwen4ExpRecurrentCache

    args = TextArgs(
        hidden_size=64,
        ple_embed_dim=64,
        ple_layer_ids=[1],
        hc_count=2,
        ngram_size=3,
        ple_conv_kernel_size=4,
    )
    ple = PLELayer(args, ple_layer_index=0)
    cache = Qwen4ExpRecurrentCache(4)

    # Batch of 2: row 0 is padded (mask=False), row 1 is valid (mask=True)
    mask = mx.array([[False], [True]])
    hidden = mx.ones((2, 1, 64 * 2))
    ids = mx.array([[0], [42]])
    prev_ctx = mx.zeros((2, 2), dtype=mx.int32)

    out = ple(hidden, ids, prev_ctx, cache, mask=mask)
    # Row 0 output must be zero because it is masked
    assert mx.all(out[0] == 0).item()
    # Row 1 output must be non-zero
    assert not mx.all(out[1] == 0).item()
    # Convolution state for row 0 must remain zero
    assert mx.all(cache[2][0] == 0).item()
    # Convolution state for row 1 must be updated with valid tokens
    assert not mx.all(cache[2][1] == 0).item()


def test_exclude_padded_blocks_before_qsa_topk_selection():
    from mtplx.models.qwen4_exp import QSAIndexer, TextArgs

    args = TextArgs(
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
        indexer_n_heads=4,
        indexer_head_dim=32,
        indexer_compress_ratio=4,
        indexer_budget=16,
        rms_norm_eps=1e-6,
    )
    indexer = QSAIndexer(args)
    indexer.block_topk = 2

    # B=2, S=1.
    # Total blocks = 6 (24 tokens).
    # Row 0 has 16 padding tokens (left_padding=16 -> first 4 blocks 0,1,2,3 are padding).
    # Row 1 has 0 padding tokens (left_padding=0).
    B, S, H, D = 2, 1, 4, 32
    n_blocks = 6
    r = 4
    kv_len = 24
    q = mx.ones((B, S, H, D))
    q_pos = mx.array([kv_len - 1])
    pooled = mx.ones((B, n_blocks, D))

    left_padding = mx.array([16, 0])
    sel = indexer.select(q, q_pos, pooled, kv_len, left_padding=left_padding)

    # For row 0, top-k blocks must ONLY be selected from non-padding blocks (index >= 4)
    row0_toks = sel.token_idx[0, 0].tolist()
    for idx, valid in zip(row0_toks, sel.valid[0, 0].tolist()):
        if valid:
            assert idx >= 16


def test_qwen4_tuning_routes_through_batched_verifier(monkeypatch):
    from mtplx.benchmarks.runners import mtp_depth_sweep
    from mtplx.commands import public
    from mtplx.server import openai

    mock_run_sweep = MagicMock(return_value={"ok": True})
    monkeypatch.setattr(mtp_depth_sweep, "run_mtp_depth_sweep", mock_run_sweep)
    monkeypatch.setattr(openai, "apply_memory_caps_preflight", lambda **kwargs: {})

    # Call _depth_sweep_native60 with a model named "Qwen/Qwen3.8-Flash-Next"
    public._depth_sweep_native60(
        model="Qwen/Qwen3.8-Flash-Next",
        prompt_suite="dummy",
        depths="1",
        max_tokens=10,
        limit=1,
        seed=0,
    )

    assert mock_run_sweep.called
    kwargs = mock_run_sweep.call_args.kwargs
    assert kwargs.get("verify_strategy") == "batched"
    assert kwargs.get("verify_core") == "stock"


def test_sparse_index_reuse_forwards_left_padding():
    from mtplx.models.qwen4_exp import QSAIndexer, TextArgs
    from mtplx.qwen4_exp_mtp_patch import SparseIndexReuse

    args = TextArgs(
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
        indexer_n_heads=4,
        indexer_head_dim=32,
        indexer_compress_ratio=4,
        indexer_budget=16,
        rms_norm_eps=1e-6,
    )
    indexer = QSAIndexer(args)
    wrapper = SparseIndexReuse(indexer)

    B, S, H, D = 2, 1, 4, 32
    n_blocks = 6
    kv_len = 24
    q = mx.ones((B, S, H, D))
    q_pos = mx.array([kv_len - 1])
    pooled = mx.ones((B, n_blocks, D))
    left_padding = mx.array([16, 0])

    sel = wrapper.select(q, q_pos, pooled, kv_len, left_padding=left_padding)
    assert sel is not None
    assert hasattr(sel, "token_idx")
    assert hasattr(sel, "valid")


def test_force_snapshots_in_qwen4_tuning_env(monkeypatch):
    import os
    from mtplx.qwen4_exp_mtp_patch import qwen4_exp_product_verify_env

    # Simulate default performance-cold profile setting MTPLX_SKIP_VERIFY_SNAPSHOT=1
    monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", "1")

    with qwen4_exp_product_verify_env(None) as strategy:
        assert strategy == "batched"
        assert os.environ.get("MTPLX_SKIP_VERIFY_SNAPSHOT") == "0"

    # Restored after context exit
    assert os.environ.get("MTPLX_SKIP_VERIFY_SNAPSHOT") == "1"


def test_recognize_all_qwen4_configurations_in_tuning(monkeypatch):
    import mtplx.artifacts
    from mtplx.benchmarks.runners import mtp_depth_sweep
    from mtplx.commands import public
    from mtplx.server import openai

    mock_run_sweep = MagicMock(return_value={"ok": True})
    monkeypatch.setattr(mtp_depth_sweep, "run_mtp_depth_sweep", mock_run_sweep)
    monkeypatch.setattr(openai, "apply_memory_caps_preflight", lambda **kwargs: {})

    # Mock inspect_model to return model_type="qwen4_exp_text" and architecture="Qwen4ExpForCausalLM"
    mock_inspection = SimpleNamespace(
        model_type="qwen4_exp_text",
        architecture="Qwen4ExpForCausalLM",
    )
    monkeypatch.setattr(mtplx.artifacts, "inspect_model", lambda model_path: mock_inspection)
    monkeypatch.setattr(public, "inspect_model", lambda model_path: mock_inspection)

    public._depth_sweep_native60(
        model="/tmp/dummy-qwen4-custom",
        prompt_suite="dummy",
        depths="1",
        max_tokens=10,
        limit=1,
        seed=0,
    )

    assert mock_run_sweep.called
    kwargs = mock_run_sweep.call_args.kwargs
    assert kwargs.get("verify_strategy") == "batched"
    assert kwargs.get("verify_core") == "stock"


def test_tensor_offset_paged_cache_update_and_fetch_contract():
    from mtplx.cache_state import (
        TensorOffsetVllmMetalPagedKVCache,
        VllmMetalPagedKVCache,
    )
    from mtplx.models.qwen4_exp import _IndexerCache

    paged = VllmMetalPagedKVCache(block_size=16, num_blocks=4)
    paged.indexer = _IndexerCache()
    paged.indexer.update(mx.ones((1, 8, 128)))

    # Populate initial paged state
    keys = mx.ones((1, 2, 4, 64))
    values = mx.ones((1, 2, 4, 64))
    paged.update_without_fetch(keys, values)

    promoted = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    assert hasattr(promoted, "indexer")

    # .state property returns (keys, values, indexer_keys) (3 elements for snapshotting)
    assert len(promoted.state) == 3

    # update_and_fetch() MUST return a 2-tuple (k, v) to satisfy the Attention contract
    new_keys = mx.ones((1, 2, 1, 64))
    new_values = mx.ones((1, 2, 1, 64))
    res = promoted.update_and_fetch(new_keys, new_values)
    assert isinstance(res, tuple)
    assert len(res) == 2
    k, v = res
    assert k.shape[0] == 1
    assert k.shape[1] == 2
    assert v.shape[0] == 1
    assert v.shape[1] == 2


def test_native_attn_cache_update_and_fetch_returns_2tuple():
    from mtplx.models.qwen4_exp import _AttnCache

    attn_cache = _AttnCache()
    attn_cache.indexer.update(mx.ones((1, 8, 128)))

    # Initial update_and_fetch
    k1 = mx.ones((1, 2, 4, 64))
    v1 = mx.ones((1, 2, 4, 64))
    res = attn_cache.update_and_fetch(k1, v1)
    assert isinstance(res, tuple)
    assert len(res) == 2
    k, v = res
    assert k.shape == (1, 2, 4, 64)
    assert v.shape == (1, 2, 4, 64)

    # State property retains indexer keys (3 elements for snapshotting)
    assert len(attn_cache.state) == 3

    # Subsequent update_and_fetch
    k2 = mx.ones((1, 2, 1, 64))
    v2 = mx.ones((1, 2, 1, 64))
    k_out, v_out = attn_cache.update_and_fetch(k2, v2)
    assert k_out.shape == (1, 2, 5, 64)
    assert v_out.shape == (1, 2, 5, 64)


def test_exclude_left_padding_from_ple_tail_context():
    from mtplx.models.qwen4_exp import _build_ple_tail_context

    ctx_len = 3
    eos = 151643
    prev_ctx = mx.full((2, ctx_len), eos, mx.int32)
    # Row 0 has 3 padding tokens (pad=999) and 1 valid token (42)
    # Row 1 has 0 padding tokens and 4 valid tokens (10, 20, 30, 40)
    ids = mx.array([[999, 999, 999, 42], [10, 20, 30, 40]], dtype=mx.int32)
    left_padding = mx.array([3, 0], dtype=mx.int32)

    tail = _build_ple_tail_context(prev_ctx, ids, ctx_len, eos, left_padding)
    assert tail.shape == (2, ctx_len)

    # Row 0: valid token is 42, preceding 2 tokens are filled with EOS (151643)
    row0 = tail[0].tolist()
    assert row0 == [eos, eos, 42]
    assert 999 not in row0

    # Row 1: valid tokens are 10, 20, 30, 40 -> last 3 are [20, 30, 40]
    row1 = tail[1].tolist()
    assert row1 == [20, 30, 40]


def test_embedded_mtp_tensors_outside_module_tree_and_consumed():
    from mlx.utils import tree_flatten
    from mtplx.models.qwen4_exp import Model, ModelArgs, TextArgs
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hc_count": 2,
        "hc_lowrank": 64,
        "rope_parameters": {},
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "full_attention_interval": 4,
        "vocab_size": 50,
        "mtp_num_hidden_layers": 1,
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)

    # Sanitize weights containing mtp.* keys
    raw_weights = {
        "model.embed_tokens.weight": mx.zeros((50, 128)),
        "mtp.layers.0.linear_attn.in_proj.weight": mx.zeros((256, 128)),
    }
    trunk_weights = model.sanitize(raw_weights)
    assert "mtp.layers.0.linear_attn.in_proj.weight" not in trunk_weights

    # Ensure mtp_weights are NOT registered in nn.Module parameters
    param_dict = dict(tree_flatten(model.parameters()))
    assert not any(k.startswith("_mtp_weights") or k.startswith("mtp_weights") for k in param_dict)

    # Side-channel property returns the tensors
    assert "mtp.layers.0.linear_attn.in_proj.weight" in model.mtp_weights

    # Injection consumes embedded tensors and clears the side-channel
    injected = inject_qwen4_exp_mtp_support(model, None, config=config, allow_random_init=True)
    assert injected
    assert len(model._mtp_weights) == 0


def test_temperature_scaling_in_sampling_and_verification():
    from mtplx.qwen4_exp_mtp_patch import _softmax, sample_logits_row, verify_draft_token

    logits = mx.array([1.0, 2.0, 3.0])
    p_t1 = _softmax(logits, temperature=1.0)
    p_cold = _softmax(logits, temperature=0.5)
    p_hot = _softmax(logits, temperature=2.0)

    # Colder temperature concentrates probability mass on the argmax
    assert p_cold[2] > p_t1[2]
    # Hotter temperature flattens distribution
    assert p_hot[2] < p_t1[2]

    # sample_logits_row applies temperature
    rng = MagicMock()
    rng.choice.return_value = 2
    tok, q = sample_logits_row(logits, temperature=0.5, rng=rng)
    assert tok == 2
    assert q[2] == p_cold[2]

    # verify_draft_token applies temperature
    rng.random.return_value = 0.5
    decision = verify_draft_token(logits, q, 2, temperature=0.5, rng=rng)
    assert decision.accepted


def test_moe_routing_policy_norm_topk_and_scaling_factor():
    from mtplx.models.qwen4_exp import SparseMoeBlock, TextArgs

    config = {
        "hidden_size": 64,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "norm_topk_prob": False,
        "routed_scaling_factor": 2.5,
    }
    args = TextArgs.from_dict(config)
    assert args.norm_topk_prob is False
    assert args.routed_scaling_factor == 2.5

    block = SparseMoeBlock(args)
    assert block.norm_topk_prob is False
    assert block.routed_scaling_factor == 2.5

    x = mx.ones((1, 2, 64))
    out = block(x)
    assert out.shape == (1, 2, 64)


def test_qsa_indexer_admits_partially_padded_blocks_with_per_token_mask():
    from mtplx.models.qwen4_exp import QSAIndexer, TextArgs

    args = TextArgs(
        hidden_size=64,
        indexer_n_heads=2,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=4,
    )
    indexer = QSAIndexer(args)
    indexer.block_topk = 1

    # Left padding is 3 (less than compress_ratio 4, so block 0 is partially padded)
    left_padding = mx.array([3])
    q = mx.ones((1, 1, 2, 32))
    q_pos = mx.array([8])
    pooled = mx.ones((1, 3, 32))  # 3 blocks: block 0 (0..3), block 1 (4..7), block 2 (8..11)

    sel = indexer.select(q, q_pos, pooled, kv_len=12, left_padding=left_padding)
    # Block 0 ends at 4 > 3, so it is admitted.
    selected_block_tokens = sel.token_idx[0, 0, :4].tolist()
    assert selected_block_tokens == [0, 1, 2, 3]
    # Tokens 0, 1, 2 are pad tokens (< 3), token 3 is valid (>= 3)
    assert sel.valid[0, 0, :4].tolist() == [False, False, False, True]


def test_preserve_conv_state_across_masked_timesteps():
    from mtplx.models.qwen4_exp import GatedDeltaNet, PLELayer, TextArgs
    from mlx_lm.models.cache import ArraysCache

    args = TextArgs(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        hc_count=2,
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=2,
    )

    # 1. GDN test
    gdn = GatedDeltaNet(args)
    cache = ArraysCache(4)
    init_conv = mx.array([[[1.0] * gdn.conv_dim] * (gdn.conv_kernel_size - 1), [[2.0] * gdn.conv_dim] * (gdn.conv_kernel_size - 1)])
    cache[0] = init_conv

    x = mx.ones((2, 1, 64))
    # Row 0 is masked (False), Row 1 is active (True)
    mask = mx.array([[False], [True]])
    _ = gdn(x, mask=mask, cache=cache)

    # Row 0 must preserve original convolution state (not shifted with zeros)
    assert mx.array_equal(cache[0][0], init_conv[0])
    # Row 1 advanced
    assert not mx.array_equal(cache[0][1], init_conv[1])

    # 2. PLE test
    ple = PLELayer(args, ple_layer_index=0)
    ple_cache = ArraysCache(4)
    init_ple_conv = mx.array([[[5.0] * (args.hidden_size * args.hc_count)] * ple.short_conv_state_len, [[6.0] * (args.hidden_size * args.hc_count)] * ple.short_conv_state_len])
    ple_cache[2] = init_ple_conv

    x_ple = mx.ones((2, 1, args.hidden_size * args.hc_count))
    _ = ple._short_conv(x_ple, ple_cache, mask=mask)

    # Row 0 must preserve original PLE convolution state
    assert mx.array_equal(ple_cache[2][0], init_ple_conv[0])
    # Row 1 advanced
    assert not mx.array_equal(ple_cache[2][1], init_ple_conv[1])


def test_accept_spliced_input_embeddings_in_mtp_history_update():
    from mtplx.models.qwen4_exp import Model, ModelArgs
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hc_count": 2,
        "hc_lowrank": 64,
        "rope_parameters": {},
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "full_attention_interval": 4,
        "vocab_size": 50,
        "mtp_num_hidden_layers": 1,
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)
    injected = inject_qwen4_exp_mtp_support(model, None, config=config, allow_random_init=True)
    assert injected

    # Verify mtp_update_cache accepts input_embeddings keyword argument
    hidden_states = mx.ones((1, 2, 128 * 2))
    next_tokens = mx.array([[10, 20]], dtype=mx.int32)
    spliced_emb = mx.ones((1, 2, 128))

    res = model.mtp_update_cache(
        hidden_states,
        next_tokens,
        input_embeddings=spliced_emb,
    )
    assert res.shape == (1, 2, 128 * 2)


def test_allocate_recurrent_caches_for_linear_mtp_layers():
    from mtplx.models.qwen4_exp import Model, ModelArgs
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support
    from mlx_lm.models.cache import ArraysCache

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hc_count": 2,
        "hc_lowrank": 64,
        "rope_parameters": {},
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "full_attention_interval": 4,
        "vocab_size": 50,
        "mtp_num_hidden_layers": 1,
        "mtp": {
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
        },
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)
    injected = inject_qwen4_exp_mtp_support(model, None, config=config, allow_random_init=True)
    assert injected

    mtp_cache = model.make_mtp_cache()
    assert len(mtp_cache) == 1
    assert isinstance(mtp_cache[0], ArraysCache)

    # mtp_forward executes and updates recurrent cache
    hidden_states = mx.ones((1, 1, 128 * 2))
    next_tokens = mx.array([[10]], dtype=mx.int32)
    logits, next_h = model.mtp_forward(
        hidden_states,
        next_tokens,
        mtp_cache=mtp_cache,
        return_hidden=True,
    )
    assert logits.shape[-1] == 50
    assert next_h.shape == (1, 1, 128 * 2)
    # Check that GDN conv_state and recurrent state in mtp_cache were written
    assert mtp_cache[0][0] is not None
    assert mtp_cache[0][1] is not None


def test_split_fused_eh_proj_weights():
    from mtplx.qwen4_exp_mtp_patch import _process_raw_mtp_weights, inject_qwen4_exp_mtp_support
    from mtplx.models.qwen4_exp import Model, ModelArgs

    raw_weights = {
        "mtp.pre_fc_norm_embedding.weight": mx.ones((128,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((256,)),
        "mtp.eh_proj.weight": mx.ones((128, 256)),
    }
    processed = _process_raw_mtp_weights(raw_weights)
    assert "fc_embedding.weight" in processed
    assert "fc_hidden.weight" in processed
    assert processed["fc_embedding.weight"].shape == (128, 128)
    assert processed["fc_hidden.weight"].shape == (128, 128)
    assert "eh_proj.weight" not in processed
    assert "fc.weight" not in processed


def test_forward_ar_allows_vision_splice_without_mtp():
    from mtplx.runtime import MTPLXRuntime

    class ModelWithInputEmbeddings:
        def __call__(self, inputs, cache=None, input_embeddings=None, **kwargs):
            return mx.zeros((1, 1, 10))

    class ModelWithoutInputEmbeddings:
        def __call__(self, inputs, cache=None, emit_logits=True, logits_keep=None):
            return mx.zeros((1, 1, 10))

    # 1. Model supports input_embeddings -> forward_ar succeeds without MTP
    rt = MTPLXRuntime(
        model=ModelWithInputEmbeddings(),
        tokenizer=None,
        model_path=Path("."),
        mtp_enabled=False,
        contract=None,
    )
    res = rt.forward_ar(
        mx.array([[1]]),
        input_embeddings=mx.ones((1, 1, 32)),
    )
    assert res is not None

    # 2. Model does not support input_embeddings -> forward_ar raises RuntimeError
    rt_no_emb = MTPLXRuntime(
        model=ModelWithoutInputEmbeddings(),
        tokenizer=None,
        model_path=Path("."),
        mtp_enabled=False,
        contract=None,
    )
    import pytest
    with pytest.raises(RuntimeError, match="does not accept input_embeddings"):
        rt_no_emb.forward_ar(
            mx.array([[1]]),
            input_embeddings=mx.ones((1, 1, 32)),
        )


def test_mtp_predictor_attention_uses_dense_attention():
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support
    from mtplx.models.qwen4_exp import Model, ModelArgs
    from mlx_lm.models.cache import KVCache

    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "hc_count": 2,
            "hc_lowrank": 64,
            "ple_embed_dim": 128,
            "ple_layer_ids": [],
            "vocab_size": 50,
            "layer_types": ["linear_attention", "full_attention"],
            "mtp_num_hidden_layers": 1,
            "mtp_layer_types": ["full_attention"],
        },
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)
    injected = inject_qwen4_exp_mtp_support(model, None, config=config, allow_random_init=True)
    assert injected

    # Verify that predictor's attention layers have indexer disabled (dense attention)
    assert model.mtp.layers[0].self_attn.indexer is None

    # Verify cache allocated for full attention predictor layer is dense KVCache
    mtp_cache = model.make_mtp_cache()
    assert len(mtp_cache) == 1
    assert isinstance(mtp_cache[0], KVCache)

    # Long context forward through MTP predictor remains dense causal without QSA selection errors
    hidden = mx.ones((1, 1, 128 * 2))
    next_tok = mx.array([[5]], dtype=mx.int32)
    logits, next_h = model.mtp_forward(hidden, next_tok, mtp_cache=mtp_cache, return_hidden=True)
    assert logits.shape[-1] == 50
    assert next_h.shape == (1, 1, 128 * 2)


def test_exclude_all_mtp_sidecars_from_trunk_check(tmp_path):
    from mtplx.backends.registry import (
        _passes_qwen4_exp_gate,
        _passes_mlx_lm_ar_gate,
        _is_mtp_sidecar_file,
    )
    from types import SimpleNamespace

    # 1. Check helper recognition
    assert _is_mtp_sidecar_file(Path("mtp.safetensors"))
    assert _is_mtp_sidecar_file(Path("model-mtp.safetensors"))
    assert _is_mtp_sidecar_file(Path("model-mtp-head.safetensors"))
    assert _is_mtp_sidecar_file(Path("mtp/weights.safetensors"))
    assert not _is_mtp_sidecar_file(Path("weights.safetensors"))
    assert _is_mtp_sidecar_file(Path("custom-mtp.safetensors"))
    assert _is_mtp_sidecar_file(Path("custom-mtp-head.safetensors"))
    assert not _is_mtp_sidecar_file(Path("model.safetensors"))
    assert not _is_mtp_sidecar_file(Path("model-00001-of-00004.safetensors"))

    # 2. Sidecar-only directory (e.g. model-mtp.safetensors only)
    sidecar_dir = tmp_path / "sidecar_only"
    sidecar_dir.mkdir()
    (sidecar_dir / "model-mtp.safetensors").write_bytes(b"dummy")
    (sidecar_dir / "model-mtp-head.safetensors").write_bytes(b"dummy")

    inspection = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=sidecar_dir,
    )

    assert not _passes_qwen4_exp_gate(inspection)
    assert not _passes_mlx_lm_ar_gate(inspection)

    # 3. Add genuine trunk shard -> gate passes
    (sidecar_dir / "model.safetensors").write_bytes(b"dummy")
    assert _passes_qwen4_exp_gate(inspection)
    assert _passes_mlx_lm_ar_gate(inspection)


def test_attn_cache_filters_left_padding_and_lengths():
    from mtplx.models.qwen4_exp import _AttnCache

    # Create two caches and merge them
    c1 = _AttnCache()
    c1.keys = mx.ones((1, 2, 8, 32))
    c1.values = mx.ones((1, 2, 8, 32))
    c1.offset = 8
    c1.left_padding = mx.array([2])
    c1.lengths = mx.array([6])

    c2 = _AttnCache()
    c2.keys = mx.ones((1, 2, 8, 32))
    c2.values = mx.ones((1, 2, 8, 32))
    c2.offset = 8
    c2.left_padding = mx.array([4])
    c2.lengths = mx.array([4])

    merged = _AttnCache.merge([c1, c2])
    assert merged.left_padding.shape[0] == 2
    assert merged.indexer.left_padding.shape[0] == 2

    # Filter merged batch to keep index 1 only (BatchKVCache shifts keys and updates left_padding)
    merged.filter(mx.array([1]))
    assert merged.keys.shape[0] == 1
    assert merged.left_padding.shape[0] == 1
    assert int(merged.left_padding[0].item()) == 0
    assert int(merged.indexer.left_padding[0].item()) == 0

    # Filter unmerged _AttnCache directly
    c_multi = _AttnCache()
    c_multi.keys = mx.ones((2, 2, 8, 32))
    c_multi.values = mx.ones((2, 2, 8, 32))
    c_multi.left_padding = mx.array([2, 4])
    c_multi.filter(mx.array([1]))
    assert c_multi.keys.shape[0] == 1
    assert int(c_multi.left_padding[0].item()) == 4

    # Extract
    c1_ext = c1.extract(0)
    assert c1_ext.left_padding.shape[0] == 1
    assert int(c1_ext.left_padding[0].item()) == 2

    # Extend
    c1_ext.extend(c2)
    assert c1_ext.left_padding.shape[0] == 2
    assert list(c1_ext.left_padding.tolist()) == [2, 4]


def test_metal_memory_caps_and_headroom_scale_with_small_models(monkeypatch, tmp_path):
    from mtplx.server.openai import (
        _minimum_resident_bytes_for_model_path,
        _apply_metal_memory_caps,
        apply_memory_caps_preflight,
    )

    # 1. Headroom on 1GB weights is scaled, not fixed 4GB
    monkeypatch.setattr("mtplx.engine_session.model_weights_bytes", lambda path: 1024**3)
    min_resident = _minimum_resident_bytes_for_model_path(str(tmp_path))
    assert min_resident is not None
    assert min_resident < 2 * 1024**3  # Far below 1GB + 4GB = 5GB

    # 2. Preflight on a 16GB Mac with a 2GB model passes without insufficient_ram error
    caps = _apply_metal_memory_caps(
        total_ram_bytes=16 * 1024**3,
        minimum_resident_bytes=min_resident,
    )
    assert caps["applied"] is True
    assert caps.get("reason") is None


def test_qwen4_exp_mtp_falls_back_to_ar_when_draft_head_absent(tmp_path):
    from mtplx.backends.registry import compatibility_for_inspection
    from types import SimpleNamespace

    model_dir = tmp_path / "qwen4_no_mtp"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"dummy")

    inspection = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        mtp_num_hidden_layers=1,
        model_dir=str(model_dir),
        mtp=SimpleNamespace(exists=False, passes_tensor_gate=False),
    )

    verdict = compatibility_for_inspection(inspection)
    assert verdict.can_run is True
    assert verdict.arch_id == "qwen4-exp"
    assert verdict.runtime_compatibility == "native-ar-only-missing-mtp"
    assert verdict.mtp_supported == "no"


def test_gdn_and_ple_preserve_conv_history_through_partial_padding():
    from mtplx.models.qwen4_exp import GatedDeltaNet, PLELayer, TextArgs, ArraysCache

    args = TextArgs(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        hc_count=2,
        hc_lowrank=32,
        ple_embed_dim=64,
        ple_layer_ids=[0],
        ngram_size=2,
        vocab_size=50,
    )
    gdn = GatedDeltaNet(args)
    cache = ArraysCache(4)

    # Initial step to populate history
    x_init = mx.ones((2, 3, 64))
    _ = gdn(x_init, mask=None, cache=cache)
    initial_slot0 = mx.array(cache[0])

    # Next step with 2 batch rows:
    # row 0: partially masked [False, True, True]
    # row 1: fully masked [False, False, False]
    x_step = mx.ones((2, 3, 64)) * 2.0
    mask = mx.array([[False, True, True], [False, False, False]])
    out = gdn(x_step, mask=mask, cache=cache)

    assert out.shape == (2, 3, 64)
    # Row 1 is fully masked, so output is 0 and slot0 is unchanged
    assert mx.all(out[1] == 0)
    assert mx.all(cache[0][1] == initial_slot0[1])

    # Row 0 advanced by 2 valid timesteps without zero-padding contamination
    # slot0 has length conv_kernel_size - 1 = 3
    # last 2 elements of cache[0][0] come from valid tokens of x_step, first element from initial_slot0[0]
    assert cache[0].shape == (2, 3, gdn.conv_dim)


def test_attn_cache_extend_without_left_padding_receiver():
    from mtplx.models.qwen4_exp import _AttnCache

    # Receiver has no left_padding (batch size 2)
    c1 = _AttnCache()
    c1.keys = mx.ones((2, 2, 8, 32))
    c1.values = mx.ones((2, 2, 8, 32))

    # Other has left_padding (batch size 1)
    c2 = _AttnCache()
    c2.keys = mx.ones((1, 2, 8, 32))
    c2.values = mx.ones((1, 2, 8, 32))
    c2.left_padding = mx.array([3], dtype=mx.int32)

    c1.extend(c2)
    assert c1.keys.shape[0] == 3
    # left_padding must have length 3 (2 for receiver, 1 for other), not 4
    assert c1.left_padding.shape[0] == 3
    assert list(c1.left_padding.tolist()) == [0, 0, 3]


def test_load_raw_sidecar_transposes_conv1d_weights(tmp_path):
    from mtplx.qwen4_exp_mtp_patch import _load_mtp_weights

    sidecar_path = tmp_path / "mtp.safetensors"
    # Raw HF conv1d weight layout: (C, 1, K)
    dummy_weights = {
        "mtp.layers.0.linear_attn.conv1d.weight": mx.ones((16, 1, 4)),
    }
    mx.save_safetensors(str(sidecar_path), dummy_weights)

    loaded = _load_mtp_weights([sidecar_path])
    conv_w = loaded["layers.0.linear_attn.conv1d.weight"]
    # MLX Conv1d weight layout: (C, K, 1)
    assert conv_w.shape == (16, 4, 1)


def test_qwen4_server_retains_d1_default_when_not_explicit():
    from argparse import Namespace
    from mtplx.server.openai import _apply_backend_server_defaults, parse_args
    from mtplx.backends.descriptors import QWEN3_NEXT_DESCRIPTOR

    # When serving Qwen4 without explicit --depth flag
    args = Namespace(
        model="Qwen/Qwen3.8-Flash-Next",
        model_id="Qwen/Qwen3.8-Flash-Next",
        backend="qwen3_next",
    )
    explicit_flags = set()
    _apply_backend_server_defaults(args, explicit_flags=explicit_flags)
    assert getattr(args, "_explicit_depth", False) is False
    assert getattr(args, "depth", None) == 1


def test_top_level_weights_safetensors_not_classified_as_sidecar(tmp_path):
    from pathlib import Path
    from mtplx.backends.registry import _is_mtp_sidecar_file, _passes_mlx_lm_ar_gate, _passes_qwen4_exp_gate
    from types import SimpleNamespace

    # Top-level weights.safetensors is a legitimate trunk shard
    top_level_weights = tmp_path / "weights.safetensors"
    top_level_weights.write_bytes(b"dummy")
    config_file = tmp_path / "config.json"
    config_file.write_text('{"model_type": "qwen4_exp"}')

    assert not _is_mtp_sidecar_file(top_level_weights)

    inspection = SimpleNamespace(
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_dir=str(tmp_path),
    )
    assert _passes_qwen4_exp_gate(inspection)
    assert _passes_mlx_lm_ar_gate(inspection)

    # Subdirectory mtp/weights.safetensors is a recognized sidecar
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    mtp_weights = mtp_dir / "weights.safetensors"
    mtp_weights.write_bytes(b"dummy")
    assert _is_mtp_sidecar_file(mtp_weights)


def test_speculative_generate_preserves_mtp_history_between_rounds():
    from mtplx.models.qwen4_exp import Model, ModelArgs
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support, speculative_generate

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "vocab_size": 50,
        "ple_layer_ids": [],
        "mtp": {
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
        },
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)
    inject_qwen4_exp_mtp_support(model, None, config=config, allow_random_init=True)

    prompt_ids = mx.array([[1, 2, 3, 4]])
    tokens = speculative_generate(model, prompt_ids, max_tokens=8, draft_depth=2)
    assert len(tokens) == 8
    assert all(isinstance(t, int) for t in tokens)


def test_server_uses_runtime_selected_draft_default_when_not_explicit():
    from types import SimpleNamespace
    from mtplx.backends.descriptors import (
        QWEN3_NEXT_DESCRIPTOR,
        QWEN4_EXP_DRAFT_SEMANTICS,
    )
    from mtplx.server.openai import _request_depth_for_generation

    # Simulate loaded state where args has depth=1 set by runtime startup
    args = SimpleNamespace(
        _explicit_depth=False,
        depth=1,
        mtp_depth=1,
    )
    state = SimpleNamespace(
        args=args,
        backend_descriptor=QWEN3_NEXT_DESCRIPTOR,  # generic descriptor has default=3
        model_id="local_model_without_name_marker",
    )
    req = SimpleNamespace(depth=None, mtp_depth=None, speculative_depth=None)
    depth = _request_depth_for_generation(
        state,
        req,
        generation_mode="speculative",
    )
    assert depth == 1


def test_rejection_restores_post_step_mtp_snapshot():
    from mtplx.models.qwen4_exp import Model, ModelArgs
    from mtplx.qwen4_exp_mtp_patch import inject_qwen4_exp_mtp_support, draft_tokens

    config = {
        "model_type": "qwen4_exp",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "vocab_size": 50,
        "ple_layer_ids": [],
        "mtp": {
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
        },
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)
    inject_qwen4_exp_mtp_support(model, None, config=config, allow_random_init=True)

    mtp_cache = model.make_mtp_cache()
    logits, h = model(mx.array([[10]]), return_hidden=True)
    drafts, qs, h_out, mtp_snaps = draft_tokens(
        model,
        h,
        token_id=10,
        n=2,
        mtp_cache=mtp_cache,
        return_snapshots=True,
    )
    assert len(mtp_snaps) == 2
    # Snapshot 0 should be post-step-0 (contains the token_id 10 KV entry)
    # Restore snapshot 0
    mtp_snaps[0].restore(mtp_cache)
    attn_cache = mtp_cache[0]
    assert attn_cache.offset == 1


def test_kv_promotion_preserves_left_padding_and_make_mask():
    from mtplx.cache_state import (
        TailOwnedKVCache,
        BlockOwnedKVCache,
        VllmMetalPagedKVCache,
        TensorOffsetVllmMetalPagedKVCache,
    )

    class MockEntry:
        def __init__(self):
            self.keys = mx.zeros((1, 2, 8, 32))
            self.values = mx.zeros((1, 2, 8, 32))
            self.offset = 8
            self.step = 256
            self.left_padding = mx.array([3], dtype=mx.int32)

    entry = MockEntry()
    tail = TailOwnedKVCache.from_cache(entry, mode="eval_only")
    assert tail.left_padding is not None
    assert int(tail.left_padding[0].item()) == 3
    mask = tail.make_mask(1)
    assert mask is not None

    block = BlockOwnedKVCache.from_cache(entry, mode="eval_only")
    assert block.left_padding is not None
    assert int(block.left_padding[0].item()) == 3

    paged = VllmMetalPagedKVCache.from_cache(entry, block_size=16, num_blocks=4)
    assert paged.left_padding is not None
    assert int(paged.left_padding[0].item()) == 3
    paged_mask = paged.make_mask(1)
    assert paged_mask is not None

    tensor_paged = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    assert tensor_paged.left_padding is not None
    assert int(tensor_paged.left_padding[0].item()) == 3
    tensor_mask = tensor_paged.make_mask(1)
    assert tensor_mask is not None

    demoted = tensor_paged.to_paged_cache()
    assert demoted.left_padding is not None
    assert int(demoted.left_padding[0].item()) == 3


def test_detect_nested_qwen4_mtp_layer_declarations(tmp_path):
    import json
    from mtplx.artifacts import inspect_model
    from mtplx.backends.registry import _detect_arch_id

    config_data = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp",
            "mtp": {
                "num_hidden_layers": 1,
            },
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    model_weight = tmp_path / "model.safetensors"
    model_weight.write_bytes(b"dummy")

    inspection = inspect_model(tmp_path)
    assert inspection.mtp_num_hidden_layers == 1
    arch_id = _detect_arch_id(inspection)
    assert arch_id == "qwen4-exp-mtp"


def test_qwen4_exp_mtp_patch_recognizes_outer_and_legacy_mtp_declarations():
    from mtplx.qwen4_exp_mtp_patch import _num_mtp_layers, is_qwen4_exp_mtp_config

    # 1. Outer mtp declaration
    cfg_outer = {
        "model_type": "qwen4_exp",
        "mtp": {"num_hidden_layers": 1},
    }
    assert _num_mtp_layers(cfg_outer) == 1
    assert is_qwen4_exp_mtp_config(cfg_outer) is True

    # 2. Legacy num_mtp_modules
    cfg_legacy = {
        "model_type": "qwen4_exp",
        "num_mtp_modules": 1,
    }
    assert _num_mtp_layers(cfg_legacy) == 1
    assert is_qwen4_exp_mtp_config(cfg_legacy) is True

    # 3. Nested text_config.num_mtp_modules
    cfg_nested = {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": "qwen4_exp", "num_mtp_modules": 1},
    }
    assert _num_mtp_layers(cfg_nested) == 1
    assert is_qwen4_exp_mtp_config(cfg_nested) is True


def test_qwen4_exp_sanitize_remaps_conditional_generation_prefixes():
    from mtplx.models.qwen4_exp import sanitize

    raw_weights = {
        "model.language_model.model.embed_tokens.weight": mx.zeros((10, 10)),
        "model.language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
        "model.language_model.lm_head.weight": mx.zeros((10, 10)),
        "model.language_model.mtp.fc.weight": mx.zeros((10, 10)),
        "model.language_model.layers.1.self_attn.q_proj.weight": mx.zeros((10, 10)),
        "model.visual.patch_embed.weight": mx.zeros((5, 5)),
        "vision_tower.encoder.weight": mx.zeros((5, 5)),
        "visual.conv.weight": mx.zeros((5, 5)),
    }

    sanitized = sanitize(raw_weights)
    assert "model.embed_tokens.weight" in sanitized
    assert "model.layers.0.self_attn.q_proj.weight" in sanitized
    assert "model.layers.1.self_attn.q_proj.weight" in sanitized
    assert "lm_head.weight" in sanitized
    assert "mtp.fc.weight" in sanitized
    assert "model.visual.patch_embed.weight" not in sanitized
    assert "vision_tower.encoder.weight" not in sanitized
    assert "visual.conv.weight" not in sanitized
    assert not any(k.startswith("model.language_model.") for k in sanitized)
    assert not any(k.startswith("language_model.") for k in sanitized)


def test_qsa_indexer_select_admits_partially_padded_blocks_with_correct_mask():
    from mtplx.models.qwen4_exp import QSAIndexer, TextArgs

    args = TextArgs(
        hidden_size=32,
        indexer_head_dim=16,
        indexer_n_heads=2,
        indexer_compress_ratio=4,
        indexer_budget=8,
    )
    indexer = QSAIndexer(args)
    indexer.block_topk = 2
    # 2 blocks of 4 tokens = 8 tokens. left_padding = 5 (tokens 0..4 are pad, 5..7 are real)
    pooled = mx.ones((1, 2, 16))
    q = mx.ones((1, 1, 2, 16))
    q_pos = mx.array([7])
    left_padding = mx.array([5])

    sel = indexer.select(
        q=q,
        q_pos=q_pos,
        pooled=pooled,
        kv_len=8,
        left_padding=left_padding,
    )
    # Block 1 should be selected since its end (8) > left_padding (5)
    # In token_idx: token 4 must be masked as invalid, while tokens 5, 6, 7 are valid
    tok_list = sel.token_idx.flatten().tolist()
    val_list = sel.valid.flatten().tolist()
    valid_tokens = [tok for tok, v in zip(tok_list, val_list) if v]
    assert len(valid_tokens) > 0
    assert all(t >= 5 for t in valid_tokens)


def test_ple_tail_context_updates_during_decode_steps():
    from mtplx.models.qwen4_exp import _build_ple_tail_context

    ctx_len = 3
    eos = 0
    # Initial prefill with prompt of length 6, left_padding = 2
    prompt_ids = mx.array([[eos, eos, 10, 11, 12, 13]])
    left_padding = mx.array([2])
    prev_ctx = mx.array([[eos, eos, eos]])

    tail_prefill = _build_ple_tail_context(
        prev_ctx,
        prompt_ids,
        ctx_len=ctx_len,
        eos=eos,
        left_padding=left_padding,
        offset=0,
    )
    # Valid tokens are [10, 11, 12, 13], last 3 are [11, 12, 13]
    assert tail_prefill.tolist() == [[11, 12, 13]]

    # Step 1 decode (offset = 6, ids = [[14]])
    decode_ids = mx.array([[14]])
    tail_step1 = _build_ple_tail_context(
        tail_prefill,
        decode_ids,
        ctx_len=ctx_len,
        eos=eos,
        left_padding=left_padding,
        offset=6,
    )
    assert tail_step1.tolist() == [[12, 13, 14]]


def test_passes_qwen4_exp_gate_supports_hf_remote_inspections():
    from types import SimpleNamespace
    from mtplx.backends.registry import _passes_qwen4_exp_gate, _passes_mlx_lm_ar_gate

    hf_inspection = SimpleNamespace(
        source="hf",
        model_dir="Qwen/Qwen3.8-Flash-Next-4bit",
        model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        model_files=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    )
    assert _passes_qwen4_exp_gate(hf_inspection) is True
    assert _passes_mlx_lm_ar_gate(hf_inspection) is True


def test_resolve_context_window_reads_original_max_position_from_rope_scaling(tmp_path):
    import json
    from mtplx.server.openai import _resolve_context_window

    config_data = {
        "max_position_embeddings": 262144,
        "rope_scaling": {
            "factor": 4.0,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    ctx = _resolve_context_window(None, str(tmp_path))
    assert ctx == 262144


def test_onboarding_scan_detects_model_mtp_head_safetensors(tmp_path):
    from mtplx.ui.onboarding import _scan_mtp_sidecar_exists

    mtp_head_file = tmp_path / "model-mtp-head.safetensors"
    mtp_head_file.write_bytes(b"dummy")

    assert _scan_mtp_sidecar_exists(tmp_path, {}) is True


def test_is_qwen4_exp_config_recognizes_architecture_aliases():
    from mtplx.qwen4_exp_mtp_patch import is_qwen4_exp_config

    assert is_qwen4_exp_config({"architectures": ["Qwen4ExpForConditionalGeneration"]}) is True
    assert is_qwen4_exp_config({"architectures": ["Qwen4ExpForCausalLM"]}) is True
    assert is_qwen4_exp_config({"text_config": {"architectures": ["Qwen4ExpForConditionalGeneration"]}}) is True
    assert is_qwen4_exp_config({"architectures": ["LlamaForCausalLM"]}) is False














