from __future__ import annotations

from mtplx.attention_split import configure_split_full_attention


class DummyProjection:
    weight = None


class DummyAttention:
    q_proj = DummyProjection()
    q_norm = DummyProjection()

    def __call__(self, x, mask=None, cache=None):
        return x


class DummyLayer:
    is_linear = False

    def __init__(self):
        self.self_attn = DummyAttention()


class DummyInner:
    def __init__(self):
        self.layers = [DummyLayer()]


class DummyModel:
    def __init__(self):
        self.model = DummyInner()


def test_vllm_paged_hook_does_not_enable_split_full_attention(monkeypatch):
    monkeypatch.delenv("MTPLX_SPLIT_FULL_ATTN", raising=False)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")

    model = DummyModel()
    stats = configure_split_full_attention(model)
    attn = model.model.layers[0].self_attn

    assert stats["enabled"] is True
    assert stats["split_full_attn_enabled"] is False
    assert attn._mtplx_vllm_metal_paged_enabled is True
    assert attn._mtplx_split_full_attention_explicit_enabled is False


def test_qsa_indexer_attention_is_not_split_hooked(monkeypatch):
    """qwen4_exp QSA must not receive the Qwen3Next split-SDPA class patch."""

    monkeypatch.setenv("MTPLX_SPLIT_FULL_ATTN", "1")

    class QsaAttention:
        indexer = object()

        def __call__(self, x, mask=None, cache=None):
            return x

    class HouseAttention:
        def __call__(self, x, mask=None, cache=None):
            return x

    class QsaLayer:
        is_linear = False

        def __init__(self):
            self.self_attn = QsaAttention()

    class HouseLayer:
        is_linear = False

        def __init__(self):
            self.self_attn = HouseAttention()

    class MixedModel:
        def __init__(self):
            self.model = type(
                "Inner", (), {"layers": [QsaLayer(), HouseLayer()]}
            )()

    model = MixedModel()
    stats = configure_split_full_attention(model)
    qsa = model.model.layers[0].self_attn
    house = model.model.layers[1].self_attn

    assert stats["layers"] == 1
    assert stats["installed"] == 1
    assert not getattr(type(qsa), "_mtplx_split_full_attention_installed", False)
    assert getattr(type(house), "_mtplx_split_full_attention_installed", False)


def test_explicit_split_full_attention_chunk_one_gets_safe_default(monkeypatch):
    monkeypatch.setenv("MTPLX_SPLIT_FULL_ATTN", "1")
    monkeypatch.setenv("MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE", "1")

    model = DummyModel()
    stats = configure_split_full_attention(model)
    attn = model.model.layers[0].self_attn

    assert stats["split_full_attn_enabled"] is True
    assert stats["split_full_attn_chunk_size"] == 2048
    assert stats["split_full_attn_chunk_size_defaulted"] is True
    assert attn._mtplx_split_full_attention_explicit_enabled is True
    assert attn._mtplx_split_full_attention_chunk_size == 2048
