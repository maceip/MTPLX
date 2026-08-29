import os
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mtplx.generation import generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class TinyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def sanitize(self, weights):
        return weights

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            if return_hidden:
                return None, hidden
            return None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


def _make_runtime(model: TinyModel) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def test_double_buffered_async_decode_default(monkeypatch):
    """Verify generate_ar runs with async evaluation by default."""
    monkeypatch.delenv("MTPLX_SYNC_AR", raising=False)
    model = TinyModel()
    rt = _make_runtime(model)
    out = generate_ar(
        rt,
        [0],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )
    assert len(out.tokens) == 4


def test_double_buffered_async_decode_sync_flag(monkeypatch):
    """Verify generate_ar respects MTPLX_SYNC_AR=1 fallback."""
    monkeypatch.setenv("MTPLX_SYNC_AR", "1")
    model = TinyModel()
    rt = _make_runtime(model)
    out = generate_ar(
        rt,
        [0],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )
    assert len(out.tokens) == 4
