"""Native MTP speculative decoding for Qwen3.8-Flash-Next (``qwen4_exp``).

The trunk is a hybrid of Gated DeltaNet (recurrent) and sparse QSA attention.
Existing MTPLX MTP patches only roll back **KV offset**; that is not enough
here. ``StateSnapshot`` captures every GDN recurrent/conv slot (and PLE
slots if present) plus attention KV lengths, then restores them on a
Leviathan-Chen rejection so the accepted prefix is the only committed state.

The MTP head itself is one sparse full-attention block (config
``text_config.mtp.layer_types = ["full_attention"]``) with the usual
enorm/hnorm/fc mix. Transformers discards ``mtp.*``
(``_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]``); this injector loads
those tensors onto a runtime module.

Draft depth: engine can emit up to N=4 (tech-report 4-step speculative
decoding). Auto-tune surface stays AR/D1/D2/D3, matching the rest of MTPLX.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, is_mtp_key, text_config
from .sampling import SpeculativeDecision, verify_one_token

logger = logging.getLogger(__name__)

QWEN4_EXP_MODEL_TYPES = {"qwen4_exp", "qwen4_exp_text"}
DEFAULT_DRAFT_DEPTH = 1
MAX_DRAFT_DEPTH = 3
TUNE_CANDIDATES = ("AR", "D1", "D2", "D3")
_REFERENCE_QWEN4_EXP = Path("/tmp/mlx-lm-pr1788/mlx_lm/models/qwen4_exp.py")
_MTP_ALIAS = {
    "enorm.weight": "pre_fc_norm_embedding.weight",
    "hnorm.weight": "pre_fc_norm_hidden.weight",
    "final_layernorm.weight": "norm.weight",
    "shared_head_norm.weight": "norm.weight",
}


# --------------------------------------------------------------------------- config


def _model_type(config: dict[str, Any]) -> str:
    tcfg = text_config(config)
    return str(tcfg.get("model_type") or config.get("model_type") or "").lower()


def _mtp_block_config(config: dict[str, Any]) -> dict[str, Any]:
    tcfg = text_config(config)
    raw = tcfg.get("mtp") if isinstance(tcfg.get("mtp"), dict) else {}
    return dict(raw)


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    mtp = _mtp_block_config(config)
    return int(
        mtp.get("num_hidden_layers")
        or tcfg.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or config.get("mtp_num_hidden_layers")
        or config.get("num_nextn_predict_layers")
        or 0
    )


def is_qwen4_exp_config(config: dict[str, Any]) -> bool:
    return _model_type(config) in QWEN4_EXP_MODEL_TYPES


def is_qwen4_exp_mtp_config(config: dict[str, Any]) -> bool:
    return is_qwen4_exp_config(config) and _num_mtp_layers(config) > 0


def clamp_draft_depth(depth: int | None) -> int:
    raw = DEFAULT_DRAFT_DEPTH if depth is None else int(depth)
    return max(1, min(MAX_DRAFT_DEPTH, raw))


def tune_label_to_depth(label: str) -> int:
    text = str(label or "").strip().upper()
    if text in {"AR", "D0", "0"}:
        return 0
    if text.startswith("D") and text[1:].isdigit():
        return clamp_draft_depth(int(text[1:]))
    if text.isdigit():
        return clamp_draft_depth(int(text))
    raise ValueError(f"unknown draft-depth label {label!r}; expected AR or D1-D{MAX_DRAFT_DEPTH}")


# --------------------------------------------------------------------------- qwen4_exp import (Lane 1, mlx-lm, or PR #1788 reference)


def import_qwen4_exp():
    """Load the AR module Lane 1 owns, falling back to the mlx-lm reference."""
    for name in ("mtplx.models.qwen4_exp", "mlx_lm.models.qwen4_exp"):
        if name in sys.modules:
            return sys.modules[name]
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    if _REFERENCE_QWEN4_EXP.is_file():
        spec = importlib.util.spec_from_file_location(
            "mlx_lm.models.qwen4_exp", _REFERENCE_QWEN4_EXP
        )
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["mlx_lm.models.qwen4_exp"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "qwen4_exp AR module is not available (mtplx.models.qwen4_exp, "
        "mlx_lm.models.qwen4_exp, or /tmp/mlx-lm-pr1788 reference)"
    )


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _text_args(model: Any) -> Any:
    owner = _text_model(model)
    args = getattr(owner, "args", None)
    if args is not None and hasattr(args, "text"):
        return args.text
    inner = getattr(owner, "model", owner)
    return getattr(inner, "args", args)


# --------------------------------------------------------------------------- weights


def _strip_mtp_prefix(key: str) -> str | None:
    k = str(key)
    for outer in ("language_model.", "model.model.", "model."):
        if k.startswith(outer) and "mtp." in k:
            k = k[k.index("mtp.") :]
            break
    if not k.startswith("mtp."):
        if is_mtp_key(str(key)):
            k = k[k.index("mtp.") :]
        else:
            return None
    local = k[len("mtp.") :]
    return _MTP_ALIAS.get(local, local)


def _candidate_weight_files(model_path: Path, config: dict[str, Any]) -> list[Path]:
    mtp_file = expected_mtp_file(model_path, config)
    if mtp_file.exists():
        return [mtp_file]
    for name in ("model-mtp-head.safetensors", "mtp.safetensors"):
        head = model_path / name
        if head.exists():
            return [head]
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        selected = {
            model_path / rel
            for key, rel in weight_map.items()
            if _strip_mtp_prefix(str(key)) is not None
        }
        if selected:
            return sorted(selected)
    return sorted(model_path.glob("model*.safetensors"))


def _split_eh_proj_weights(weights: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    out = dict(weights)
    keys = list(out.keys())
    for k in keys:
        if "eh_proj." in k or k.startswith("fc.") or ".fc." in k:
            val = out.pop(k)
            for name in ("eh_proj.", "fc."):
                if name in k:
                    prefix, suffix = k.split(name, 1)
                    break
            else:
                prefix = ""
                suffix = k

            if (
                getattr(val, "shape", None)
                and len(val.shape) >= 2
                and val.shape[-1] % 2 == 0
            ):
                e_val, h_val = mx.split(val, 2, axis=-1)
                out[f"{prefix}fc_embedding.{suffix}"] = e_val
                out[f"{prefix}fc_hidden.{suffix}"] = h_val
            elif (
                getattr(val, "shape", None)
                and len(val.shape) == 1
                and val.shape[0] % 2 == 0
            ):
                e_val, h_val = mx.split(val, 2, axis=0)
                out[f"{prefix}fc_embedding.{suffix}"] = e_val
                out[f"{prefix}fc_hidden.{suffix}"] = h_val
            else:
                out[k] = val
    return out


def _transpose_mtp_conv_weights(weights: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in weights.items():
        if "conv1d.weight" in k and getattr(v, "ndim", 0) == 3 and v.shape[-1] != 1:
            if v.shape[1] == 1:
                v = v.transpose(0, 2, 1)
        out[k] = v
    return out


def _process_raw_mtp_weights(
    raw: dict[str, Any], is_mlx_format: bool = False
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        local = _strip_mtp_prefix(key)
        if local is not None:
            mapped[local] = value
        else:
            mapped[key] = value
    remapped = _remap_mtp_moe_weights(mapped)
    remapped = _split_eh_proj_weights(remapped)
    if is_mlx_format:
        return remapped
    transposed = _transpose_mtp_conv_weights(remapped)
    return _shift_qwen4_gemma_mtp_norms(transposed)


def _load_mtp_weights(paths: list[Path]) -> dict[str, Any]:
    import mlx.core as mx

    mapped: dict[str, Any] = {}
    is_mlx_format = False
    for path in paths:
        if path.suffix != ".safetensors":
            continue
        try:
            loaded, metadata = mx.load(str(path), return_metadata=True)
            if isinstance(metadata, dict) and str(metadata.get("format", "")).lower() == "mlx":
                is_mlx_format = True
        except Exception:
            loaded = mx.load(str(path))
            metadata = None
        for key, value in loaded.items():
            local = _strip_mtp_prefix(key)
            if local is not None:
                mapped[local] = value
    remapped = _remap_mtp_moe_weights(mapped)
    remapped = _split_eh_proj_weights(remapped)
    if is_mlx_format:
        return remapped
    transposed = _transpose_mtp_conv_weights(remapped)
    return _shift_qwen4_gemma_mtp_norms(transposed)


# Qwen4ExpTextRMSNorm is Gemma-style ``(1 + weight)`` with zero init. MLX
# ``mx.fast.rms_norm`` multiplies by ``weight`` (one-centered). The MLX-4bit
# sidecar is ``torch-layout-fp16-v1`` (raw HF), so these 1-D gains must be
# shifted. q/k norms train large enough that the shared delta detector
# (``max(qk) < 1.25``) treats the sidecar as absolute and would leave
# ``pre_fc_norm_*`` at negative gains.
_QWEN4_GEMMA_NORM_SUFFIXES = (
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
)


def _remap_mtp_moe_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Map converter-left packed expert tensors onto fused SwitchGLU names.

    Transformers ``Qwen4ExpTextExperts.gate_up_proj`` is ``(E, 2*I, H)`` and
    ``F.linear(...).chunk(2, dim=-1)`` splits **gate | up** (concatenated, not
    interleaved). FusedSwitchGLU wants that packed ``(E, 2*I, H)`` tensor as
    ``switch_mlp.gate_up_proj``. Shared-expert and GDN split pairs are fused
    by the same load-time remap as the trunk.
    """
    qwen4 = import_qwen4_exp()
    remap = getattr(qwen4, "remap_fused_projections", None)
    if remap is None:
        raise RuntimeError("qwen4_exp remap_fused_projections is not available")
    return remap(dict(weights))


def _shift_qwen4_gemma_mtp_norms(
    weights: dict[str, Any], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Restore MLX-absolute RMSNorm gains on a raw-HF qwen4_exp MTP sidecar."""
    if metadata and str(metadata.get("format", "")).lower() == "mlx":
        return weights
    targets = [
        (key, value)
        for key, value in weights.items()
        if getattr(value, "ndim", None) == 1
        and any(str(key).endswith(suffix) for suffix in _QWEN4_GEMMA_NORM_SUFFIXES)
    ]
    if not targets:
        return weights
    for key, value in targets:
        if str(key).endswith("pre_fc_norm_embedding.weight") or str(key).endswith(
            "pre_fc_norm_hidden.weight"
        ):
            try:
                if float(value.mean().item()) > 0.5:
                    return weights
            except Exception:
                pass
    try:
        means = [float(v.mean().item()) for _, v in targets]
        if sum(means) / len(means) > 0.5:
            return weights
    except Exception:
        pass

    out = dict(weights)
    for key, value in targets:
        out[key] = value + 1.0
    logger.info(
        "[Qwen4Exp MTP inject] Gemma-delta MTP norms +1.0 (%d tensors)",
        len(targets),
    )
    return out


# --------------------------------------------------------------------------- sparse top-k reuse (tech-report trick)


class SparseIndexReuse:
    """Reuse QSA top-k block indices across single-token MTP draft steps.

    The indexer keys still advance (new token is always appended). The keep
    mask for the already-scored prefix is reused; newly appended tokens are
    forced visible. Falls back to a full indexer call on the first step or
    whenever the budget has not kicked in.
    """

    def __init__(self, indexer: Any):
        self.indexer = indexer
        self.prev_keep = None
        self.enabled = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.indexer, name)

    def reset(self) -> None:
        self.prev_keep = None

    def prepare(self, x, rope, cache, offset: int):
        return self.indexer.prepare(x, rope, cache, offset)

    def select(self, q, q_pos, pooled, kv_len, left_padding=None, *args, **kwargs):
        return self.indexer.select(
            q, q_pos, pooled, kv_len, left_padding=left_padding, *args, **kwargs
        )

    def __call__(self, x, rope, cache, offset: int):
        import mlx.core as mx

        if not self.enabled or self.prev_keep is None or int(x.shape[1]) != 1:
            keep = self.indexer(x, rope, cache, offset)
            self.prev_keep = keep
            return keep

        B, S, _ = x.shape
        qk = self.indexer.index_qk_proj(x)
        split = self.indexer.n_heads * self.indexer.head_dim
        raw_k = qk[..., split:].reshape(B, S, self.indexer.head_dim)
        if cache is not None:
            raw_k = cache.update(raw_k)
        kv_len = int(raw_k.shape[1])
        if kv_len <= int(self.indexer.token_budget):
            self.prev_keep = None
            return None
        prev = self.prev_keep
        if hasattr(prev, "token_idx") and hasattr(prev, "valid"):
            new_idx = mx.full((B, 1, 1), kv_len - 1, dtype=mx.int32)
            new_ok = mx.ones((B, 1, 1), dtype=mx.bool_)
            keep = type(prev)(
                token_idx=mx.concatenate([prev.token_idx, new_idx], axis=-1),
                valid=mx.concatenate([prev.valid, new_ok], axis=-1),
            )
        else:
            old = int(prev.shape[-1])
            extra = kv_len - old
            if extra < 0:
                keep = prev[..., :kv_len]
            elif extra == 0:
                keep = prev
            else:
                pad = mx.ones((*prev.shape[:-1], extra), dtype=prev.dtype)
                keep = mx.concatenate([prev, pad], axis=-1)
        self.prev_keep = keep
        return keep


# --------------------------------------------------------------------------- GDN / KV snapshot


def _is_arrays_cache(entry: Any) -> bool:
    return hasattr(entry, "cache") and isinstance(getattr(entry, "cache", None), list)


def _is_kv_cache(entry: Any) -> bool:
    return hasattr(entry, "offset") and (
        hasattr(entry, "keys") or hasattr(entry, "update_and_fetch")
    )


def _hold_array(value: Any, *, copy: bool) -> Any:
    if value is None:
        return None
    import mlx.core as mx

    if copy:
        held = value + mx.zeros_like(value)
        mx.eval(held)
        return held
    # Lazy COW view: `value[...]` retains the current buffer so a later
    # `cache[i] = new_state` rebind cannot donate/mutate the snapshot.
    return value[...]


@dataclass
class LayerSnapshot:
    kind: str
    slots: tuple[Any, ...] = ()
    lengths: Any = None
    left_padding: Any = None
    offset: int = 0
    keys: Any = None
    values: Any = None
    indexer_keys: Any = None


@dataclass
class StateSnapshot:
    """Pre-draft snapshot of trunk recurrent state + attention KV lengths.

    GDN updates rebind ``ArraysCache`` slots (``cache[i] = new_state``), so a
    lazy view of the pre-forward array stays valid. Attention layers trim by
    offset. On rejection at draft position k, :meth:`restore` returns the
    cache to this snapshot; the caller then replays the accepted prefix.
    """

    layers: tuple[LayerSnapshot, ...] = ()

    @classmethod
    def capture(cls, cache: list[Any] | None, *, copy: bool = False) -> "StateSnapshot":
        if not cache:
            return cls(layers=())
        layers: list[LayerSnapshot] = []
        for entry in cache:
            if entry is None:
                layers.append(LayerSnapshot(kind="empty"))
            elif _is_arrays_cache(entry):
                slots = tuple(_hold_array(slot, copy=copy) for slot in entry.cache)
                layers.append(
                    LayerSnapshot(
                        kind="gdn",
                        slots=slots,
                        lengths=_hold_array(getattr(entry, "lengths", None), copy=copy),
                        left_padding=_hold_array(
                            getattr(entry, "left_padding", None), copy=copy
                        ),
                    )
                )
            elif _is_kv_cache(entry):
                indexer = getattr(entry, "indexer", None)
                layers.append(
                    LayerSnapshot(
                        kind="kv",
                        offset=int(getattr(entry, "offset", 0) or 0),
                        keys=_hold_array(getattr(entry, "keys", None), copy=copy),
                        values=_hold_array(getattr(entry, "values", None), copy=copy),
                        indexer_keys=_hold_array(
                            getattr(indexer, "keys", None), copy=copy
                        ),
                    )
                )
            else:
                layers.append(LayerSnapshot(kind="unknown"))
        return cls(layers=tuple(layers))

    def restore(self, cache: list[Any] | None) -> None:
        if not cache:
            return
        if len(cache) != len(self.layers):
            raise ValueError(
                f"StateSnapshot size mismatch: cache has {len(cache)} layers, "
                f"snapshot has {len(self.layers)}"
            )
        for entry, snap in zip(cache, self.layers):
            if entry is None or snap.kind in {"empty", "unknown"}:
                continue
            if snap.kind == "gdn" and _is_arrays_cache(entry):
                for idx, slot in enumerate(snap.slots):
                    if idx < len(entry.cache):
                        entry.cache[idx] = slot
                    else:
                        entry.cache.append(slot)
                if hasattr(entry, "lengths"):
                    entry.lengths = snap.lengths
                if hasattr(entry, "left_padding"):
                    entry.left_padding = snap.left_padding
                continue
            if snap.kind == "kv" and _is_kv_cache(entry):
                if snap.keys is not None:
                    entry.keys = snap.keys
                if snap.values is not None:
                    entry.values = snap.values
                current = int(getattr(entry, "offset", 0) or 0)
                target = int(snap.offset)
                if current > target and callable(getattr(entry, "trim", None)):
                    entry.trim(current - target)
                else:
                    entry.offset = target
                indexer = getattr(entry, "indexer", None)
                if indexer is not None:
                    indexer.keys = snap.indexer_keys

    def gdn_tensors(self) -> list[Any]:
        """Conv + recurrent (+ PLE) arrays, for equality checks."""
        out: list[Any] = []
        for snap in self.layers:
            if snap.kind == "gdn":
                out.extend(slot for slot in snap.slots if slot is not None)
        return out


def collect_gdn_tensors(cache: list[Any] | None) -> list[Any]:
    out: list[Any] = []
    if not cache:
        return out
    for entry in cache:
        if _is_arrays_cache(entry):
            out.extend(slot for slot in entry.cache if slot is not None)
    return out


def gdn_states_equal(left: list[Any], right: list[Any]) -> bool:
    import mlx.core as mx

    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if tuple(a.shape) != tuple(b.shape):
            return False
        mx.eval(a, b)
        if not bool(mx.all(a == b).item()):
            return False
    return True


# --------------------------------------------------------------------------- capture protocol (CHK-02 / CHK-04)


QWEN4_EXP_VERIFY_STRATEGIES = (
    "batched",
    "capture_commit",
    "graphbank",
    "graphbank_capture_commit",
    "target_prefix",
    "trim_commit",
)


def _skip_verify_snapshot_env() -> bool:
    return os.environ.get("MTPLX_SKIP_VERIFY_SNAPSHOT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _inner_model(model: Any) -> Any:
    owner = _text_model(model)
    return getattr(owner, "model", owner)


def _layer_is_linear(layer: Any) -> bool:
    if hasattr(layer, "is_linear"):
        return bool(layer.is_linear)
    kind = getattr(layer, "layer_type", None)
    return kind == "linear_attention" or hasattr(layer, "linear_attn")


def _inner_has_ple(inner: Any) -> bool:
    layers = getattr(inner, "layers", ()) or ()
    return any(getattr(layer, "ple", None) is not None for layer in layers)


def _alias_gdn_capture_attrs(gdn: Any) -> None:
    """Map qwen4_exp GDN names onto the gdn_forward_with_capture surface."""
    if not hasattr(gdn, "num_v_heads"):
        gdn.num_v_heads = int(getattr(gdn, "n_v"))
    if not hasattr(gdn, "num_k_heads"):
        gdn.num_k_heads = int(getattr(gdn, "n_k"))
    if not hasattr(gdn, "head_v_dim"):
        gdn.head_v_dim = int(getattr(gdn, "dv"))
    if not hasattr(gdn, "head_k_dim"):
        gdn.head_k_dim = int(getattr(gdn, "dk"))


def bind_qwen4_exp_capture_protocol(inner: Any) -> None:
    """Set fa_idx/ssm_idx and per-layer is_linear for MTPLX capture dispatch."""
    layers = list(getattr(inner, "layers", ()) or ())
    if not layers:
        raise RuntimeError("qwen4_exp inner model has no layers to bind capture markers")
    fa_idx = None
    ssm_idx = None
    for index, layer in enumerate(layers):
        is_linear = _layer_is_linear(layer)
        layer.is_linear = bool(is_linear)
        if is_linear:
            if ssm_idx is None:
                ssm_idx = index
            gdn = getattr(layer, "linear_attn", None)
            if gdn is not None:
                _alias_gdn_capture_attrs(gdn)
        elif fa_idx is None:
            fa_idx = index
    if ssm_idx is None or fa_idx is None:
        raise RuntimeError(
            "qwen4_exp capture protocol requires at least one linear_attention "
            "(GDN) layer and one full_attention layer"
        )
    inner.ssm_idx = int(ssm_idx)
    inner.fa_idx = int(fa_idx)
    inner._mtplx_qwen4_exp_capture = True


class Qwen4ExpRecurrentCache:
    """Non-trimmable GDN/PLE cache that matches the OwnedRecurrentStateCache protocol.

    ``commit_captured_prefix`` installs conv+GDN via ``replace_state([conv, gdn])``.
    Extra PLE slots (2, 3) are preserved — they are not in the 2-leaf capture
    tuple. Models with PLE must refuse capture_commit (see
    :func:`maybe_refuse_qwen4_exp_verify_lane`).
    """

    def __init__(
        self,
        size: int = 4,
        *,
        initial: list[Any] | tuple[Any, ...] | None = None,
        left_padding: Any | None = None,
        lengths: Any | None = None,
    ) -> None:
        n = max(int(size), 2)
        self.cache = [None] * n
        self.left_padding = left_padding
        self.lengths = lengths
        if initial is not None:
            self.replace_state(initial)

    @classmethod
    def from_entry(cls, entry: Any) -> "Qwen4ExpRecurrentCache":
        slots = list(getattr(entry, "cache", getattr(entry, "state", [None, None])))
        return cls(
            size=max(len(slots), 2),
            initial=slots,
            left_padding=getattr(entry, "left_padding", None),
            lengths=getattr(entry, "lengths", None),
        )

    def __getitem__(self, idx: int) -> Any:
        return self.cache[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self.cache[idx] = value

    @property
    def state(self) -> list[Any]:
        return self.cache

    @state.setter
    def state(self, value: list[Any] | tuple[Any, ...] | None) -> None:
        self.replace_state(value)

    def replace_state(self, value: list[Any] | tuple[Any, ...] | None) -> None:
        if value is None:
            self.cache = [None] * len(self.cache)
            return
        for idx, item in enumerate(value):
            if idx >= len(self.cache):
                break
            self.cache[idx] = item

    @property
    def batch_size(self) -> int:
        for c in self.cache:
            if c is not None:
                return int(c.shape[0])
        if self.left_padding is not None:
            return int(self.left_padding.size)
        elif self.lengths is not None:
            return int(self.lengths.size)
        return 1

    def filter(self, batch_indices: Any) -> None:
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other: Any) -> None:
        import mlx.core as mx

        a_batch = self.batch_size
        b_batch = getattr(other, "batch_size", 1)

        def cat(a: Any, b: Any) -> Any:
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype

            if shape is None:
                return None

            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)

            return mx.concatenate([a, b])

        other_cache = getattr(other, "cache", getattr(other, "state", []))
        self.cache = [cat(c, o) for c, o in zip(self.cache, other_cache)]
        self.left_padding = cat(self.left_padding, getattr(other, "left_padding", None))
        self.lengths = cat(self.lengths, getattr(other, "lengths", None))

    def extract(self, idx: int) -> "Qwen4ExpRecurrentCache":
        cache = Qwen4ExpRecurrentCache(len(self.cache))
        cache.cache = [c[idx : idx + 1] if c is not None else None for c in self.cache]
        cache.left_padding = (
            self.left_padding[idx : idx + 1]
            if self.left_padding is not None
            else None
        )
        cache.lengths = (
            self.lengths[idx : idx + 1]
            if self.lengths is not None
            else None
        )
        return cache

    def prepare(self, lengths=None, left_padding=None, **kwargs) -> None:
        import mlx.core as mx

        if lengths is not None:
            self.lengths = (
                mx.array(lengths) if not isinstance(lengths, mx.array) else lengths
            )
        if left_padding is not None:
            self.left_padding = (
                mx.array(left_padding)
                if not isinstance(left_padding, mx.array)
                else left_padding
            )

    def finalize(self) -> None:
        self.lengths = None
        self.left_padding = None

    @classmethod
    def merge(cls, caches):
        import mlx.core as mx

        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)

        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            return cache

        for e in range(n_state):
            c_init = next((c[e] for c in caches if c[e] is not None), None)
            if c_init is None:
                continue
            shape = list(c_init.shape)
            shape[0] = B
            cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is None:
                    continue
                cache[e][i : i + 1] = caches[i][e]
        return cache

    @property
    def nbytes(self) -> int:
        total = 0
        for c in self.cache:
            if c is not None:
                total += int(c.nbytes)
        return total

    def is_trimmable(self) -> bool:
        return False

    def advance(self, n: int) -> None:
        if self.lengths is not None:
            self.lengths -= n
        if self.left_padding is not None:
            self.left_padding -= n

    def empty(self) -> bool:
        return all(item is None for item in self.cache)

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def meta_state(self) -> tuple[str, str]:
        return ("qwen4_exp_recurrent", str(len(self.cache)))

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        del value

    def make_mask(self, n: int):
        import mlx.core as mx

        if self.left_padding is not None:
            pos = mx.arange(n)
            return pos >= self.left_padding[:, None]
        if self.lengths is not None:
            pos = mx.arange(n)
            return pos < self.lengths[:, None]
        return None


def _trim_qsa_indexer(entry: Any, n: int) -> None:
    indexer = getattr(entry, "indexer", None)
    if indexer is None:
        return
    if hasattr(indexer, "trim") and callable(indexer.trim):
        indexer.trim(n)
    else:
        keys = getattr(indexer, "keys", None)
        if keys is None:
            return
        keep = int(keys.shape[1]) - int(n)
        indexer.keys = keys[:, :keep, :] if keep > 0 else None
    reuse = getattr(entry, "_sparse_index_reuse", None)
    if reuse is not None and hasattr(reuse, "reset"):
        reuse.reset()


def _install_indexer_aware_trim(entry: Any) -> Any:
    """KV cache operations must also update the QSA indexer; otherwise batch operations desync them."""
    if entry is None or getattr(entry, "_mtplx_indexer_trim", False):
        return entry
    if type(entry).__name__ == "_AttnCache":
        entry._mtplx_indexer_trim = True
        return entry

    orig = getattr(entry, "trim", None)
    if callable(orig):
        def trim(n: int, _orig=orig, _entry=entry):
            trimmed = _orig(n)
            dropped = int(n if trimmed is None else trimmed)
            if dropped:
                _trim_qsa_indexer(_entry, dropped)
            else:
                reuse = getattr(_entry, "_sparse_index_reuse", None)
                if reuse is not None and hasattr(reuse, "reset"):
                    reuse.reset()
            return trimmed

        entry.trim = trim

    orig_filter = getattr(entry, "filter", None)
    if callable(orig_filter):
        def filter(batch_indices: Any, _orig=orig_filter, _entry=entry):
            res = _orig(batch_indices)
            indexer = getattr(_entry, "indexer", None)
            if indexer is not None and hasattr(indexer, "filter"):
                indexer.filter(batch_indices)
                if getattr(_entry, "left_padding", None) is not None:
                    indexer.left_padding = _entry.left_padding
            reuse = getattr(_entry, "_sparse_index_reuse", None)
            if reuse is not None and hasattr(reuse, "reset"):
                reuse.reset()
            return res

        entry.filter = filter

    orig_extend = getattr(entry, "extend", None)
    if callable(orig_extend):
        def extend(other: Any, _orig=orig_extend, _entry=entry):
            res = _orig(other)
            indexer = getattr(_entry, "indexer", None)
            other_indexer = getattr(other, "indexer", None)
            if indexer is not None and hasattr(indexer, "extend"):
                indexer.extend(other_indexer)
            reuse = getattr(_entry, "_sparse_index_reuse", None)
            if reuse is not None and hasattr(reuse, "reset"):
                reuse.reset()
            return res

        entry.extend = extend

    orig_extract = getattr(entry, "extract", None)
    if callable(orig_extract):
        def extract(idx: int, _orig=orig_extract, _entry=entry):
            res = _orig(idx)
            indexer = getattr(_entry, "indexer", None)
            if indexer is not None and hasattr(indexer, "extract"):
                res.indexer = indexer.extract(idx)
                _install_indexer_aware_trim(res)
            return res

        entry.extract = extract

    entry._mtplx_indexer_trim = True
    return entry


def wrap_qwen4_exp_cache(cache: list[Any] | None, inner: Any) -> list[Any]:
    if not cache:
        return cache if cache is not None else []
    layers = list(getattr(inner, "layers", ()) or ())
    out: list[Any] = []
    for idx, entry in enumerate(cache):
        layer = layers[idx] if idx < len(layers) else None
        if entry is None:
            out.append(None)
            continue
        if isinstance(entry, Qwen4ExpRecurrentCache):
            out.append(entry)
            continue
        if layer is not None and _layer_is_linear(layer):
            out.append(Qwen4ExpRecurrentCache.from_entry(entry))
            continue
        if _is_arrays_cache(entry) and not _is_kv_cache(entry):
            out.append(Qwen4ExpRecurrentCache.from_entry(entry))
            continue
        if _is_kv_cache(entry):
            out.append(_install_indexer_aware_trim(entry))
            continue
        out.append(entry)
    return out


def _cache_is_primed(cache: list[Any] | None) -> bool:
    if not cache:
        return False
    for entry in cache:
        if entry is None:
            continue
        offset = getattr(entry, "offset", None)
        if offset is not None:
            try:
                if int(offset) > 0:
                    return True
            except Exception:
                pass
        if _is_arrays_cache(entry) and any(slot is not None for slot in entry.cache):
            return True
    return False


def _sequential_owner_forward(
    owner: Any,
    inputs,
    cache,
    *,
    return_hidden: bool,
    hidden_variant: str | None,
    input_embeddings=None,
    **kwargs,
):
    import mlx.core as mx

    del hidden_variant, input_embeddings
    emit_logits = kwargs.get("emit_logits", True)
    logits_keep = kwargs.get("logits_keep", None)
    length = int(inputs.shape[1])
    logits_steps = []
    hidden_steps = []
    last_logits = None
    last_hidden = None
    for t in range(length):
        step = inputs[:, t : t + 1]
        if return_hidden:
            last_logits, last_hidden = owner(
                step, cache=cache, return_hidden=True, emit_logits=emit_logits
            )
            if emit_logits and last_logits is not None:
                logits_steps.append(last_logits)
            hidden_steps.append(last_hidden)
        else:
            last_logits = owner(
                step, cache=cache, return_hidden=False, emit_logits=emit_logits
            )
            if emit_logits and last_logits is not None:
                logits_steps.append(last_logits)
    logits = mx.concatenate(logits_steps, axis=1) if emit_logits and logits_steps else None
    if logits is not None and logits_keep is not None:
        logits = logits[:, -max(1, int(logits_keep)) :, :]
    if not return_hidden:
        return logits
    hidden = mx.concatenate(hidden_steps, axis=1)
    return logits, hidden


def forward_with_qwen4_exp_gdn_capture(
    model: Any,
    inputs,
    cache=None,
    return_hidden: bool = False,
    *,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
):
    """Per-position conv+GDN capture in the commit_captured_prefix layout.

    Qwen4_exp DecoderLayer is not Qwen3Next-shaped (hyper-connections, QSA,
    optional PLE), so stock ``forward_with_gdn_capture`` cannot walk it.
    Verify windows are run token-by-token so QSA cannot leak uncommitted
    drafts; after each step the live GDN conv/recurrent leaves are stacked
    as ``conv_states`` / ``states`` with a time axis.
    """
    import mlx.core as mx

    del capture_backend, hidden_variant
    owner = _text_model(model)
    inner = _inner_model(model)
    if cache is None:
        cache = owner.make_cache()
    length = int(inputs.shape[1])
    logits_steps = []
    hidden_steps = []
    conv_tapes: dict[int, list] = {}
    state_tapes: dict[int, list] = {}
    layers = list(inner.layers)
    for t in range(length):
        step = inputs[:, t : t + 1]
        logits, hidden = owner(step, cache=cache, return_hidden=True)
        logits_steps.append(logits)
        hidden_steps.append(hidden)
        for idx, (layer, entry) in enumerate(zip(layers, cache)):
            if entry is None or not getattr(layer, "is_linear", False):
                continue
            conv = entry[0]
            state = entry[1]
            if conv is None or state is None:
                continue
            conv_tapes.setdefault(idx, []).append(mx.expand_dims(conv, axis=1))
            state_tapes.setdefault(idx, []).append(mx.expand_dims(state, axis=1))
    logits = mx.concatenate(logits_steps, axis=1)
    hidden_out = mx.concatenate(hidden_steps, axis=1)
    captures: dict[int, dict[str, Any]] = {}
    for idx in conv_tapes:
        captures[idx] = {
            "conv_states": mx.concatenate(conv_tapes[idx], axis=1),
            "states": mx.concatenate(state_tapes[idx], axis=1),
        }
    if return_hidden:
        return logits, hidden_out, captures
    return logits, captures


def maybe_refuse_qwen4_exp_verify_lane(
    model: Any,
    verify_strategy: str,
    *,
    skip_snapshot: bool | None = None,
) -> None:
    """Refuse product lanes that would silently corrupt GDN or fail mid-request."""
    inner = _inner_model(model)
    if not getattr(inner, "_mtplx_qwen4_exp_capture", False):
        return
    strategy = str(verify_strategy or "").strip().lower()
    skip = _skip_verify_snapshot_env() if skip_snapshot is None else bool(skip_snapshot)
    if _inner_has_ple(inner) and strategy in {
        "capture_commit",
        "graphbank_capture_commit",
    }:
        raise RuntimeError(
            "qwen4_exp refuses capture_commit: PLE conv/n-gram slots are not "
            "in commit_captured_prefix's 2-leaf replace_state protocol"
        )
    if strategy in {"trim_commit", "target_prefix"}:
        raise RuntimeError(
            "qwen4_exp refuses trim-only verify_strategy="
            f"{strategy!r}: GDN recurrent cache is not trimmable; "
            "use capture_commit (or batched with a verify snapshot)"
        )
    if strategy in {"graphbank", "graphbank_capture_commit"}:
        raise RuntimeError(
            "qwen4_exp refuses compiled graphbank verify_strategy="
            f"{strategy!r}: the hybrid GDN+QSA capture graph is not a "
            "verified lane"
        )
    if strategy == "batched" and skip:
        raise RuntimeError(
            "qwen4_exp refuses batched verify with MTPLX_SKIP_VERIFY_SNAPSHOT=1: "
            "GDN state cannot be restored without capture_commit or a snapshot"
        )
    if strategy not in {"batched", "capture_commit"}:
        raise RuntimeError(
            f"qwen4_exp refuses verify_strategy={strategy!r}; "
            "supported: capture_commit, or batched with a verify snapshot"
        )


def qwen4_exp_product_verify_strategy(model: Any) -> str | None:
    """Product MTP lane that :func:`maybe_refuse_qwen4_exp_verify_lane` accepts.

    House ``ask``/quickstart default to Qwen3Next ``capture_commit`` plus the
    profile ``MTPLX_SKIP_VERIFY_SNAPSHOT=1``. On qwen4_exp that is either
    refused (PLE conv/n-gram is not in the 2-leaf replace_state protocol) or
    exact-AR (no PLE, skip-snapshot capture_commit). Real speculative decode
    is batched verify with a live snapshot. Returns None for other models.
    """
    if model is None:
        return "batched"
    if isinstance(model, (str, Path)):
        return "batched"
    inner = _inner_model(model)
    if not getattr(inner, "_mtplx_qwen4_exp_capture", False):
        return None
    return "batched"


@contextlib.contextmanager
def qwen4_exp_product_verify_env(model: Any = None):
    """Yield the product verify_strategy, forcing a live snapshot for qwen4_exp."""
    strategy = qwen4_exp_product_verify_strategy(model)
    if strategy is None:
        yield "capture_commit"
        return
    key = "MTPLX_SKIP_VERIFY_SNAPSHOT"
    previous = os.environ.get(key)
    os.environ[key] = "0"
    try:
        yield strategy
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def cache_entry_rollback_protocol(entry: Any) -> str:
    """Return 'trimmable' or 'replace_state' — or raise if the entry is a shim."""
    if entry is None:
        return "empty"
    trimmable = False
    checker = getattr(entry, "is_trimmable", None)
    if callable(checker):
        trimmable = bool(checker())
    if trimmable:
        trim = getattr(entry, "trim", None)
        if not callable(trim) or getattr(entry, "offset", None) is None:
            raise RuntimeError(
                "qwen4_exp cache entry claims is_trimmable=True without offset/trim"
            )
        return "trimmable"
    if not hasattr(entry, "state"):
        raise RuntimeError("qwen4_exp non-trimmable cache entry has no .state")
    if not hasattr(entry, "meta_state"):
        raise RuntimeError("qwen4_exp non-trimmable cache entry has no .meta_state")
    if not callable(getattr(entry, "replace_state", None)):
        raise RuntimeError(
            "qwen4_exp non-trimmable cache entry has no replace_state "
            "(ArraysCache is not capture-commit compatible)"
        )
    return "replace_state"


# --------------------------------------------------------------------------- MTP module + inject


def _mtp_text_args(args: Any, config: dict[str, Any], n_layers: int) -> Any:
    mtp_cfg = _mtp_block_config(config)
    layer_types = list(mtp_cfg.get("layer_types") or ["full_attention"] * n_layers)
    if len(layer_types) < n_layers:
        layer_types = layer_types + ["full_attention"] * (n_layers - len(layer_types))
    rope_theta = float(mtp_cfg.get("rope_theta") or getattr(args, "rope_theta", 10_000_000.0))
    updates = {
        "layer_types": layer_types[:n_layers],
        "ple_layer_ids": [],
        "rope_theta": rope_theta,
        "num_hidden_layers": n_layers,
    }
    try:
        return replace(args, **{k: v for k, v in updates.items() if hasattr(args, k)})
    except Exception:
        for key, value in updates.items():
            if hasattr(args, key):
                setattr(args, key, value)
        return args


def _make_mtp_module(args: Any, n_layers: int, qwen4: Any):
    import mlx.nn as nn

    DecoderLayer = qwen4.DecoderLayer
    RotaryEmbedding = qwen4.RotaryEmbedding
    GatedResidual = qwen4.GatedResidual
    RMSNorm = qwen4.RMSNorm

    rotary_dim = int(args.head_dim * getattr(args, "partial_rotary_factor", 0.25))

    class _Qwen4ExpMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_fc_norm_embedding = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            # vLLM Qwen4ExpMultiTokenPredictor: GemmaRMSNorm(H*hc) — one
            # statistic over the flattened multi-stream, not per-branch.
            self.pre_fc_norm_hidden = RMSNorm(
                args.hidden_size * int(args.hc_count),
                eps=args.rms_norm_eps,
            )
            self.fc_embedding = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
            self.fc_hidden = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
            inner_rope = RotaryEmbedding(
                rotary_dim, args.rope_theta, getattr(args, "rope_parameters", None)
            )
            self.rope_shift = 0
            mtp_holder: list[Any] = []

            class _ShiftedRope:
                """vLLM draft RoPE uses target sequence positions, not cache-local 0..k."""

                @property
                def dim(self):
                    return inner_rope.dim

                @property
                def freqs(self):
                    return inner_rope.freqs

                @property
                def mscale(self):
                    return inner_rope.mscale

                @property
                def position_shift(self):
                    return int(getattr(mtp_holder[0], "rope_shift", 0) or 0)

                def __call__(self, positions):
                    shift = self.position_shift
                    if shift:
                        positions = positions + shift
                    return inner_rope(positions)

            self.rope = _ShiftedRope()
            mtp_holder.append(self)
            self.layers = [DecoderLayer(args, layer_idx=i) for i in range(n_layers)]
            self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
            self.hc = int(args.hc_count)
            self.hidden_size = int(args.hidden_size)
            self._sparse_reuse: list[SparseIndexReuse] = []
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                attn.rope = self.rope
                attn.indexer = None

        def install_sparse_reuse(self) -> None:
            self._sparse_reuse = []
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                indexer = getattr(attn, "indexer", None)
                if indexer is None:
                    continue
                if isinstance(indexer, SparseIndexReuse):
                    self._sparse_reuse.append(indexer)
                    continue
                wrapper = SparseIndexReuse(indexer)
                attn.indexer = wrapper
                self._sparse_reuse.append(wrapper)

        def reset_sparse_reuse(self) -> None:
            for wrapper in self._sparse_reuse:
                wrapper.reset()

        def set_sparse_reuse(self, enabled: bool) -> None:
            for wrapper in self._sparse_reuse:
                wrapper.enabled = bool(enabled)

    return _Qwen4ExpMTP()


def _make_attn_cache(qwen4: Any):
    cls = getattr(qwen4, "_AttnCache", None) or getattr(qwen4, "AttnCache", None)
    if cls is not None:
        return cls()
    from mlx_lm.models.cache import KVCache

    cache = KVCache()
    idx_cls = getattr(qwen4, "_IndexerCache", None)
    if idx_cls is not None:
        cache.indexer = idx_cls()
    return cache


def inject_qwen4_exp_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: Any | None = None,
    *,
    allow_random_init: bool = False,
) -> bool:
    """Attach the qwen4_exp MTP head and the draft/verify/rollback surface."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import ArraysCache

    if not is_qwen4_exp_mtp_config(config) and not (
        allow_random_init and is_qwen4_exp_config(config)
    ):
        return False

    qwen4 = import_qwen4_exp()
    model_path = Path(model_path) if model_path else Path(".")
    n_layers = max(_num_mtp_layers(config), 1)
    args = _text_args(model)
    if args is None:
        TextArgs = getattr(qwen4, "TextArgs", None)
        if TextArgs is None:
            raise RuntimeError("qwen4_exp TextArgs is not available")
        args = TextArgs.from_dict(text_config(config))
    mtp_args = _mtp_text_args(args, config, n_layers)

    text_model = _text_model(model)
    embedded = (
        getattr(model, "_mtp_weights", None)
        or getattr(text_model, "_mtp_weights", None)
        or getattr(model, "mtp_weights", None)
        or getattr(text_model, "mtp_weights", None)
    )
    if embedded:
        weights = _process_raw_mtp_weights(embedded)
        if hasattr(model, "_mtp_weights"):
            model._mtp_weights = {}
        if hasattr(text_model, "_mtp_weights"):
            text_model._mtp_weights = {}
    else:
        weights = _load_mtp_weights(_candidate_weight_files(model_path, config))
    if not weights and not allow_random_init:
        logger.warning("[Qwen4Exp MTP inject] no mtp.* weights found in %s", model_path)
        return False

    mtp = _make_mtp_module(mtp_args, n_layers, qwen4)
    if contract is not None:
        from .mtp_patch import _quantize_mtp_module

        if getattr(contract, "mtp_prequantized", False):
            _quantize_mtp_module(mtp, contract)
    if weights:
        mtp.load_weights(list(weights.items()), strict=False)
        from mlx.utils import tree_flatten

        param_keys = {k for k, _ in tree_flatten(mtp.parameters())}
        loaded = set(weights)
        missing = sorted(param_keys - loaded)
        extra = sorted(loaded - param_keys)
        if missing:
            logger.warning(
                "[Qwen4Exp MTP inject] %d parameter keys not in sidecar (first %s)",
                len(missing),
                missing[:8],
            )
            if not allow_random_init:
                return False
        if extra:
            logger.warning(
                "[Qwen4Exp MTP inject] %d sidecar keys unused (first %s)",
                len(extra),
                extra[:8],
            )
        if not any(k.endswith("switch_mlp.gate_up_proj.weight") for k in loaded):
            logger.warning(
                "[Qwen4Exp MTP inject] experts.gate_up_proj did not remap onto switch_mlp.gate_up_proj"
            )
    if contract is not None and not getattr(contract, "mtp_prequantized", False):
        from .mtp_patch import _quantize_mtp_module

        _quantize_mtp_module(mtp, contract)
    mtp.install_sparse_reuse()
    mx.eval(mtp.parameters())

    concat_order = getattr(contract, "concat_order", None) or "embedding_hidden"
    hidden_variant = getattr(contract, "hidden_variant", None) or "post_norm"
    text_model.mtp = mtp
    text_model._mtplx_hidden_variant = hidden_variant
    text_model._mtplx_concat_order = concat_order
    text_model._mtplx_draft_depth_max = MAX_DRAFT_DEPTH
    text_model._mtplx_draft_depth_default = DEFAULT_DRAFT_DEPTH
    text_model._mtplx_tune_candidates = TUNE_CANDIDATES

    original_class = text_model.__class__
    hc = int(getattr(mtp_args, "hc_count", 4) or 4)

    class _MTPLXQwen4ExpModel(original_class):
        def _lm_logits(self, h):
            lm = getattr(self, "lm_head", None)
            if lm is not None:
                return lm(h)
            inner = getattr(self, "model", self)
            embed = getattr(inner, "embed_tokens", None)
            if embed is not None and hasattr(embed, "as_linear"):
                return embed.as_linear(h)
            raise RuntimeError("qwen4_exp model has no lm_head / tied embedding")

        def _draft_lm_logits(self, h):
            draft_head = getattr(self, "_mtplx_draft_lm_head", None)
            if draft_head is not None:
                return draft_head(h)
            return self._lm_logits(h)

        def _embed(self, token_ids):
            inner = getattr(self, "model", self)
            return inner.embed_tokens(token_ids)

        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            **kwargs,
        ):
            # Warm-cache *short* multi-token forwards are verify windows.
            # Batching uncommitted drafts lets QSA leak future draft keys
            # through the indexer; step those so logits stay causal.
            # Prompt prefill is the opposite: generation chunks at 2048
            # committed tokens. After chunk 0 the cache is primed, so a
            # primed-only check would serialize the rest of the prompt
            # (and OOM the Metal allocator on ~4k+). Treat T larger than
            # a draft window as prefill and keep it batched.
            seq_len = int(inputs.shape[1]) if getattr(inputs, "ndim", 0) >= 2 else 1
            if (
                cache is not None
                and input_embeddings is None
                and getattr(inputs, "ndim", 0) >= 2
                and 1 < seq_len <= (MAX_DRAFT_DEPTH + 1)
                and _cache_is_primed(cache)
            ):
                return _sequential_owner_forward(
                    self,
                    inputs,
                    cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                    **kwargs,
                )
            if not return_hidden:
                return super().__call__(
                    inputs, cache=cache, input_embeddings=input_embeddings, **kwargs
                )
            mixed, hyper = self.model(
                inputs, cache, input_embeddings, return_hyper=True
            )
            emit_logits = kwargs.get("emit_logits", True)
            logits_keep = kwargs.get("logits_keep", None)
            if not emit_logits:
                return None, hyper
            out = mixed
            if logits_keep is not None:
                out = out[:, -max(1, int(logits_keep)) :, :]
            logits = self._lm_logits(out)
            return logits, hyper

        def make_cache(self):
            cache = super().make_cache()
            return wrap_qwen4_exp_cache(cache, getattr(self, "model", self))

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str | None = None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
            reuse_sparse_indices: bool | None = None,
            input_embeddings=None,
        ):
            del mtp_hidden_variant, concat_order
            layer_caches = mtp_cache if mtp_cache is not None else self.make_mtp_cache()
            if reuse_sparse_indices is not None:
                self.mtp.set_sparse_reuse(bool(reuse_sparse_indices))
            if mtp_depth is None or int(mtp_depth) <= 1:
                self.mtp.reset_sparse_reuse()
            d = int(self.mtp.hidden_size)
            expected = d * hc
            if int(hidden_states.shape[-1]) != expected:
                raise ValueError(
                    "qwen4_exp MTP pre_fc_norm_hidden expects hyper hidden "
                    f"last-dim {expected}, got {tuple(hidden_states.shape)}"
                )
            # vLLM residual_linear_shared (nvidia/mtp.py):
            #   e = fc_embedding(pre_fc_norm_embedding(embed(ids)))   # [T, H]
            #   h = pre_fc_norm_hidden(multi).view(T, hc, H)
            #   h = fc_hidden(h)   # shared Linear(H,H) per stream
            #   x = e.unsqueeze(-2) + h ; flatten to [T, hc*H]
            # Mean-collapse-then-tile was the zero-accept assembly bug.
            if input_embeddings is not None:
                e = self.mtp.fc_embedding(
                    self.mtp.pre_fc_norm_embedding(input_embeddings)
                )
            else:
                e = self.mtp.fc_embedding(
                    self.mtp.pre_fc_norm_embedding(self._embed(next_token_ids))
                )
            h = self.mtp.pre_fc_norm_hidden(hidden_states)
            h = h.reshape(*h.shape[:-1], hc, d)
            h = self.mtp.fc_hidden(h)
            x = (e[..., None, :] + h).reshape(*h.shape[:-2], expected)
            rope_shift = 0
            if position_offset is not None:
                rope_shift = int(position_offset)
            elif cache is not None:
                for entry in cache if isinstance(cache, (list, tuple)) else (cache,):
                    off = getattr(entry, "offset", None)
                    if off is not None:
                        try:
                            rope_shift = int(off)
                            break
                        except Exception:
                            pass
            if rope_shift:
                attn_cache = next(
                    (
                        c
                        for l, c in zip(self.mtp.layers, layer_caches)
                        if (
                            getattr(l, "layer_type", None) != "linear_attention"
                            and not hasattr(l, "linear_attn")
                        )
                        and c is not None
                        and getattr(c, "offset", None) is not None
                    ),
                    None,
                )
                local = int(getattr(attn_cache, "offset", 0) or 0)
                self.mtp.rope_shift = rope_shift - local
            else:
                self.mtp.rope_shift = 0
            attn_cache = next(
                (
                    c
                    for l, c in zip(self.mtp.layers, layer_caches)
                    if (
                        getattr(l, "layer_type", None) != "linear_attention"
                        and not hasattr(l, "linear_attn")
                    )
                    and c is not None
                    and getattr(c, "offset", None) is not None
                ),
                None,
            )
            mask = create_attention_mask(
                e, [attn_cache] if attn_cache is not None else None
            )
            linear_cache = next(
                (
                    c
                    for l, c in zip(self.mtp.layers, layer_caches)
                    if (
                        getattr(l, "layer_type", None) == "linear_attention"
                        or hasattr(l, "linear_attn")
                    )
                    and c is not None
                    and hasattr(c, "make_mask")
                ),
                None,
            )
            conv_mask = (
                create_ssm_mask(e, linear_cache)
                if linear_cache is not None
                else None
            )
            ids = next_token_ids
            for layer, layer_cache in zip(self.mtp.layers, layer_caches):
                idx_c = (
                    layer_cache.indexer
                    if (layer_cache is not None and hasattr(layer_cache, "indexer"))
                    else None
                )
                x = layer(
                    x,
                    self.mtp.rope,
                    mask,
                    conv_mask,
                    layer_cache,
                    idx_c,
                    ids,
                    None,
                )
            hidden = self.mtp.hyper_connection_mixer(x)
            logits = self._draft_lm_logits(hidden)
            if not return_hidden:
                return logits
            return logits, x

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
            input_embeddings=None,
        ):
            _logits, hidden = self.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                return_hidden=True,
                position_offset=position_offset,
                mtp_depth=mtp_depth,
                input_embeddings=input_embeddings,
            )
            return hidden

        def make_mtp_cache(self):
            caches = []
            for layer in self.mtp.layers:
                is_linear = (
                    getattr(layer, "layer_type", None) == "linear_attention"
                    or hasattr(layer, "linear_attn")
                )
                if is_linear:
                    caches.append(ArraysCache(4))
                else:
                    attn = getattr(layer, "self_attn", None)
                    indexer = getattr(attn, "indexer", None)
                    if indexer is not None:
                        entry = _install_indexer_aware_trim(_make_attn_cache(qwen4))
                        if isinstance(indexer, SparseIndexReuse):
                            entry._sparse_index_reuse = indexer
                        caches.append(entry)
                    else:
                        from mlx_lm.models.cache import KVCache

                        caches.append(KVCache())
            return caches

        def snapshot_state(self, cache, *, copy: bool = False) -> StateSnapshot:
            return StateSnapshot.capture(cache, copy=copy)

        def restore_state(self, cache, snapshot: StateSnapshot) -> None:
            snapshot.restore(cache)
            self.mtp.reset_sparse_reuse()

    text_model.__class__ = _MTPLXQwen4ExpModel
    bind_qwen4_exp_capture_protocol(getattr(text_model, "model", text_model))

    if getattr(model, "language_model", None) is text_model:
        model.mtp = mtp
        original_outer = model.__class__

        class _MTPLXQwen4ExpOuter(original_outer):
            def __call__(
                self,
                inputs,
                cache=None,
                return_hidden: bool = False,
                input_embeddings=None,
                hidden_variant: str | None = None,
                **kwargs,
            ):
                return self.language_model(
                    inputs,
                    cache=cache,
                    return_hidden=return_hidden,
                    input_embeddings=input_embeddings,
                    hidden_variant=hidden_variant,
                    **kwargs,
                )

            def mtp_forward(self, *args, **kwargs):
                return self.language_model.mtp_forward(*args, **kwargs)

            def mtp_update_cache(self, *args, **kwargs):
                return self.language_model.mtp_update_cache(*args, **kwargs)

            def make_mtp_cache(self):
                return self.language_model.make_mtp_cache()

            def snapshot_state(self, cache, *, copy: bool = False) -> StateSnapshot:
                return self.language_model.snapshot_state(cache, copy=copy)

            def restore_state(self, cache, snapshot: StateSnapshot) -> None:
                return self.language_model.restore_state(cache, snapshot)

            def make_cache(self):
                return self.language_model.make_cache()

        model.__class__ = _MTPLXQwen4ExpOuter

    logger.info(
        "[Qwen4Exp MTP inject] native head bound (depth max %d, default D%d, "
        "tune %s, %d tensors%s) for %s",
        MAX_DRAFT_DEPTH,
        DEFAULT_DRAFT_DEPTH,
        ",".join(TUNE_CANDIDATES),
        len(weights),
        "" if weights else ", random-init",
        model_path,
    )
    return True


def validate_qwen4_exp_mtp_support(model: Any) -> bool:
    owner = _text_model(model)
    if getattr(owner, "mtp", None) is None:
        return False
    if not getattr(owner.mtp, "layers", None):
        return False
    inner = _inner_model(model)
    if not (hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx")):
        return False
    layers = getattr(inner, "layers", ()) or ()
    if not layers or not all(hasattr(layer, "is_linear") for layer in layers):
        return False
    return callable(getattr(owner, "mtp_forward", None)) and callable(
        getattr(owner, "make_mtp_cache", None)
    )


# --------------------------------------------------------------------------- draft / verify (exact Leviathan-Chen)


def _row_to_numpy(row) -> Any:
    import numpy as np

    if hasattr(row, "tolist"):
        return np.asarray(row.tolist(), dtype=np.float64)
    return np.asarray(row, dtype=np.float64)


def _softmax(row, temperature: float = 1.0) -> Any:
    import numpy as np

    x = _row_to_numpy(row)
    t = float(temperature)
    if t > 0 and t != 1.0:
        x = x / t
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _one_hot(index: int, vocab: int) -> Any:
    import numpy as np

    p = np.zeros((vocab,), dtype=np.float64)
    p[int(index)] = 1.0
    return p


def _argmax_token(row) -> int:
    import mlx.core as mx

    return int(mx.argmax(row, axis=-1).item())


def sample_logits_row(
    row,
    *,
    temperature: float,
    rng: Any,
) -> tuple[int, Any]:
    """Return ``(token, q_distribution)``. Greedy skips large-vocab allocations."""
    token = _argmax_token(row)
    if float(temperature) <= 0:
        return token, None
    import numpy as np

    vocab = int(row.shape[-1])
    q = _softmax(row, temperature=temperature)
    token = int(rng.choice(np.arange(vocab), p=q))
    return token, q


def verify_draft_token(
    target_row,
    draft_q,
    draft_token: int,
    *,
    temperature: float,
    rng: Any,
) -> SpeculativeDecision:
    """Exact Leviathan-Chen one-token verify. Greedy uses argmax comparison."""
    if float(temperature) <= 0:
        target_token = _argmax_token(target_row)
        if target_token == int(draft_token):
            return SpeculativeDecision(True, int(draft_token), 1.0)
        return SpeculativeDecision(False, int(target_token), 0.0)
    vocab = int(target_row.shape[-1])
    target_p = _softmax(target_row, temperature=temperature)
    return verify_one_token(target_p, draft_q, int(draft_token), rng)


def draft_tokens(
    model: Any,
    hidden,
    token_id: int,
    n: int,
    *,
    temperature: float = 0.0,
    rng: Any | None = None,
    reuse_sparse_indices: bool = True,
) -> tuple[list[int], list[Any], Any]:
    """Run the MTP head for up to ``n`` draft tokens. Returns ids, q's, last hidden."""
    import mlx.core as mx
    import numpy as np

    rng = rng or np.random.default_rng()
    n = clamp_draft_depth(n)
    mtp_cache = model.make_mtp_cache()
    if hasattr(model, "mtp") and hasattr(model.mtp, "reset_sparse_reuse"):
        model.mtp.reset_sparse_reuse()
        model.mtp.set_sparse_reuse(reuse_sparse_indices)
    drafts: list[int] = []
    qs: list[Any] = []
    tok = int(token_id)
    h = hidden
    for _ in range(n):
        logits, h = model.mtp_forward(
            h,
            mx.array([[tok]], dtype=mx.int32),
            mtp_cache=mtp_cache,
            return_hidden=True,
            reuse_sparse_indices=reuse_sparse_indices,
        )
        mx.eval(logits, h)
        tok, q = sample_logits_row(logits[0, -1], temperature=temperature, rng=rng)
        drafts.append(tok)
        qs.append(q)
        h = h[:, -1:, :]
    return drafts, qs, h


def speculative_generate(
    model: Any,
    prompt_ids,
    max_tokens: int,
    *,
    draft_depth: int = DEFAULT_DRAFT_DEPTH,
    temperature: float = 0.0,
    rng: Any | None = None,
    reuse_sparse_indices: bool = True,
) -> list[int]:
    """Greedy-or-sampled decode with MTP drafts + exact verify + GDN rollback."""
    import mlx.core as mx
    import numpy as np

    rng = rng or np.random.default_rng(0)
    depth = clamp_draft_depth(draft_depth)
    cache = model.make_cache()
    logits, hidden = model(prompt_ids, cache=cache, return_hidden=True)
    mx.eval(logits, hidden)
    primary, _ = sample_logits_row(logits[0, -1], temperature=temperature, rng=rng)
    tokens = [primary]
    hidden = hidden[:, -1:, :]

    while len(tokens) < max_tokens:
        remaining = max_tokens - len(tokens)
        n = min(depth, remaining)
        drafts, draft_qs, _draft_hidden = draft_tokens(
            model,
            hidden,
            primary,
            n,
            temperature=temperature,
            rng=rng,
            reuse_sparse_indices=reuse_sparse_indices,
        )
        step_snapshots: list[StateSnapshot] = []
        step_hiddens: list[Any] = []
        accepted: list[int] = []
        correction: int | None = None
        last_logits = None

        # Sequential verify: QSA's incomplete-block tail is always-visible, so a
        # chunked [primary, *drafts] forward leaks future draft tokens into
        # earlier logits once kv_len exceeds indexer_budget. Feed one token at
        # a time; per-step snapshots eliminate redundant replay forwards on rejection.
        step_ids = [int(primary), *[int(t) for t in drafts]]
        for i, step_tok in enumerate(step_ids):
            step = mx.array([[int(step_tok)]], dtype=mx.int32)
            last_logits, last_hidden = model(step, cache=cache, return_hidden=True)
            mx.eval(last_logits, last_hidden)
            step_snapshots.append(StateSnapshot.capture(cache, copy=False))
            step_hiddens.append(last_hidden[:, -1:, :])
            if i >= len(drafts):
                break
            decision = verify_draft_token(
                last_logits[0, -1],
                draft_qs[i],
                drafts[i],
                temperature=temperature,
                rng=rng,
            )
            if decision.accepted:
                accepted.append(int(drafts[i]))
            else:
                correction = int(decision.token_id)
                # Instant zero-replay rollback: restore state to position i (after primary + accepted[:i])
                step_snapshots[i].restore(cache)
                hidden = step_hiddens[i]
                tokens.extend(accepted)
                tokens.append(int(correction))
                primary = int(correction)
                break

        if correction is None:
            tokens.extend(accepted)
            if len(tokens) < max_tokens:
                bonus, _ = sample_logits_row(
                    last_logits[0, -1], temperature=temperature, rng=rng
                )
                tokens.append(int(bonus))
                primary = int(bonus)
                hidden = step_hiddens[-1]
            continue

    return [int(t) for t in tokens[:max_tokens]]


def greedy_ar_generate(model: Any, prompt_ids, max_tokens: int) -> list[int]:
    import mlx.core as mx

    cache = model.make_cache()
    logits, hidden = model(prompt_ids, cache=cache, return_hidden=True)
    mx.eval(logits, hidden)
    tok = _argmax_token(logits[0, -1])
    tokens = [tok]
    for _ in range(max_tokens - 1):
        logits, hidden = model(
            mx.array([[tok]], dtype=mx.int32), cache=cache, return_hidden=True
        )
        mx.eval(logits, hidden)
        tok = _argmax_token(logits[0, -1])
        tokens.append(tok)
    return tokens
