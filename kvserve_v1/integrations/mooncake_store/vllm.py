"""Optional structural bridge from vLLM Store metadata to KVServe runtime.

There is deliberately no vLLM import here. Keeping the bridge structural lets
KVServe remain independently installable and makes unsupported cache specs
fail at setup, before the Store starts accepting work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_LAYER_INDEX = re.compile(r"(?:layers?|blocks?)[._](\d+)(?:[._]|$)")


@dataclass(frozen=True)
class _CacheSpecBridge:
    """Runtime-compatible cache geometry without importing runtime modules."""

    layer_index: int
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: Any
    head_size_v: int


def _attr(spec: Any, *names: str) -> Any:
    for name in names:
        value = getattr(spec, name, None)
        if value is not None:
            return value
    return None


def _layer_index(name: str, spec: Any) -> int:
    value = _attr(spec, "layer_index", "layer_idx", "layer_id")
    if value is not None:
        return int(value)
    match = _LAYER_INDEX.search(name)
    if match is None:
        raise ValueError(
            f"Cannot infer layer index for cache {name!r}; pass a spec with "
            "layer_index/layer_idx or use a standard layers.N name."
        )
    return int(match.group(1))


def cache_specs_from_vllm(kv_cache_specs: dict[str, Any]) -> dict[str, Any]:
    """Convert FullAttention-like vLLM specs without importing vLLM classes."""
    converted: dict[str, _CacheSpecBridge] = {}
    rejected: list[str] = []
    for name, spec in kv_cache_specs.items():
        block_size = _attr(spec, "block_size")
        num_kv_heads = _attr(spec, "num_kv_heads", "num_kv_head")
        head_size = _attr(spec, "head_size", "head_size_k")
        dtype = _attr(spec, "dtype")
        if None in (block_size, num_kv_heads, head_size, dtype):
            rejected.append(name)
            continue
        head_size_v = _attr(spec, "head_size_v") or head_size
        converted[name] = _CacheSpecBridge(
            layer_index=_layer_index(name, spec),
            block_size=int(block_size),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            dtype=dtype,
            head_size_v=int(head_size_v),
        )
    if rejected:
        raise ValueError(
            "Mooncake Store compression currently requires dense FullAttention "
            "cache specs; unsupported entries: " + ", ".join(rejected)
        )
    if not converted:
        raise ValueError("No FullAttention cache specs were provided")
    if len({spec.block_size for spec in converted.values()}) != 1:
        raise ValueError("Initial Store adapter requires one common block size")
    return converted


def create_runtime_from_vllm(
    kv_cache_specs: dict[str, Any],
    compression: dict[str, Any],
    *,
    cache_layout: str,
    blocks_per_logical: int = 1,
    tp_size: int = 1,
    tp_rank: int = 0,
    num_layers: int = 0,
    device: Any = None,
) -> Any:
    """Create a KVServe runtime from the stable attributes of Store specs."""
    import torch

    from kvserve_v1.compression.runtime import (
        CompressionConfig,
        CompressionRuntime,
        build_memory_plan,
    )

    specs = cache_specs_from_vllm(kv_cache_specs)
    extra = dict(compression)
    extra["enable-compression-pipeline"] = True
    config = CompressionConfig.from_extra_config({"compression": extra})
    if not config.enabled:
        raise ValueError("Store runtime requires compression to be enabled")
    plan = build_memory_plan(
        config,
        specs,
        blocks_per_logical=blocks_per_logical,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_layers=num_layers,
        device=device,
    )
    if plan is None:
        raise RuntimeError("KVServe did not produce a compression memory plan")
    return CompressionRuntime(
        plan,
        cache_layout,
        blocks_per_logical,
        device or torch.device("cuda", torch.cuda.current_device()),
    )
