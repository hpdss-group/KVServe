"""Mooncake Store adapter primitives backed by KVServe's fixed-profile runtime.

The module does not import vLLM or Mooncake. A vLLM Store connector supplies
cache geometry, block IDs, and Store read/write calls through this small API.
"""

from .adapter import (
    HEADER_BYTES,
    MooncakeStoreCompressionAdapter,
    StoreManifest,
    StorePayloadBuffer,
    StoreReadPlan,
    StoreWritePlan,
)
from .vllm import cache_specs_from_vllm, create_runtime_from_vllm

__all__ = [
    "HEADER_BYTES",
    "MooncakeStoreCompressionAdapter",
    "StoreManifest",
    "StorePayloadBuffer",
    "StoreReadPlan",
    "StoreWritePlan",
    "cache_specs_from_vllm",
    "create_runtime_from_vllm",
]

