"""Versioned fixed-profile KV compression API for transfer/storage adapters.

The caller owns cache registration and slot scheduling. The runtime owns
bounded codec buffers and returns CUDA events; callers must fence them before
network reads or slot reuse. No vLLM import or implicit network operation occurs.
"""

from .config import CompressionConfig
from .manager import CompressionRuntime
from .memory import build_memory_plan as _build_memory_plan
from .spec import CacheSpec

API_VERSION = 1


def build_memory_plan(config, *args, **kwargs):
    if getattr(config, "operator_profile", None) is not None:
        from .operator_runtime import build_operator_plan

        return build_operator_plan(config, *args, **kwargs)
    return _build_memory_plan(config, *args, **kwargs)


def create_runtime(*args, **kwargs):
    """Create the preferred supported runtime without eagerly importing CUDA extras."""
    plan = args[0] if args else kwargs["plan"]
    if getattr(getattr(plan, "config", None), "operator_profile", None) is not None:
        from .operator_runtime import OperatorRuntime

        return OperatorRuntime(*args, **kwargs)
    from .fused_runtime import create_runtime as _create_runtime

    return _create_runtime(*args, **kwargs)


__all__ = [
    "API_VERSION",
    "CacheSpec",
    "CompressionConfig",
    "CompressionRuntime",
    "build_memory_plan",
    "create_runtime",
]
