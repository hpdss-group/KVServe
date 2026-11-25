"""Fixed-profile KVServe adapter contract for a Mooncake Store connector.

Each Store object is split into one K object and one V object. This keeps the
runtime's bounded single-slot workspace and variable-length codec output
compatible with Store APIs that accept lists of source/destination buffers.
The Store connector must read the fixed header first, then fetch the payload
buffers described by that header.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from ...compression.runtime.representation import (
    HEADER_BYTES,
    ChunkIO,
    StoreManifest,
    _profile_digest,
)

_KEY_TAG = "@kvserve-store-v1"


@dataclass(frozen=True)
class StorePayloadBuffer:
    address: int
    nbytes: int


@dataclass(frozen=True)
class StoreWritePlan:
    key: str
    header_key: str
    kv_part: int
    header: bytes
    header_buffer: StorePayloadBuffer
    buffers: tuple[StorePayloadBuffer, ...]
    total_bytes: int
    event: Any


@dataclass(frozen=True)
class StoreReadPlan:
    """Addresses/sizes for the second Store read after fetching the header."""

    key: str
    header_key: str
    manifest: StoreManifest
    buffers: tuple[StorePayloadBuffer, ...]


def _profile_key(profile: bytes) -> str:
    return profile.hex()[:16]


class MooncakeStoreCompressionAdapter:
    """Expose KVServe runtime operations in a Store-friendly buffer contract.

    A vLLM adapter must:

    * reserve ``reserved_bytes`` before KV-cache allocation;
    * bind registered cache tensors and group indices;
    * acquire a slot before operating on it;
    * store the header and payload under their returned namespaced keys;
    * read the header first, then the buffers in ``StoreReadPlan``;
    * retain the slot until Store I/O and the returned CUDA event have finished.

    Store calls are intentionally not made here. This keeps Mooncake optional
    and leaves scheduling, retries, and completion semantics with the connector.
    """

    def __init__(self, runtime: Any):
        if getattr(runtime, "plan", None) is None:
            raise ValueError("runtime must be a planned KVServe CompressionRuntime")
        if not getattr(runtime, "slots", None):
            raise ValueError("runtime has no compression slots")
        if not callable(getattr(runtime, "data_bytes", None)):
            raise ValueError("Mooncake Store requires a runtime with decoded byte counts")
        self.runtime = runtime
        self._chunk_io = ChunkIO(runtime)
        # vLLM allocates/registers cache tensors after connector planning;
        # CompressionRuntime computes the peer profile when those caches bind.
        self._profile: bytes | None = None
        self._slot_count = len(runtime.slots)
        self._free_slots = deque(range(self._slot_count))
        self._active_slots: set[int] = set()
        self._slot_condition = threading.Condition()
        self._host_headers = [
            torch.empty(HEADER_BYTES, dtype=torch.uint8, pin_memory=True)
            for _ in range(self._slot_count)
        ]
        self.reserved_bytes = int(runtime.plan.reserved_bytes)

    @classmethod
    def from_vllm(
        cls,
        kv_cache_specs: dict[str, Any],
        compression: dict[str, Any],
        *,
        cache_layout: str,
        blocks_per_logical: int = 1,
        tp_size: int = 1,
        tp_rank: int = 0,
        num_layers: int = 0,
        device: torch.device | None = None,
    ) -> MooncakeStoreCompressionAdapter:
        """Build the optional runtime from vLLM-like cache specs by attributes.

        No vLLM import is performed. See :mod:`.vllm` for the supported
        FullAttention geometry and layer-name requirements.
        """
        from .vllm import create_runtime_from_vllm

        runtime = create_runtime_from_vllm(
            kv_cache_specs,
            compression,
            cache_layout=cache_layout,
            blocks_per_logical=blocks_per_logical,
            tp_size=tp_size,
            tp_rank=tp_rank,
            num_layers=num_layers,
            device=device,
        )
        return cls(runtime)

    def bind_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        group_indices: dict[str, int],
    ) -> None:
        self.runtime.bind_kv_caches(kv_caches, group_indices)
        self._profile = _profile_digest(self.runtime)

    def registered_memory(self) -> list[tuple[int, int]]:
        addresses, lengths = self.runtime.registered_memory()
        return list(zip(addresses, lengths, strict=True)) + [
            (header.data_ptr(), header.numel()) for header in self._host_headers
        ]

    def key_for(self, base_key: str, kv_part: int) -> str:
        """Namespace K and V objects by profile to prevent raw/compressed hits."""
        if kv_part not in (0, 1):
            raise ValueError("kv_part must be 0 (K) or 1 (V)")
        if self._profile is None:
            raise RuntimeError("Bind the vLLM KV caches before deriving Store keys")
        part = "k" if kv_part == 0 else "v"
        return f"{base_key}{_KEY_TAG}:{_profile_key(self._profile)}:{part}"

    def acquire_slot(self, timeout: float | None = None) -> int:
        """Acquire a bounded workspace slot; caller returns it after all I/O."""
        with self._slot_condition:
            if timeout is None:
                while not self._free_slots:
                    self._slot_condition.wait()
            else:
                import time

                deadline = time.monotonic() + timeout
                while not self._free_slots:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._slot_condition.wait(remaining):
                        raise TimeoutError("Timed out waiting for a KVServe Store slot")
            slot_id = self._free_slots.popleft()
            self._active_slots.add(slot_id)
            return slot_id

    def release_slot(self, slot_id: int) -> None:
        if not 0 <= slot_id < self._slot_count:
            raise ValueError(f"Invalid slot id: {slot_id}")
        with self._slot_condition:
            if slot_id not in self._active_slots:
                raise RuntimeError(f"KVServe Store slot {slot_id} is not acquired")
            self._active_slots.remove(slot_id)
            self._free_slots.append(slot_id)
            self._slot_condition.notify()

    def compress_for_store(
        self,
        slot_id: int,
        base_key: str,
        logical_ids_by_group: list[list[int]],
        kv_part: int,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> StoreWritePlan:
        """Compress one K or V sub-object and return Store buffer descriptors."""
        self._validate_slot(slot_id)
        chunk = self._chunk_io.compress(
            self._chunk_io.plan(
                slot_id, logical_ids_by_group, kv_part, valid_tokens_by_group
            )
        )
        if len(chunk.descriptor) != HEADER_BYTES:
            raise ValueError("Compressed block exceeds the Store v1 format")
        header_tensor = self._host_headers[slot_id]
        header_tensor.copy_(
            torch.frombuffer(bytearray(chunk.descriptor), dtype=torch.uint8)
        )
        buffers = tuple(
            StorePayloadBuffer(addr, size) for addr, size in chunk.buffers if size
        )
        key = self.key_for(base_key, kv_part)
        return StoreWritePlan(
            key=key,
            header_key=f"{key}:manifest",
            kv_part=kv_part,
            header=chunk.descriptor,
            header_buffer=StorePayloadBuffer(header_tensor.data_ptr(), HEADER_BYTES),
            buffers=buffers,
            total_bytes=HEADER_BYTES + sum(buffer.nbytes for buffer in buffers),
            event=chunk.event,
        )

    def header_read_target(self, slot_id: int) -> StorePayloadBuffer:
        """Return a pinned host target for the metadata-only first GET."""
        self._validate_slot(slot_id)
        return StorePayloadBuffer(self._host_headers[slot_id].data_ptr(), HEADER_BYTES)

    def read_descriptor(self, slot_id: int) -> bytes:
        self._validate_slot(slot_id)
        return bytes(self._host_headers[slot_id].tolist())

    def read_manifest(self, slot_id: int) -> StoreManifest:
        """Parse the header just fetched into ``header_read_target(slot_id)``."""
        return StoreManifest.from_bytes(self.read_descriptor(slot_id))

    def payload_read_plan(
        self,
        slot_id: int,
        base_key: str,
        kv_part: int,
        header: StoreManifest | bytes | bytearray | memoryview,
    ) -> StoreReadPlan:
        """Validate a fetched header and describe the second GET's GPU buffers."""
        self._validate_slot(slot_id)
        profile = self._require_profile()
        manifest = (
            header
            if isinstance(header, StoreManifest)
            else StoreManifest.from_bytes(header)
        )
        if manifest.kv_part != kv_part:
            raise ValueError("Store object K/V part does not match the requested key")
        if manifest.profile != profile:
            raise ValueError("Store object compression profile does not match runtime")
        buffers = self._chunk_io.receive_buffers(slot_id, manifest.lengths)
        return StoreReadPlan(
            key=self.key_for(base_key, kv_part),
            header_key=f"{self.key_for(base_key, kv_part)}:manifest",
            manifest=manifest,
            buffers=tuple(
                StorePayloadBuffer(addr, size) for addr, size in buffers if size
            ),
        )

    def decompress_from_store(
        self,
        slot_id: int,
        manifest: StoreManifest | bytes,
        logical_ids_by_group: list[list[int]],
        kv_part: int,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> Any:
        """Decompress into the bound KV cache; caller fences the event before use."""
        self._validate_slot(slot_id)
        if isinstance(manifest, bytes):
            descriptor = manifest
            manifest = StoreManifest.from_bytes(descriptor)
        else:
            descriptor = manifest.to_bytes()
        return self._chunk_io.decompress(
            self._chunk_io.plan(
                slot_id, logical_ids_by_group, kv_part, valid_tokens_by_group
            ),
            manifest.lengths,
            descriptor,
        )

    def close(self) -> None:
        """Fence and release runtime resources after Store jobs have drained."""
        with self._slot_condition:
            if len(self._free_slots) != self._slot_count:
                raise RuntimeError("Cannot close while Store slots are still in use")
        self.runtime.shutdown()

    def _validate_slot(self, slot_id: int) -> None:
        if not 0 <= slot_id < self._slot_count:
            raise ValueError(f"Invalid slot id: {slot_id}")
        with self._slot_condition:
            if slot_id not in self._active_slots:
                raise RuntimeError(f"KVServe Store slot {slot_id} is not acquired")

    def _require_profile(self) -> bytes:
        if self._profile is None:
            raise RuntimeError("Bind the vLLM KV caches before Store compression")
        return self._profile
