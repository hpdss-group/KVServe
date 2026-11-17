"""Fixed-profile KVServe adapter contract for a Mooncake Store connector.

Each Store object is split into one K object and one V object. This keeps the
runtime's bounded single-slot workspace and variable-length codec output
compatible with Store APIs that accept lists of source/destination buffers.
The Store connector must read the fixed header first, then fetch the payload
buffers described by that header.
"""

from __future__ import annotations

import struct
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

_MAGIC = b"KVSSTOR1"
_HEADER = struct.Struct(">8sBBHIIII32s")
HEADER_BYTES = _HEADER.size
_KEY_TAG = "@kvserve-store-v1"
# Keep the Store envelope independently importable. Bump together with the
# public compression runtime API when its payload contract changes.
_API_VERSION = 1


@dataclass(frozen=True)
class StoreManifest:
    """Fixed-size metadata required to restore one compressed K or V object."""

    kv_part: int
    codec_bytes: int
    decoded_bytes: int
    aux_bytes: int
    raw_tail_bytes: int
    profile: bytes

    def __post_init__(self) -> None:
        if self.kv_part not in (0, 1):
            raise ValueError("kv_part must be 0 (K) or 1 (V)")
        for field in ("codec_bytes", "decoded_bytes", "aux_bytes", "raw_tail_bytes"):
            if not 0 <= getattr(self, field) <= 0xFFFFFFFF:
                raise ValueError(f"{field} is outside the Store header range")
        if len(self.profile) != 32:
            raise ValueError("profile must be a 32-byte SHA-256 digest")

    def encode(self) -> bytes:
        return _HEADER.pack(
            _MAGIC,
            _API_VERSION,
            self.kv_part,
            0,
            self.codec_bytes,
            self.decoded_bytes,
            self.aux_bytes,
            self.raw_tail_bytes,
            self.profile,
        )

    @classmethod
    def decode(cls, value: bytes | bytearray | memoryview) -> StoreManifest:
        if len(value) != HEADER_BYTES:
            raise ValueError(f"Store header must be exactly {HEADER_BYTES} bytes")
        magic, version, kv_part, flags, codec, decoded, aux, tail, profile = (
            _HEADER.unpack(value)
        )
        if magic != _MAGIC:
            raise ValueError("Store object is not a KVServe-compressed value")
        if version != _API_VERSION:
            raise ValueError(f"Unsupported KVServe Store API version: {version}")
        if flags:
            raise ValueError("Store header contains unsupported flags")
        return cls(kv_part, codec, decoded, aux, tail, profile)


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


def _profile_digest(runtime: Any) -> bytes:
    profile = getattr(runtime, "wire_profile", None)
    if not isinstance(profile, str) or len(profile) != 64:
        raise RuntimeError(
            "Bind the KV caches before creating Store keys; the runtime must "
            "expose its 64-character wire_profile."
        )
    try:
        return bytes.fromhex(profile)
    except ValueError as exc:
        raise RuntimeError("Runtime wire_profile is not a SHA-256 digest") from exc


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
        self.runtime = runtime
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

    def encode_for_store(
        self,
        slot_id: int,
        base_key: str,
        logical_ids_by_group: list[list[int]],
        kv_part: int,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> StoreWritePlan:
        """Encode one K or V sub-object and return Store buffer descriptors."""
        self._validate_slot(slot_id)
        profile = self._require_profile()
        if kv_part not in (0, 1):
            raise ValueError("kv_part must be 0 (K) or 1 (V)")
        encoded = self.runtime.encode(
            slot_id,
            logical_ids_by_group,
            kv_part,
            valid_tokens_by_group=valid_tokens_by_group,
        )
        encoded.event.synchronize()
        codec_bytes = encoded.codec_bytes
        aux_bytes = self.runtime.aux_bytes(kv_part, valid_tokens_by_group)
        manifest = StoreManifest(
            kv_part=kv_part,
            codec_bytes=codec_bytes,
            decoded_bytes=encoded.u8_bytes,
            aux_bytes=aux_bytes,
            raw_tail_bytes=encoded.raw_tail_bytes,
            profile=profile,
        )
        header_tensor = self._host_headers[slot_id]
        header_tensor.copy_(torch.frombuffer(bytearray(manifest.encode()), dtype=torch.uint8))
        buffers: list[StorePayloadBuffer] = []
        if codec_bytes:
            buffers.append(
                StorePayloadBuffer(self.runtime.slots[slot_id].arena_a.data_ptr(), codec_bytes)
            )
        if aux_bytes:
            buffers.append(
                StorePayloadBuffer(self.runtime.slots[slot_id].aux.data_ptr(), aux_bytes)
            )
        if manifest.raw_tail_bytes:
            if not encoded.raw_tail_addr:
                raise RuntimeError("Runtime returned raw-tail bytes without a buffer")
            buffers.append(
                StorePayloadBuffer(encoded.raw_tail_addr, manifest.raw_tail_bytes)
            )
        key = self.key_for(base_key, kv_part)
        return StoreWritePlan(
            key=key,
            header_key=f"{key}:manifest",
            kv_part=kv_part,
            header=manifest.encode(),
            header_buffer=StorePayloadBuffer(header_tensor.data_ptr(), HEADER_BYTES),
            buffers=tuple(buffers),
            total_bytes=HEADER_BYTES + sum(buffer.nbytes for buffer in buffers),
            event=encoded.event,
        )

    def header_read_target(self, slot_id: int) -> StorePayloadBuffer:
        """Return a pinned host target for the metadata-only first GET."""
        self._validate_slot(slot_id)
        return StorePayloadBuffer(self._host_headers[slot_id].data_ptr(), HEADER_BYTES)

    def decode_header(self, slot_id: int) -> StoreManifest:
        """Parse the header just fetched into ``header_read_target(slot_id)``."""
        self._validate_slot(slot_id)
        return StoreManifest.decode(bytes(self._host_headers[slot_id].tolist()))

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
            header if isinstance(header, StoreManifest) else StoreManifest.decode(header)
        )
        if manifest.kv_part != kv_part:
            raise ValueError("Store object K/V part does not match the requested key")
        if manifest.profile != profile:
            raise ValueError("Store object compression profile does not match runtime")
        slot = self.runtime.slots[slot_id]
        if manifest.codec_bytes > slot.arena_a.numel():
            raise ValueError("Store payload exceeds the runtime codec buffer")
        if manifest.aux_bytes > slot.aux.numel():
            raise ValueError("Store auxiliary data exceeds the runtime buffer")
        if manifest.raw_tail_bytes and (
            slot.raw_tail is None or manifest.raw_tail_bytes > slot.raw_tail.numel()
        ):
            raise ValueError("Store raw tail exceeds the runtime buffer")
        buffers: list[StorePayloadBuffer] = []
        if manifest.codec_bytes:
            buffers.append(StorePayloadBuffer(slot.arena_a.data_ptr(), manifest.codec_bytes))
        if manifest.aux_bytes:
            buffers.append(StorePayloadBuffer(slot.aux.data_ptr(), manifest.aux_bytes))
        if manifest.raw_tail_bytes:
            buffers.append(
                StorePayloadBuffer(slot.raw_tail.data_ptr(), manifest.raw_tail_bytes)
            )
        return StoreReadPlan(
            key=self.key_for(base_key, kv_part),
            header_key=f"{self.key_for(base_key, kv_part)}:manifest",
            manifest=manifest,
            buffers=tuple(buffers),
        )

    def decode_from_store(
        self,
        slot_id: int,
        manifest: StoreManifest,
        logical_ids_by_group: list[list[int]],
        kv_part: int,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> Any:
        """Decode into the bound KV cache; caller fences the event before use."""
        self._validate_slot(slot_id)
        if manifest.kv_part != kv_part or manifest.profile != self._require_profile():
            raise ValueError("Store manifest is incompatible with this decode request")
        expected_aux = self.runtime.aux_bytes(kv_part, valid_tokens_by_group)
        expected_decoded = self.runtime.data_bytes(kv_part, valid_tokens_by_group)
        expected_tail = self.runtime.layout.raw_tail_bytes(kv_part, valid_tokens_by_group)
        if manifest.aux_bytes != expected_aux:
            raise ValueError("Store auxiliary byte count does not match cache geometry")
        if manifest.decoded_bytes != expected_decoded:
            raise ValueError("Store decoded byte count does not match cache geometry")
        if manifest.raw_tail_bytes != expected_tail:
            raise ValueError("Store raw-tail byte count does not match cache geometry")
        return self.runtime.decode(
            slot_id,
            logical_ids_by_group,
            kv_part,
            manifest.codec_bytes,
            manifest.decoded_bytes,
            valid_tokens_by_group=valid_tokens_by_group,
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

