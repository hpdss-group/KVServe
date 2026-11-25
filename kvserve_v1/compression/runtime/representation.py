"""Opaque KVServe chunk representation shared by transfer and storage."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, NamedTuple

BufferSpan = tuple[int, int]


_MAGIC = b"KVSSTOR1"
_HEADER = struct.Struct(">8sBBHIIII32s")
_WIDE_HEADER = struct.Struct(">8sBBHQQQQ32s")
HEADER_BYTES = _HEADER.size
# Store keeps its original persistent envelope; version 2 is for wide transfers.
_STORE_VERSION = 1
_WIDE_VERSION = 2


@dataclass(frozen=True)
class StoreManifest:
    """Versioned metadata required to restore one compressed K or V chunk."""

    kv_part: int
    codec_bytes: int
    decoded_bytes: int
    aux_bytes: int
    raw_tail_bytes: int
    profile: bytes

    def __post_init__(self) -> None:
        if self.kv_part not in (0, 1):
            raise ValueError("kv_part must be 0 (K) or 1 (V)")
        if any(
            not 0 <= length < 1 << 64 for length in (*self.lengths, self.decoded_bytes)
        ):
            raise ValueError("Compressed chunk length is outside the descriptor range")
        if len(self.profile) != 32:
            raise ValueError("profile must be a 32-byte SHA-256 digest")

    @property
    def lengths(self) -> tuple[int, int, int]:
        return self.codec_bytes, self.aux_bytes, self.raw_tail_bytes

    def to_bytes(self) -> bytes:
        # Preserve the fixed Store envelope for ordinary blocks. P/D batches
        # can contain larger spans, including raw aggregation.
        wide = max(*self.lengths, self.decoded_bytes) >= 1 << 32
        header = _WIDE_HEADER if wide else _HEADER
        return header.pack(
            _MAGIC,
            _WIDE_VERSION if wide else _STORE_VERSION,
            self.kv_part,
            0,
            self.codec_bytes,
            self.decoded_bytes,
            self.aux_bytes,
            self.raw_tail_bytes,
            self.profile,
        )

    @classmethod
    def from_bytes(cls, value: bytes | bytearray | memoryview) -> StoreManifest:
        if len(value) == HEADER_BYTES:
            header, expected_version = _HEADER, _STORE_VERSION
        elif len(value) == _WIDE_HEADER.size:
            header, expected_version = _WIDE_HEADER, _WIDE_VERSION
        else:
            raise ValueError("Invalid KVServe descriptor length")
        magic, version, kv_part, flags, codec, decoded, aux, tail, profile = (
            header.unpack(value)
        )
        if magic != _MAGIC:
            raise ValueError("Store object is not a KVServe-compressed value")
        if version != expected_version:
            raise ValueError(f"Unsupported KVServe descriptor version: {version}")
        if flags:
            raise ValueError("Store header contains unsupported flags")
        return cls(kv_part, codec, decoded, aux, tail, profile)


def _profile_digest(runtime: Any) -> bytes:
    profile = getattr(runtime, "wire_profile", None)
    if not isinstance(profile, str) or len(profile) != 64:
        raise RuntimeError(
            "Bind the KV caches before using compressed chunks; the runtime must "
            "expose its 64-character wire_profile."
        )
    try:
        return bytes.fromhex(profile)
    except ValueError as exc:
        raise RuntimeError("Runtime wire_profile is not a SHA-256 digest") from exc


class ChunkPlan(NamedTuple):
    slot_id: int
    blocks: list[list[int]]
    kv_part: int
    valid_tokens: dict[int, int]


class CompressedChunk(NamedTuple):
    buffers: tuple[BufferSpan, ...]
    descriptor: bytes
    event: Any


class ChunkIO:
    """Bind fixed-profile execution to already-reserved runtime slots."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    @property
    def wire_profile(self) -> str:
        return f"kvserve:{_profile_digest(self.runtime).hex()}"

    def plan(
        self,
        slot_id: int,
        blocks: list[list[int]],
        kv_part: int,
        valid_tokens: dict[int, int],
    ) -> ChunkPlan:
        if kv_part not in (0, 1):
            raise ValueError("kv_part must be 0 (K) or 1 (V)")
        return ChunkPlan(slot_id, blocks, kv_part, valid_tokens)

    def slot_buffers(self, slot_id: int) -> tuple[BufferSpan, ...]:
        slot = self.runtime.slots[slot_id]
        return tuple(
            (buf.data_ptr(), buf.numel()) if buf is not None else (0, 0)
            for buf in (slot.arena_a, slot.aux, slot.raw_tail)
        )

    def receive_buffers(
        self, slot_id: int, lengths: tuple[int, ...]
    ) -> tuple[BufferSpan, ...]:
        buffers = self.slot_buffers(slot_id)
        if len(lengths) != len(buffers):
            raise ValueError("Compressed buffer count does not match the slot")
        spans = []
        for (address, capacity), length in zip(buffers, lengths, strict=True):
            if not 0 <= length <= capacity or (length and not address):
                raise ValueError("Compressed buffer exceeds the slot capacity")
            spans.append((address, length))
        return tuple(spans)

    def compress(self, plan: ChunkPlan) -> CompressedChunk:
        slot, blocks, part, tokens = plan
        encoded = self.runtime.compress(slot, blocks, part, valid_tokens_by_group=tokens)
        # Variable codec lengths and borrowed buffers are ready after this fence.
        encoded.event.synchronize()
        manifest = StoreManifest(
            part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            self.runtime.aux_bytes(part, tokens),
            encoded.raw_tail_bytes,
            _profile_digest(self.runtime),
        )
        buffers = self.receive_buffers(slot, manifest.lengths)
        if manifest.raw_tail_bytes and encoded.raw_tail_addr != buffers[-1][0]:
            raise ValueError("Runtime returned a tail outside its registered slot")
        return CompressedChunk(buffers, manifest.to_bytes(), encoded.event)

    def decompress(
        self, plan: ChunkPlan, lengths: tuple[int, ...], descriptor: bytes
    ) -> Any:
        slot, blocks, part, tokens = plan
        manifest = StoreManifest.from_bytes(descriptor)
        if manifest.kv_part != part or manifest.profile != _profile_digest(
            self.runtime
        ):
            raise ValueError("Incompatible compressed chunk profile or K/V part")
        if tuple(lengths) != manifest.lengths:
            raise ValueError("Compressed buffer lengths do not match the descriptor")
        self.receive_buffers(slot, lengths)
        if manifest.aux_bytes != self.runtime.aux_bytes(part, tokens):
            raise ValueError("Compressed auxiliary bytes do not match cache geometry")
        tail = self.runtime.slots[slot].raw_tail
        expected_tail = (
            self.runtime.layout.raw_tail_bytes(part, tokens)
            if tail is not None and tail.numel()
            else 0
        )
        if manifest.raw_tail_bytes != expected_tail:
            raise ValueError("Compressed tail bytes do not match cache geometry")
        return self.runtime.decompress(
            slot,
            blocks,
            part,
            manifest.codec_bytes,
            manifest.decoded_bytes,
            valid_tokens_by_group=tokens,
        )
