# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-profile adapter for KVServe's modular compression operators."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, replace

import msgpack
import torch

from kvserve_v1.compression.manager import KVCompressionAdapter
from kvserve_v1.compression.wire import CompressedWire, build_wire, restore_from_wire

from .codec import EncodedChunk
from .memory import build_memory_plan
from .pipeline import CompressionSlotBuffers
from .spec import round_up

_HEADER_LIMIT = 1 << 20
_ALIGNMENT = 256


def _capacity(plan) -> tuple[int, int]:
    raw = max(plan.layout.raw_bytes_by_part)
    # Two KV planes, their auxiliary tensors, and a bounded metadata header.
    wire = round_up(6 * raw + _HEADER_LIMIT, _ALIGNMENT)
    # Leave room for the legacy operators' temporary tensors and codec workspace.
    reserve = wire + 12 * raw + (64 << 20)
    return wire, reserve


def build_operator_plan(config, specs, **kwargs):
    ordered = sorted(specs.values(), key=lambda spec: spec.layer_index)
    indices = [s.layer_index for s in ordered]
    if not ordered or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError(
            "Modular operators require unique full-attention layer indices"
        )
    geometry = {
        (s.block_size, s.num_kv_heads, s.head_size, s.head_size_v, s.dtype)
        for s in ordered
    }
    if len(geometry) != 1 or ordered[0].head_size != ordered[0].head_size_v:
        raise ValueError("Modular operators require uniform K/V attention geometry")
    planning = replace(config, operator_profile=None, pipeline=(), split_type="layer")
    plan = build_memory_plan(planning, specs, **kwargs)
    if plan is None:
        return None
    _, reserve = _capacity(plan)
    return replace(
        plan,
        config=config,
        reserved_bytes=plan.reserved_bytes + reserve * config.slot_count,
        tp_rank=kwargs.get("tp_rank", 0),
        tp_size=kwargs.get("tp_size", 1),
    )


class _LayerMap:
    """Map compact attention slots back to model layer ids for existing operators."""

    def __init__(self, operator, layer_ids):
        self.operator = operator
        self.layer_ids = layer_ids

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def transform(self, layer_id, *args, **kwargs):
        return self.operator.transform(self.layer_ids[layer_id], *args, **kwargs)

    def inverse(self, layer_id, *args, **kwargs):
        return self.operator.inverse(self.layer_ids[layer_id], *args, **kwargs)

    def quantize(self, layer_id, *args, **kwargs):
        return self.operator.quantize(self.layer_ids[layer_id], *args, **kwargs)

    def dequantize(self, layer_id, *args, **kwargs):
        return self.operator.dequantize(self.layer_ids[layer_id], *args, **kwargs)


class OperatorRuntime:
    """Expose the existing per-layer operators through the bounded slot API.

    Each slot owns an independent manager and stream. The peer packet contains
    the existing KVServe wire metadata and tensors in one registered buffer.
    """

    def __init__(self, plan, cache_layout, blocks_per_logical, device, **kwargs):
        if kwargs.get("backend", "auto") not in ("auto", "standard"):
            raise ValueError("Modular operator profiles cannot use the fused backend")
        if kwargs.get("output_slots", 0):
            raise ValueError("Modular operators do not support leased output banks")
        if cache_layout not in ("LBHNC", "HND"):
            raise ValueError("Modular operators require LBHNC/HND KV cache layout")
        self.execution = kwargs.get("execution", "both")
        if self.execution not in ("both", "encode", "decode"):
            raise ValueError("Unknown KVServe execution capability")
        self.plan = plan
        self.config = plan.config
        self.layout = plan.layout
        self.blocks_per_logical = blocks_per_logical
        self.device = torch.device(device)
        self._parts = self.layout.parts(0)
        part = self._parts[0]
        self._heads = part.num_heads
        self._dim = part.head_size
        self._tokens_per_block = part.block_size // blocks_per_logical
        self._max_blocks = self.config.logical_blocks_per_chunk * blocks_per_logical
        self._wire_capacity, _ = _capacity(plan)
        self.bulk = torch.empty(
            self._wire_capacity * self.config.slot_count,
            dtype=torch.uint8,
            device=self.device,
        )
        self.streams = [
            torch.cuda.Stream(device=self.device) for _ in range(self.config.slot_count)
        ]
        self.slots = []
        self._dense = []
        self._managers = []
        for index in range(self.config.slot_count):
            begin = index * self._wire_capacity
            empty = torch.empty(0, dtype=torch.uint8, device=self.device)
            self.slots.append(
                CompressionSlotBuffers(
                    arena_a=self.bulk[begin : begin + self._wire_capacity],
                    arena_b=empty,
                    aux=empty,
                    block_indices=torch.empty(0, dtype=torch.int64, device=self.device),
                    completion_event=torch.cuda.Event(),
                )
            )
            self._dense.append(
                torch.empty(
                    len(self._parts),
                    2,
                    self._max_blocks,
                    self._tokens_per_block,
                    self._heads,
                    self._dim,
                    dtype=part.dtype,
                    device=self.device,
                )
            )
            adapter = KVCompressionAdapter(
                self.config.operator_profile,
                num_kv_heads=self._heads,
                head_size=self._dim,
                tp_rank=plan.tp_rank,
                tp_size=plan.tp_size,
            )
            layer_ids = tuple(part.layer_index for part in self._parts)
            for name in ("transformer", "quantizer"):
                operator = getattr(adapter._manager, name)
                if operator is not None:
                    setattr(adapter._manager, name, _LayerMap(operator, layer_ids))
            self._managers.append(adapter)
        self._caches = {}
        self._group = None

    @property
    def fa_group_indices(self):
        return self.layout.fa_group_indices

    def bind_kv_caches(self, caches, group_indices):
        layer_parts = tuple(
            replace(part, group_index=group_indices[part.layer_name])
            for part in self.layout.layer_parts
        )
        groups = frozenset(part.group_index for part in layer_parts)
        if len(groups) != 1:
            raise ValueError("Modular operators currently require one attention group")
        self.layout = replace(
            self.layout, layer_parts=layer_parts, fa_group_indices=groups
        )
        self._parts = self.layout.parts(0)
        self._group = next(iter(groups))
        self._caches = {
            part.layer_name: caches[part.layer_name] for part in self._parts
        }
        for name, cache in self._caches.items():
            pieces = (
                cache
                if isinstance(cache, tuple)
                else (
                    cache.narrow(-1, 0, self._dim),
                    cache.narrow(-1, self._dim, self._dim),
                )
            )
            if len(pieces) != 2 or any(
                value.ndim != 4
                or value.dtype != torch.bfloat16
                or tuple(value.shape[1:])
                != (self._heads, self._tokens_per_block, self._dim)
                or value.device != self.device
                or value.stride(-1) != 1
                for value in pieces
            ):
                raise ValueError(
                    f"Unsupported modular operator cache layout for {name}"
                )
        profile = asdict(self.config)
        profile.pop("slot_count", None)
        self.wire_profile = hashlib.sha256(
            json.dumps(
                {
                    "version": "operators-v1",
                    "config": profile,
                    "layers": [
                        asdict(part) | {"dtype": str(part.dtype)}
                        for part in self._parts
                    ],
                    "blocks_per_logical": self.blocks_per_logical,
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def registered_memory(self):
        return [self.bulk.data_ptr()], [self.bulk.numel()]

    def aux_bytes(self, kv_part, tokens=None):
        return 0

    def _part(self, name, kv_part):
        cache = self._caches[name]
        if isinstance(cache, tuple):
            return cache[kv_part]
        return cache.narrow(-1, kv_part * self._dim, self._dim)

    def _indices(self, logical_ids):
        values = [
            block * self.blocks_per_logical + offset
            for block in logical_ids
            for offset in range(self.blocks_per_logical)
        ]
        if len(values) > self._max_blocks:
            raise ValueError("Chunk exceeds the planned operator slot")
        return torch.tensor(values, dtype=torch.int64, device=self.device)

    @staticmethod
    def _pack(slot, wire, raw):
        tensors = [raw] if raw is not None else wire.body_chunks + wire.aux_tensors
        descriptors = []
        offset = 0
        for tensor in tensors:
            data = tensor.contiguous()
            offset = round_up(offset, max(8, data.element_size()))
            size = data.numel() * data.element_size()
            descriptors.append(
                {
                    "offset": offset,
                    "size": size,
                    "shape": list(data.shape),
                    "dtype": str(data.dtype).removeprefix("torch."),
                }
            )
            offset += size
        header = msgpack.packb(
            {
                "version": 1,
                "raw": raw is not None,
                "body_count": 1 if raw is not None else len(wire.body_chunks),
                "meta": None if raw is not None else wire.meta,
                "tensors": descriptors,
            },
            use_bin_type=True,
        )
        if len(header) > _HEADER_LIMIT:
            raise ValueError("Modular operator metadata exceeds its header limit")
        start = round_up(4 + len(header), 8)
        total = start + offset
        if total > slot.arena_a.numel():
            raise ValueError("Modular operator packet exceeds its planned slot")
        prefix = torch.frombuffer(
            bytearray(struct.pack("<I", len(header)) + header), dtype=torch.uint8
        )
        slot.arena_a[: prefix.numel()].copy_(prefix)
        for tensor, descriptor in zip(tensors, descriptors):
            data = tensor.contiguous().reshape(-1).view(torch.uint8)
            begin = start + descriptor["offset"]
            slot.arena_a[begin : begin + descriptor["size"]].copy_(
                data, non_blocking=True
            )
        return total

    @staticmethod
    def _unpack(slot, length):
        if length < 4 or length > slot.arena_a.numel():
            raise ValueError("Invalid modular operator packet size")
        size = struct.unpack("<I", bytes(slot.arena_a[:4].cpu().tolist()))[0]
        if size > _HEADER_LIMIT or 4 + size > length:
            raise ValueError("Invalid modular operator header")
        header = msgpack.unpackb(
            bytes(slot.arena_a[4 : 4 + size].cpu().tolist()),
            raw=False,
            strict_map_key=False,
        )
        if header.get("version") != 1:
            raise ValueError("Unsupported modular operator packet")
        start = round_up(4 + size, 8)
        tensors = []
        allowed = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.bool,
        }
        for item in header["tensors"]:
            dtype = getattr(torch, item["dtype"], None)
            if dtype not in allowed:
                raise ValueError("Unsupported modular operator tensor dtype")
            begin, size = start + int(item["offset"]), int(item["size"])
            if begin < start or size < 0 or begin + size > length:
                raise ValueError("Modular operator tensor exceeds packet")
            shape = tuple(int(dimension) for dimension in item["shape"])
            if any(dimension < 0 for dimension in shape):
                raise ValueError("Invalid modular operator tensor shape")
            element_size = torch.empty((), dtype=dtype).element_size()
            if begin % element_size or math.prod(shape) * element_size != size:
                raise ValueError("Invalid modular operator tensor storage")
            tensors.append(
                slot.arena_a[begin : begin + size].view(dtype).reshape(shape)
            )
        count = int(header["body_count"])
        if not 0 <= count <= len(tensors):
            raise ValueError("Invalid modular operator tensor count")
        if header["raw"]:
            if count != 1 or len(tensors) != 1:
                raise ValueError("Invalid raw modular operator packet")
            return tensors[0], None
        return None, CompressedWire(header["meta"], tensors[:count], tensors[count:])

    def compress(self, slot_id, logical_ids_by_group, kv_part, *, valid_tokens_by_group):
        if self.execution == "decode":
            raise RuntimeError("Decode-only runtime cannot encode")
        slot, stream = self.slots[slot_id], self.streams[slot_id]
        with torch.cuda.stream(stream):
            indices = self._indices(logical_ids_by_group[self._group])
            count = indices.numel()
            dense = self._dense[slot_id][:, :, :count]
            dense.zero_()
            for layer, part in enumerate(self._parts):
                dense[layer, kv_part].copy_(
                    self._part(part.layer_name, kv_part)
                    .index_select(0, indices)
                    .permute(0, 2, 1, 3)
                )
            valid = valid_tokens_by_group[self._group]
            if not 0 < valid <= count * self._tokens_per_block:
                raise ValueError("Invalid valid-token count for modular operator chunk")
            block, remainder = divmod(valid, self._tokens_per_block)
            if remainder:
                dense[:, kv_part, block, remainder:].zero_()
                block += 1
            dense[:, kv_part, block:].zero_()
            threshold = self.config.operator_profile.get("min_compress_size", 0)
            raw = (
                dense[:, kv_part].contiguous()
                if dense.numel() * dense.element_size() < threshold
                else None
            )
            packet = (
                None
                if raw is not None
                else self._managers[slot_id].compress(dense, "fixed-profile")
            )
            if packet is None and raw is None:
                raise RuntimeError("KVServe modular operator compression failed")
            wire = None if raw is not None else build_wire(packet, self._wire_capacity)
            size = self._pack(slot, wire, raw)
            slot.completion_event.record(stream)
            raw_bytes = (
                count
                * self._tokens_per_block
                * self._heads
                * self._dim
                * len(self._parts)
                * 2
            )
            return EncodedChunk(
                None, slot.completion_event, raw_bytes, payload_bytes=size
            )

    def decompress(
        self,
        slot_id,
        logical_ids_by_group,
        kv_part,
        codec_bytes,
        u8_bytes,
        *,
        valid_tokens_by_group,
    ):
        if self.execution == "encode":
            raise RuntimeError("Encode-only runtime cannot decode")
        slot, stream = self.slots[slot_id], self.streams[slot_id]
        with torch.cuda.stream(stream):
            indices = self._indices(logical_ids_by_group[self._group])
            count = indices.numel()
            expected = (
                count
                * self._tokens_per_block
                * self._heads
                * self._dim
                * len(self._parts)
                * 2
            )
            if u8_bytes != expected:
                raise ValueError("Modular operator chunk has incompatible decoded size")
            raw, wire = self._unpack(slot, codec_bytes)
            if raw is None:
                packet = restore_from_wire(wire)
                packet.metadata["device"] = str(self.device)
                for metadata in packet.chunk_metadata or ():
                    metadata["device"] = str(self.device)
                restored = self._managers[slot_id].decompress(packet)
                if isinstance(restored, list):
                    restored = torch.cat(restored)
                if restored is None:
                    raise RuntimeError("KVServe modular operator decompression failed")
                raw = restored[:, kv_part]
            expected_shape = (
                len(self._parts),
                count,
                self._tokens_per_block,
                self._heads,
                self._dim,
            )
            if tuple(raw.shape) != expected_shape or raw.dtype != torch.bfloat16:
                raise ValueError("Modular operator returned incompatible KV geometry")
            for layer, part in enumerate(self._parts):
                self._part(part.layer_name, kv_part).index_copy_(
                    0, indices, raw[layer].permute(0, 2, 1, 3)
                )
            slot.completion_event.record(stream)
            return slot.completion_event

    def shutdown(self):
        for stream in self.streams:
            stream.synchronize()
