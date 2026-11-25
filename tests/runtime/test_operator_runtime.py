"""Fixed operator profiles through the public paged-runtime API."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from kvserve_v1.compression.runtime import (
    CacheSpec,
    ChunkIO,
    CompressionConfig,
    build_memory_plan,
    create_runtime,
)


@pytest.mark.parametrize("size,version", [(1 << 16, 1), (1 << 32, 2)])
def test_chunk_descriptor_preserves_store_and_large_transfer_lengths(size, version):
    from kvserve_v1.compression.runtime.representation import (
        HEADER_BYTES,
        StoreManifest,
    )

    manifest = StoreManifest(0, size, size, 0, 0, bytes(32))
    descriptor = manifest.to_bytes()
    assert descriptor[8] == version
    assert (len(descriptor) == HEADER_BYTES) == (version == 1)
    assert StoreManifest.from_bytes(descriptor) == manifest


def test_store_requires_decoded_byte_counts_before_allocating_buffers():
    from kvserve_v1.integrations.mooncake_store import MooncakeStoreCompressionAdapter

    runtime = SimpleNamespace(plan=object(), slots=[object()])
    with pytest.raises(ValueError, match="decoded byte counts"):
        MooncakeStoreCompressionAdapter(runtime)


def _config(profile, **kwargs):
    return CompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "operator_profile": profile,
                "logical_blocks_per_chunk": 2,
                **kwargs,
            }
        }
    )


def test_operator_profile_is_fixed_and_rejects_controller():
    profile = {"pipeline": ["codec"], "codec_config": {"codec_type": "bitpack"}}
    config = _config(profile)
    profile["pipeline"].append("quantizer")
    assert config.pipeline == ("codec",)
    assert config.operator_profile["pipeline"] == ["codec"]
    with pytest.raises(ValueError, match="fixed operator"):
        _config({"mode": "controller", "pipeline": ["codec"]})
    with pytest.raises(ValueError, match="must match"):
        _config({"pipeline": ["codec"]}, pipeline=["quantizer"])
    with pytest.raises(TypeError, match="codec_config must be an object"):
        _config({"pipeline": ["codec"], "codec_config": "invalid"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("transform", ["dct", "haar"])
@pytest.mark.parametrize("layer_indices", [(0, 1), (0, 5)])
def test_modular_operator_profiles_restore_paged_kv_on_parallel_slots(
    transform, layer_indices
):
    pytest.importorskip("triton")
    from kvserve_v1.compression.quantizer import CenteredCodebookQuantizer
    from kvserve_v1.compression.transformer import KVServeTransformer

    profile = {
        "pipeline": ["transformer", "quantizer", "codec"],
        "transformer_config": {"transform_type": transform, "seed": 123},
        "quantizer_config": {
            "impl": "centered_codebook",
            "key_bits": 3,
            "value_bits": 3,
        },
        "codec_config": {"codec_type": "bitpack", "bitpack_bits": 3},
    }
    specs = {
        f"model.layers.{i}.self_attn": CacheSpec(i, 16, 2, 128, torch.bfloat16)
        for i in layer_indices
    }
    plan = build_memory_plan(_config(profile), specs, num_layers=2)
    producer = create_runtime(plan, "LBHNC", 1, torch.device("cuda", 0))
    consumer = create_runtime(plan, "LBHNC", 1, torch.device("cuda", 0))
    torch.manual_seed(7)
    source = {
        name: torch.randn(4, 2, 16, 256, dtype=torch.bfloat16, device="cuda")
        for name in specs
    }
    target = {name: torch.full_like(value, -9) for name, value in source.items()}
    try:
        producer.bind_kv_caches(source, dict.fromkeys(specs, 0))
        consumer.bind_kv_caches(target, dict.fromkeys(specs, 0))
        assert producer.wire_profile == consumer.wire_profile

        source_io, target_io = ChunkIO(producer), ChunkIO(consumer)

        def transfer(part):
            chunk = source_io.compress(source_io.plan(part, [[2, 0]], part, {0: 31}))
            lengths = tuple(length for _, length in chunk.buffers)
            (payload_bytes, _, _) = lengths
            consumer.slots[part].arena_a[:payload_bytes].copy_(
                producer.slots[part].arena_a[:payload_bytes]
            )
            torch.cuda.synchronize()
            plan = target_io.plan(part, [[1, 3]], part, {0: 31})
            target_io.decompress(plan, lengths, chunk.descriptor).synchronize()

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(transfer, (0, 1)))
        reference_transform = KVServeTransformer(transform_type=transform, seed=123)
        reference_quantizer = CenteredCodebookQuantizer(key_bits=3, value_bits=3)
        for layer_id, name in zip(layer_indices, specs):
            for part in (0, 1):
                select = slice(part * 128, (part + 1) * 128)
                expected = (
                    source[name][[2, 0], :, :, select]
                    .permute(0, 2, 1, 3)
                    .flatten(0, 1)[:31]
                    .float()
                )
                actual = (
                    target[name][[1, 3], :, :, select]
                    .permute(0, 2, 1, 3)
                    .flatten(0, 1)[:31]
                    .float()
                )
                relative_mse = (
                    actual - expected
                ).square().mean() / expected.square().mean()
                assert relative_mse < 0.10
                dense = torch.zeros(
                    (2, 2, 16, 2, 128), dtype=torch.bfloat16, device="cuda"
                )
                dense[part] = source[name][[2, 0], :, :, select].permute(0, 2, 1, 3)
                dense[part, 1, 15:].zero_()
                rotated = reference_transform.transform(layer_id, dense)
                symbols, metadata = reference_quantizer.quantize(layer_id, rotated)
                restored = reference_transform.inverse(
                    layer_id,
                    reference_quantizer.dequantize(layer_id, symbols, metadata),
                )
                reference = restored[part].flatten(0, 1)[:31].float()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert (target[name][[0, 2]] == -9).all()
    finally:
        producer.shutdown()
        consumer.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("codec,threshold", [("bitpack", 1 << 30), ("nvcomp", 0)])
def test_lossless_operator_path_preserves_valid_tokens(codec, threshold):
    if codec == "nvcomp":
        pytest.importorskip("nvidia.nvcomp")
    profile = {
        "pipeline": ["codec"],
        "min_compress_size": threshold,
        "codec_config": {"codec_type": codec, "nvcomp_algorithm": "ANS"},
    }
    specs = {"model.layers.0.self_attn": CacheSpec(0, 16, 2, 128, torch.bfloat16)}
    plan = build_memory_plan(_config(profile), specs, num_layers=1)
    producer = create_runtime(plan, "LBHNC", 1, torch.device("cuda", 0))
    consumer = create_runtime(plan, "LBHNC", 1, torch.device("cuda", 0))
    src = torch.randn(4, 2, 16, 256, dtype=torch.bfloat16, device="cuda")
    dst = torch.full_like(src, -9)
    try:
        producer.bind_kv_caches({next(iter(specs)): src}, dict.fromkeys(specs, 0))
        consumer.bind_kv_caches({next(iter(specs)): dst}, dict.fromkeys(specs, 0))
        encoded = producer.compress(0, [[2, 0]], 0, valid_tokens_by_group={0: 31})
        encoded.event.synchronize()
        consumer.slots[0].arena_a[: encoded.codec_bytes].copy_(
            producer.slots[0].arena_a[: encoded.codec_bytes]
        )
        torch.cuda.synchronize()
        consumer.decompress(
            0,
            [[1, 3]],
            0,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: 31},
        ).synchronize()
        actual = dst[[1, 3], :, :, :128].permute(0, 2, 1, 3).flatten(0, 1)
        expected = src[[2, 0], :, :, :128].permute(0, 2, 1, 3).flatten(0, 1)
        assert torch.equal(actual[:31], expected[:31])
        assert (dst[:, :, :, 128:] == -9).all()
    finally:
        producer.shutdown()
        consumer.shutdown()
