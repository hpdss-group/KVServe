"""Offline Pareto / lower-envelope compiler for the KVServe online controller.

Input: JSON array of measured Profile.to_dict() records (or {"profiles": ...}).
No GPU dependency. Accurate CR, combined encode/decode throughput and quality
measurements must be supplied; the tool does not invent benchmark values.

Example:
    python -m kvserve_v1.compression.controller.build_library \
      --profiles measurements.json --output profiles.json
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from kvserve_v1.compression.controller.profile import Profile


def _measured_profiles(data: list[dict]) -> list[Profile]:
    if not data:
        raise ValueError("No measured profiles supplied")
    profiles = [Profile.from_dict(d) for d in data]
    seen = set()
    for p in profiles:
        if p.profile_id in seen:
            raise ValueError(f"Duplicate profile_id: {p.profile_id}")
        seen.add(p.profile_id)
        if not (math.isfinite(p.accuracy) and 0 <= p.accuracy <= 1):
            raise ValueError(f"{p.profile_id}: missing/invalid measured accuracy")
        if not (math.isfinite(p.compression_ratio)
                and p.compression_ratio > 1.0
                and math.isfinite(p.harmonic_speed)
                and p.harmonic_speed > 0):
            raise ValueError(f"{p.profile_id}: missing/invalid measured throughput/CR")
        # Legacy examples sometimes record the conventional (factor-of-two)
        # harmonic mean. The analytical model instead requires the aggregate
        # throughput of *both* serial codec stages.
        enc = p.performance_metrics.get("encode_speed_mbps")
        dec = p.performance_metrics.get("decode_speed_mbps")
        if enc is not None or dec is not None:
            if (enc is None or dec is None or not math.isfinite(enc)
                    or not math.isfinite(dec) or enc <= 0 or dec <= 0):
                raise ValueError(f"{p.profile_id}: invalid codec stage speeds")
            aggregate = 1.0 / (1.0 / enc + 1.0 / dec)
            if abs(p.harmonic_speed - aggregate) / aggregate > 0.02:
                raise ValueError(
                    f"{p.profile_id}: harmonic_speed_mbps must be "
                    f"1/(1/encode + 1/decode)={aggregate:.6g} MiB/s"
                )
        config = p.compression_config
        pipeline = config.get("pipeline")
        if not isinstance(pipeline, list) or not pipeline:
            raise ValueError(f"{p.profile_id}: missing runnable compression pipeline")
        if not set(pipeline) <= {"transformer", "quantizer", "codec"}:
            raise ValueError(f"{p.profile_id}: unsupported pipeline stage {pipeline}")
    # Scores obtained for different tasks/hardware are not interchangeable.
    for metadata_key in ("model_name", "dataset", "quality_metric"):
        values = {
            p.metadata.get(metadata_key)
            or (p.quality_metrics.get("metric_name") if metadata_key == "quality_metric" else None)
            for p in profiles
        }
        if len(values) != 1 or None in values:
            raise ValueError(f"Profiles must share a known {metadata_key}: {values}")
    return profiles


def pareto_frontier(profiles: Iterable[Profile]) -> list[Profile]:
    """Undominated in (quality, compressed bytes saved, codec throughput)."""
    candidates = list(profiles)
    return [
        p for p in candidates
        if not any(
            (q.accuracy >= p.accuracy
             and q.compression_ratio >= p.compression_ratio
             and q.harmonic_speed >= p.harmonic_speed
             and (q.accuracy > p.accuracy
                  or q.compression_ratio > p.compression_ratio
                  or q.harmonic_speed > p.harmonic_speed))
            for q in candidates if q is not p
        )
    ]


def _cost_per_mib(p: Profile, x: float) -> float:
    """Profile cost in seconds per MiB at x = 1/B (B in MiB/s)."""
    return 1.0 / p.harmonic_speed + x / p.compression_ratio


def bandwidth_intervals(
    profiles: list[Profile], b_min: float, b_max: float
) -> list[dict]:
    """Find exact crossings of affine latency lines in 1/B space."""
    if not profiles:
        return []
    if not (0 < b_min < b_max and math.isfinite(b_max)):
        raise ValueError("Require finite 0 < b_min < b_max")
    x_lo, x_hi = 1.0 / b_max, 1.0 / b_min
    intersections = [x_lo, x_hi]
    for i, p in enumerate(profiles):
        for q in profiles[i + 1:]:
            a_diff = 1.0 / p.harmonic_speed - 1.0 / q.harmonic_speed
            b_diff = 1.0 / p.compression_ratio - 1.0 / q.compression_ratio
            if abs(b_diff) <= 1e-15:
                continue
            crossing = -a_diff / b_diff
            if x_lo < crossing < x_hi:
                intersections.append(crossing)
    intersections.sort()
    cuts = [intersections[0]]
    for x in intersections[1:]:
        if x - cuts[-1] > max(1e-16, 1e-12 * x):
            cuts.append(x)
    if cuts[-1] < x_hi:
        cuts.append(x_hi)

    segments = []
    for left, right in zip(cuts, cuts[1:]):
        middle = (left + right) / 2
        ranked = sorted(profiles, key=lambda p: (_cost_per_mib(p, middle), p.profile_id))
        segments.append({
            "x_range": [left, right],
            "best": ranked[0].profile_id,
            "ranked": [p.profile_id for p in ranked],
        })

    intervals = []
    for i, segment in enumerate(segments):
        # Keep the model optimum and its left/right neighbors around crossings;
        # optionally fill the 3-profile shortlist using the cost ranking.
        shortlisted = [segment["best"]]
        for j in (i - 1, i + 1):
            if 0 <= j < len(segments):
                pid = segments[j]["best"]
                if pid not in shortlisted and len(shortlisted) < 3:
                    shortlisted.append(pid)
        for pid in segment["ranked"]:
            if len(shortlisted) >= 3:
                break
            if pid not in shortlisted:
                shortlisted.append(pid)
        intervals.append({
            "interval_id": i,
            "x_range": segment["x_range"],
            "model_optimal_profile_id": segment["best"],
            "candidate_profile_ids": shortlisted,
        })
    return intervals


def build_profile_library(
    measurements: list[dict], *, b_min: float = 1.0,
    b_max: float = 100_000.0
) -> dict:
    """Compile measured profiles into quality buckets and 1/B intervals.

    Buckets terminate just *after* the measured quality value, so requests
    in an interval can only select profiles meeting its strict upper bound.
    This avoids optimistic quality bucket rounding at runtime.
    """
    profiles = _measured_profiles(measurements)
    if not (math.isfinite(b_min) and math.isfinite(b_max)
            and 0 < b_min < b_max):
        raise ValueError("Invalid bandwidth bounds")
    qualities = sorted({p.accuracy for p in profiles})
    accuracy_buckets = []
    lower = 0.0
    for quality in qualities:
        upper = math.nextafter(quality, math.inf)
        if upper <= lower:
            continue
        feasible = [p for p in profiles if p.accuracy >= quality]
        pareto = pareto_frontier(feasible)
        intervals = bandwidth_intervals(pareto, b_min, b_max)
        accuracy_buckets.append({
            "bucket_id": len(accuracy_buckets),
            "acc_range": [lower, upper],
            "pareto_profiles": [p.to_dict() for p in pareto],
            "bandwidth_intervals": intervals,
        })
        lower = upper
    if not accuracy_buckets:
        raise ValueError("All measured qualities were unusable")
    first = profiles[0]
    return {
        "library_metadata": {
            "schema_version": 1,
            "model_name": first.metadata["model_name"],
            "dataset": first.metadata["dataset"],
            "quality_metric": (
                first.metadata.get("quality_metric")
                or first.quality_metrics["metric_name"]
            ),
            "bandwidth_units": "MiB/s",
            "bandwidth_range": [b_min, b_max],
            "input_profiles": len(profiles),
            "source_sha256": hashlib.sha256(
                json.dumps(measurements, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "description": "Measured profiles; lower-envelope controller candidates",
        },
        "accuracy_buckets": accuracy_buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bandwidth-min", type=float, default=1.0,
                        help="Lower effective bandwidth in MiB/s")
    parser.add_argument("--bandwidth-max", type=float, default=100000.0,
                        help="Upper effective bandwidth in MiB/s")
    args = parser.parse_args()
    with args.profiles.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["profiles"]
    if not isinstance(data, list):
        raise ValueError("Input must be a list of measured profiles")
    library = build_profile_library(
        data, b_min=args.bandwidth_min, b_max=args.bandwidth_max
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False, allow_nan=False)
    # Import only after producing output, validating the same public schema.
    from kvserve_v1.compression.controller.profile_library import ProfileLibrary
    result = ProfileLibrary(str(args.output))
    print(f"Wrote {args.output}: {result.summary()}")


if __name__ == "__main__":
    main()
