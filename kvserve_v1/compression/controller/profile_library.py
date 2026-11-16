"""Validated, indexed Pareto profile library for constant-size online lookups."""

import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional

from kvserve_v1.compression.controller.profile import Profile


class ProfileLibrary:
    """Offline accuracy buckets -> 1/B intervals -> up to three candidates.

    Lookup never silently chooses an unrelated bucket or bandwidth interval.
    Such misses are handled by the controller's lossless raw fallback.
    """

    def __init__(self, json_path: str):
        with open(Path(json_path), encoding="utf-8") as f:
            data = json.load(f)
        self.metadata = data.get("library_metadata", {})
        self.accuracy_buckets: List[Dict] = data.get("accuracy_buckets", [])
        self.profile_map: Dict[str, Profile] = {}
        self._interval_starts: Dict[int, List[float]] = {}
        self._validate_and_index()

    def _validate_and_index(self) -> None:
        if not self.accuracy_buckets:
            raise ValueError("Profile library must contain accuracy_buckets")
        bucket_ids = set()
        ranges = []
        for bucket in self.accuracy_buckets:
            if not {"bucket_id", "acc_range", "bandwidth_intervals"} <= bucket.keys():
                raise ValueError("Incomplete accuracy bucket")
            if bucket["bucket_id"] in bucket_ids:
                raise ValueError("Duplicate bucket_id")
            bucket_ids.add(bucket["bucket_id"])
            lower, upper = bucket["acc_range"]
            if not (0 <= lower <= 1 and lower < upper <= math.nextafter(1.0, math.inf)):
                raise ValueError("Invalid acc_range")
            ranges.append((lower, upper))
            for raw in bucket.get("pareto_profiles", []):
                p = Profile.from_dict(raw)
                if not p.profile_id:
                    raise ValueError("Empty profile_id")
                if not (math.isfinite(p.accuracy) and 0.0 <= p.accuracy <= 1.0):
                    raise ValueError(f"{p.profile_id}: invalid accuracy")
                if not (math.isfinite(p.compression_ratio)
                        and p.compression_ratio >= 1.0
                        and math.isfinite(p.harmonic_speed)
                        and p.harmonic_speed > 0.0):
                    raise ValueError(f"{p.profile_id}: invalid CR/speed")
                if not isinstance(p.compression_config, dict):
                    raise ValueError(f"{p.profile_id}: invalid compression config")
                stages = p.compression_config.get("pipeline")
                if (not isinstance(stages, list) or not stages
                        or not set(stages) <= {"transformer", "quantizer", "codec"}):
                    raise ValueError(f"{p.profile_id}: unsupported pipeline {stages}")
                old = self.profile_map.get(p.profile_id)
                if old is not None and old.to_dict() != p.to_dict():
                    raise ValueError(f"Conflicting data for profile_id={p.profile_id}")
                self.profile_map[p.profile_id] = p
            prev_upper = -1.0
            interval_ids = set()
            self._interval_starts[bucket["bucket_id"]] = [
                interval["x_range"][0]
                for interval in bucket["bandwidth_intervals"]
            ]
            for interval in bucket["bandwidth_intervals"]:
                if not {"interval_id", "x_range", "candidate_profile_ids"} <= interval.keys():
                    raise ValueError("Incomplete bandwidth interval")
                if interval["interval_id"] in interval_ids:
                    raise ValueError("Duplicate interval_id")
                interval_ids.add(interval["interval_id"])
                x_lower, x_upper = interval["x_range"]
                if not (0 <= x_lower < x_upper and math.isfinite(x_lower)
                        and math.isfinite(x_upper)):
                    raise ValueError("Invalid 1/B interval")
                if x_lower < prev_upper - 1e-14:
                    raise ValueError("Overlapping or unsorted bandwidth intervals")
                prev_upper = x_upper
                candidates = interval["candidate_profile_ids"]
                if not isinstance(candidates, list) or len(candidates) > 3:
                    raise ValueError("Each interval supports at most three candidates")
                if len(set(candidates)) != len(candidates):
                    raise ValueError("Duplicate candidate profile")
                missing = [pid for pid in candidates if pid not in self.profile_map]
                if missing:
                    raise ValueError(f"Candidate profiles missing in bucket: {missing}")
                opt = interval.get("model_optimal_profile_id")
                if opt is not None and opt not in candidates:
                    raise ValueError("Model-optimal profile must be a candidate")
        sorted_ranges = sorted(ranges)
        for (_, hi), (lo2, _) in zip(sorted_ranges, sorted_ranges[1:]):
            if lo2 < hi:
                raise ValueError("Overlapping accuracy ranges")

    def find_bucket(self, acc_req: float) -> Optional[Dict]:
        if not math.isfinite(acc_req) or not (0 <= acc_req <= 1):
            return None
        for bucket in self.accuracy_buckets:
            low, high = bucket["acc_range"]
            if low <= acc_req < high:
                return bucket
        return None

    def find_interval(self, bucket: Dict, B_mbps: float) -> Optional[Dict]:
        if not math.isfinite(B_mbps) or B_mbps <= 0:
            return None
        x = 1.0 / B_mbps
        intervals = bucket.get("bandwidth_intervals", [])
        if not intervals:
            return None
        starts = self._interval_starts[bucket["bucket_id"]]
        idx = bisect_right(starts, x) - 1
        if idx >= 0:
            lo, hi = intervals[idx]["x_range"]
            if lo <= x < hi or (idx == len(intervals) - 1 and x == hi):
                return intervals[idx]
        return None

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        return self.profile_map.get(profile_id)

    def get_candidate_profiles(self, bucket: Dict, interval: Dict) -> List[Profile]:
        return [
            self.profile_map[pid]
            for pid in interval.get("candidate_profile_ids", ())
            if pid in self.profile_map
        ]

    def get_optimal_profile(self, bucket: Dict, interval: Dict) -> Optional[Profile]:
        return self.get_profile(interval.get("model_optimal_profile_id", ""))

    def get_all_profiles(self) -> List[Profile]:
        return list(self.profile_map.values())

    def get_num_profiles(self) -> int:
        return len(self.profile_map)

    def get_num_buckets(self) -> int:
        return len(self.accuracy_buckets)

    def get_bucket_info(self, bucket_id: int) -> Optional[Dict]:
        return next(
            (b for b in self.accuracy_buckets if b["bucket_id"] == bucket_id),
            None,
        )

    def summary(self) -> str:
        intervals = sum(
            len(b["bandwidth_intervals"]) for b in self.accuracy_buckets
        )
        return (
            f"ProfileLibrary(model={self.metadata.get('model_name', 'unknown')}, "
            f"profiles={len(self.profile_map)}, buckets={len(self.accuracy_buckets)}, "
            f"intervals={intervals})"
        )

    def __repr__(self) -> str:
        return self.summary()
