"""Service-aware online profile selector: analytical screen + residual bandit.

Only a small, *offline-generated* set of <= 3 profiles is scored per request.
The fallback is lossless, uncompressed KV. This module has no torch dependency.
"""

from collections import deque
import math
import random
import threading
import time
from typing import Any, Dict, Optional, Tuple

from kvserve_v1.compression.controller.analytical_model import AnalyticalModel
from kvserve_v1.compression.controller.bandit_state import BanditStateManager
from kvserve_v1.compression.controller.profile import Profile
from kvserve_v1.compression.controller.profile_library import ProfileLibrary


class OnlineController:
    """Paper-style lightweight, quality/SLO-constrained online controller.

    update() expects a *completed* observation of the same latency quantity
    predicted by the model (ideally end-to-end request JCT). Enqueue latency
    must never be passed as the observed job completion time.
    """

    def __init__(
        self,
        profile_library: ProfileLibrary,
        epsilon: float = 0.1,
        bandit_state: Optional[BanditStateManager] = None,
        alpha: float = 0.2,
        seed: Optional[int] = None,
        violation_threshold: int = 3,
        violation_window: int = 10,
        cooldown_s: float = 60.0,
        min_improvement: float = 0.0,
    ):
        if not (math.isfinite(epsilon) and 0 <= epsilon <= 1):
            raise ValueError("epsilon must be in [0, 1]")
        if violation_window < 1 or not (1 <= violation_threshold <= violation_window):
            raise ValueError("Require 1 <= violation_threshold <= violation_window")
        if not math.isfinite(cooldown_s) or cooldown_s < 0:
            raise ValueError("cooldown_s must be finite and nonnegative")
        if not (math.isfinite(min_improvement) and 0 <= min_improvement < 1):
            raise ValueError("min_improvement must be in [0, 1)")
        self.library = profile_library
        self.epsilon = epsilon
        self.bandit = bandit_state or BanditStateManager(alpha=alpha)
        self.model = AnalyticalModel()
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._violation_window = violation_window
        self._violation_threshold = violation_threshold
        self._cooldown_s = cooldown_s
        self._min_improvement = min_improvement
        self._recent_violations: dict[tuple, deque] = {}
        self._cooldown_until: dict[tuple, float] = {}
        self.total_decisions = 0
        self.exploit_count = 0
        self.explore_count = 0
        self.no_compression_count = 0
        self.update_count = 0
        self.cooldown_count = 0
        self.profile_selections: dict[str, int] = {}
        self.fallback_reasons: dict[str, int] = {}

    def _fallback(self, reason: str, **details) -> Tuple[None, Dict[str, Any]]:
        self.no_compression_count += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        return None, {"reason": reason, "action": "raw", **details}

    def select_profile(
        self,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float,
        T_SLO_ms: float,
        acc_req: float,
        workload_id: Optional[str] = None,
    ) -> Tuple[Optional[Profile], Dict[str, Any]]:
        """Select a profile under quality, SLO, and measured-bandwidth constraints.

        V_bytes: raw KV bytes, B_mbps: effective MiB/s, times: milliseconds.
        An explicit workload_id restricts selection to profiles calibrated for
        that workload; omitted workload_id assumes the library is workload-scoped.
        """
        with self._lock:
            self.total_decisions += 1
            if (
                not math.isfinite(V_bytes) or V_bytes <= 0
                or not math.isfinite(B_mbps) or B_mbps <= 0
                or not math.isfinite(T_model_ms) or T_model_ms < 0
                or (not math.isfinite(T_SLO_ms) and T_SLO_ms != float("inf"))
                or T_SLO_ms <= 0
                or not math.isfinite(acc_req) or not 0 <= acc_req <= 1
            ):
                return self._fallback("invalid_service_context")

            bucket = self.library.find_bucket(acc_req)
            if bucket is None:
                return self._fallback("no_matching_quality_bucket")
            interval = self.library.find_interval(bucket, B_mbps)
            if interval is None:
                return self._fallback("no_matching_bandwidth_interval")
            candidates = self.library.get_candidate_profiles(bucket, interval)
            if not candidates:
                return self._fallback("no_candidates")

            raw_latency = self.model.predict_baseline_latency(
                V_bytes, B_mbps, T_model_ms
            )
            upper_bound = raw_latency * (1.0 - self._min_improvement)
            predictions = []
            now = time.monotonic()
            for profile in candidates:
                # Required even when the library was generated for a quality
                # bucket: a range can contain stricter requests than its floor.
                if profile.accuracy < acc_req:
                    continue
                if workload_id is not None:
                    workload = (
                        profile.metadata.get("workload")
                        or profile.metadata.get("dataset")
                    )
                    if workload != workload_id:
                        continue
                if not self.model.check_benefit(profile, B_mbps):
                    continue
                key = (bucket["bucket_id"], interval["interval_id"],
                       profile.profile_id)
                if now < self._cooldown_until.get(key, 0.0):
                    continue
                predicted = self.model.predict_latency(
                    profile, V_bytes, B_mbps, T_model_ms
                )
                # The conservative model must pass SLO independently of bandit.
                if predicted > T_SLO_ms or predicted >= upper_bound:
                    continue
                _, residual = self.bandit.get_stats(*key)
                corrected = max(0.001, predicted + residual)
                if corrected > T_SLO_ms or corrected >= upper_bound:
                    continue
                predictions.append((corrected, predicted, residual, profile, key))

            if not predictions:
                return self._fallback(
                    "no_feasible_profile",
                    bucket_id=bucket["bucket_id"],
                    interval_id=interval["interval_id"],
                    baseline_latency=raw_latency,
                )

            explore = self._rng.random() < self.epsilon
            if explore:
                selected = self._rng.choice(predictions)
                self.explore_count += 1
            else:
                selected = min(predictions, key=lambda item: item[0])
                self.exploit_count += 1
            corrected, predicted, residual, profile, key = selected
            self.profile_selections[profile.profile_id] = (
                self.profile_selections.get(profile.profile_id, 0) + 1
            )
            return profile, {
                "bucket_id": key[0],
                "interval_id": key[1],
                "profile_id": key[2],
                "T_hat": predicted,
                "T_eff": corrected,
                "delta_bar": residual,
                "speedup_predicted": raw_latency / corrected,
                "action": "explore" if explore else "exploit",
                "num_candidates": len(candidates),
                "num_feasible": len(predictions),
                "baseline_latency": raw_latency,
                "V_bytes": V_bytes,
                "B_mbps": B_mbps,
                "T_model_ms": T_model_ms,
                "T_SLO_ms": T_SLO_ms,
                "acc_req": acc_req,
                "workload_id": workload_id,
            }

    def update(
        self,
        context: Dict[str, Any],
        T_obs_ms: float,
        *,
        actual_jct: bool = True,
    ) -> None:
        """Update EWMA after completion; only real JCT affects SLO cooldown.

        For experiments with a full request-completion signal, call with the
        default actual_jct=True. A proxy for the same predicted metric may
        update the residual with actual_jct=False, but cannot enforce SLO.
        """
        if (
            "profile_id" not in context
            or not math.isfinite(T_obs_ms)
            or T_obs_ms <= 0
        ):
            return
        predicted = context["T_hat"]
        if not math.isfinite(predicted):
            return
        key = (context["bucket_id"], context["interval_id"],
               context["profile_id"])
        with self._lock:
            self.bandit.update(*key, T_obs_ms - predicted)
            self.update_count += 1
            if not actual_jct:
                return
            slo = context.get("T_SLO_ms", float("inf"))
            if not math.isfinite(slo):
                return
            history = self._recent_violations.setdefault(
                key, deque(maxlen=self._violation_window)
            )
            history.append(T_obs_ms > slo)
            if sum(history) >= self._violation_threshold:
                self._cooldown_until[key] = time.monotonic() + self._cooldown_s
                history.clear()
                self.cooldown_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_decisions": self.total_decisions,
                "exploit_count": self.exploit_count,
                "explore_count": self.explore_count,
                "no_compression_count": self.no_compression_count,
                "compression_rate": (
                    (self.total_decisions - self.no_compression_count)
                    / self.total_decisions if self.total_decisions else 0
                ),
                "update_count": self.update_count,
                "cooldown_count": self.cooldown_count,
                "bandit_states": self.bandit.get_total_states(),
                "bandit_memory_bytes": self.bandit.get_memory_size(),
                "profile_selections": dict(self.profile_selections),
                "fallback_reasons": dict(self.fallback_reasons),
            }

    def reset_statistics(self) -> None:
        with self._lock:
            self.total_decisions = 0
            self.exploit_count = 0
            self.explore_count = 0
            self.no_compression_count = 0
            self.update_count = 0
            self.cooldown_count = 0
            self.profile_selections.clear()
            self.fallback_reasons.clear()

    def save_state(self, path: str) -> None:
        with self._lock:
            self.bandit.save(path)

    def load_state(self, path: str) -> None:
        with self._lock:
            self.bandit.load(path)

    def __repr__(self) -> str:
        s = self.get_statistics()
        return (
            f"OnlineController(epsilon={self.epsilon}, "
            f"decisions={s['total_decisions']}, updates={s['update_count']}, "
            f"bandit_states={s['bandit_states']})"
        )
