"""Lightweight KV-transfer latency model used by the online controller.

Compression ratio is uncompressed bytes / transferred bytes (>= 1).
All throughput inputs are MiB/s, V is in bytes, and latency is in ms.
The profile's effective codec throughput S is defined by
    1/S = 1/S_encode + 1/S_decode.
This is not the conventional factor-of-two harmonic mean.
"""

import math

from kvserve_v1.compression.controller.profile import Profile


class AnalyticalModel:
    """Paper-style serial cost model; deliberately inexpensive per request."""

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
        return value

    @staticmethod
    def predict_latency(
        profile: Profile, V_bytes: float, B_mbps: float, T_model_ms: float
    ) -> float:
        """T = T_model + V/S_codec + V/(B*CR), in milliseconds."""
        bandwidth = AnalyticalModel._positive_finite(B_mbps, "B_mbps")
        speed = AnalyticalModel._positive_finite(
            profile.harmonic_speed, "harmonic_speed_mbps"
        )
        ratio = AnalyticalModel._positive_finite(
            profile.compression_ratio, "compression_ratio"
        )
        if ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1")
        volume_mib = float(V_bytes) / (1024.0 * 1024.0)
        if not math.isfinite(volume_mib) or volume_mib < 0:
            raise ValueError("V_bytes must be finite and nonnegative")
        if not math.isfinite(T_model_ms) or T_model_ms < 0:
            raise ValueError("T_model_ms must be finite and nonnegative")
        return T_model_ms + 1000.0 * volume_mib * (
            1.0 / speed + 1.0 / (bandwidth * ratio)
        )

    @staticmethod
    def predict_baseline_latency(
        V_bytes: float, B_mbps: float, T_model_ms: float
    ) -> float:
        """T_raw = T_model + V/B, in milliseconds."""
        bandwidth = AnalyticalModel._positive_finite(B_mbps, "B_mbps")
        volume_mib = float(V_bytes) / (1024.0 * 1024.0)
        if not math.isfinite(volume_mib) or volume_mib < 0:
            raise ValueError("V_bytes must be finite and nonnegative")
        if not math.isfinite(T_model_ms) or T_model_ms < 0:
            raise ValueError("T_model_ms must be finite and nonnegative")
        return T_model_ms + 1000.0 * volume_mib / bandwidth

    @staticmethod
    def check_benefit(profile: Profile, B_mbps: float) -> bool:
        """Compression is worthwhile when B < (1 - 1/CR) * S."""
        bandwidth = AnalyticalModel._positive_finite(B_mbps, "B_mbps")
        speed = AnalyticalModel._positive_finite(
            profile.harmonic_speed, "harmonic_speed_mbps"
        )
        ratio = AnalyticalModel._positive_finite(
            profile.compression_ratio, "compression_ratio"
        )
        return ratio > 1.0 and bandwidth < (1.0 - 1.0 / ratio) * speed

    @staticmethod
    def compute_speedup(
        profile: Profile, V_bytes: float, B_mbps: float, T_model_ms: float
    ) -> float:
        raw = AnalyticalModel.predict_baseline_latency(
            V_bytes, B_mbps, T_model_ms
        )
        compressed = AnalyticalModel.predict_latency(
            profile, V_bytes, B_mbps, T_model_ms
        )
        return raw / compressed if compressed > 0 else 1.0
