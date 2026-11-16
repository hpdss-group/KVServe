"""Low-overhead effective-bandwidth estimator for completed NCCL transfers."""

import math
import threading
import time


class BandwidthEstimator:
    """EWMA from completed payload bytes / device transfer time.

    Tiny transfers, invalid samples and stale samples are ignored. Only the
    NCCL completion thread writes observations; request threads read a scalar.
    The initial configured bandwidth remains the fallback at cold start.
    """

    def __init__(
        self, initial_mibps: float, *, alpha: float = 0.25,
        min_samples: int = 2, min_bytes: int = 1 << 20,
        min_duration_s: float = 1e-4, stale_after_s: float = 30.0,
    ):
        if not math.isfinite(initial_mibps) or initial_mibps <= 0:
            raise ValueError("initial_mibps must be finite and positive")
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if min_samples < 1 or min_bytes <= 0 or min_duration_s <= 0:
            raise ValueError("Invalid bandwidth sampling thresholds")
        self.initial_mibps = float(initial_mibps)
        self.alpha = alpha
        self.min_samples = min_samples
        self.min_bytes = min_bytes
        self.min_duration_s = min_duration_s
        self.stale_after_s = stale_after_s
        self._samples = 0
        self._estimate_mibps = float(initial_mibps)
        self._last_sample_time = 0.0
        self._lock = threading.Lock()

    def observe(self, payload_bytes: int, transfer_s: float) -> bool:
        """Nonblocking O(1) completed-transfer update."""
        if (
            payload_bytes < self.min_bytes
            or not math.isfinite(transfer_s)
            or transfer_s < self.min_duration_s
        ):
            return False
        observed = payload_bytes / (1024.0 * 1024.0 * transfer_s)
        if not math.isfinite(observed) or observed <= 0:
            return False
        with self._lock:
            if not self._samples:
                self._estimate_mibps = observed
            else:
                # Bound a single outlier's influence, but allow meaningful
                # bandwidth changes over successive samples.
                low = self._estimate_mibps / 4.0
                high = self._estimate_mibps * 4.0
                bounded = min(max(observed, low), high)
                self._estimate_mibps += self.alpha * (
                    bounded - self._estimate_mibps
                )
            self._samples += 1
            self._last_sample_time = time.monotonic()
        return True

    def get(self) -> float:
        with self._lock:
            if (
                self._samples < self.min_samples
                or time.monotonic() - self._last_sample_time > self.stale_after_s
            ):
                return self.initial_mibps
            return self._estimate_mibps

    def stats(self) -> dict:
        with self._lock:
            active = (
                self._samples >= self.min_samples
                and time.monotonic() - self._last_sample_time <= self.stale_after_s
            )
            return {
                "bandwidth_mibps": self._estimate_mibps if active
                else self.initial_mibps,
                "bandwidth_samples": self._samples,
                "bandwidth_dynamic": active,
            }
