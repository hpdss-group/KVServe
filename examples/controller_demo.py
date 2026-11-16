"""CPU-only demonstration of the paper-style online controller.

Synthetic profiles are used strictly to exercise algorithm behavior, not to
claim compression quality or measured GPU performance.

Run from repository root:
    python3 examples/controller_demo.py
"""
import json
import tempfile
import time
from pathlib import Path

from kvserve_v1.compression.controller.build_library import build_profile_library
from kvserve_v1.compression.controller.online_controller import OnlineController
from kvserve_v1.compression.controller.profile_library import ProfileLibrary


def profile(name, ratio, speed, quality):
    return {
        "profile_id": name,
        "metadata": {"model_name": "synthetic-model",
                     "dataset": "synthetic-workload",
                     "quality_metric": "accuracy"},
        "compression_config": {
            "pipeline": ["quantizer"],
            "quantizer_config": {"model_name": "synthetic-model"},
        },
        "performance_metrics": {
            "compression_ratio": ratio,
            "harmonic_speed_mbps": speed,
            "encode_speed_mbps": speed * 2,
            "decode_speed_mbps": speed * 2,
        },
        "quality_metrics": {"accuracy": quality, "metric_name": "accuracy"},
    }


def main():
    records = [
        profile("fast-codec", ratio=2, speed=500, quality=.99),
        profile("compact-codec", ratio=5, speed=150, quality=.95),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        library_file = Path(tmp) / "controller_profiles.json"
        library_file.write_text(
            json.dumps(
                build_profile_library(records, b_min=1, b_max=1000),
                indent=2,
            ),
            encoding="utf-8",
        )
        ctl = OnlineController(ProfileLibrary(str(library_file)),
                               epsilon=.1, seed=2026)
        V = 100 * 1024**2
        print("Synthetic demonstration (not GPU benchmark):")
        print(f"{'bandwidth MiB/s':>16} {'selected':>16} {'T_hat ms':>12}")
        for bandwidth in [900, 100, 10, 10, 10, 100, 900]:
            selected, ctx = ctl.select_profile(
                V_bytes=V, B_mbps=bandwidth, T_model_ms=0,
                T_SLO_ms=30000, acc_req=.95,
                workload_id="synthetic-workload",
            )
            print(f"{bandwidth:16.0f} "
                  f"{selected.profile_id if selected else 'raw':>16} "
                  f"{ctx.get('T_hat', ctx.get('baseline_latency', 0)):12.1f}")
            if selected:
                # Synthetic residual (GPU contention), to show online update.
                ctl.update(ctx, ctx["T_hat"] + (120 if bandwidth < 50 else 0),
                           actual_jct=False)

        start = time.perf_counter()
        for _ in range(20000):
            ctl.select_profile(V, 100, 0, 30000, .95)
        avg_us = (time.perf_counter() - start) * 1e6 / 20000
        print(f"Controller: {avg_us:.2f} microseconds/decision "
              "(20000 CPU-only calls; machine-dependent)")
        print("Stats:", ctl.get_statistics())


if __name__ == "__main__":
    main()
