"""CPU-only regression tests: python3 -m unittest discover -s tests/controller -v."""
import json
import math
import os
import tempfile
import unittest

from kvserve_v1.compression.controller.analytical_model import AnalyticalModel
from kvserve_v1.compression.controller.bandwidth import BandwidthEstimator
from kvserve_v1.compression.controller.build_library import build_profile_library, pareto_frontier
from kvserve_v1.compression.controller.online_controller import OnlineController
from kvserve_v1.compression.controller.profile import Profile
from kvserve_v1.compression.controller.profile_library import ProfileLibrary


def profile(name, accuracy, cr, speed):
    return Profile(
        name,
        {"model_name": "M", "dataset": "agent", "quality_metric": "accuracy"},
        {"pipeline": ["quantizer"], "quantizer_config": {"bits": 4}},
        {"compression_ratio": cr, "harmonic_speed_mbps": speed,
         "encode_speed_mbps": speed * 2, "decode_speed_mbps": speed * 2},
        {"accuracy": accuracy, "metric_name": "accuracy"},
    )


def measurements():
    return [
        profile("fast", .99, 2, 500).to_dict(),
        profile("compact", .95, 5, 150).to_dict(),
        profile("dominated", .93, 1.5, 100).to_dict(),
    ]


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        library = build_profile_library(measurements(), b_min=1, b_max=1000)
        self.path = os.path.join(self.tmp.name, "profiles.json")
        with open(self.path, "w") as f:
            json.dump(library, f)
        self.library = ProfileLibrary(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def select(self, ctl, bandwidth, quality=.95, slo=100000, workload=None):
        return ctl.select_profile(
            100 * 1024 ** 2, bandwidth, 0, slo, quality, workload_id=workload
        )

    def test_analytical_formula(self):
        p = profile("codec", .99, 4, 500)
        self.assertAlmostEqual(
            AnalyticalModel.predict_baseline_latency(100 * 1024 ** 2, 100, 0), 1000
        )
        self.assertAlmostEqual(
            AnalyticalModel.predict_latency(p, 100 * 1024 ** 2, 100, 0), 450
        )
        self.assertTrue(AnalyticalModel.check_benefit(p, 100))

    def test_lower_envelope_switching(self):
        ctl = OnlineController(self.library, epsilon=0, seed=0)
        self.assertIsNone(self.select(ctl, 900)[0])
        self.assertEqual(self.select(ctl, 100)[0].profile_id, "fast")
        self.assertEqual(self.select(ctl, 10)[0].profile_id, "compact")

    def test_quality_hard_constraint(self):
        ctl = OnlineController(self.library, epsilon=0)
        self.assertEqual(self.select(ctl, 10, quality=.98)[0].profile_id, "fast")
        self.assertIsNone(self.select(ctl, 10, quality=1.0)[0])
        # Even a malformed bucket must not violate per-profile quality.
        with open(self.path) as f:
            data = json.load(f)
        data["accuracy_buckets"] = [{
            "bucket_id": 0,
            "acc_range": [0.0, math.nextafter(1.0, math.inf)],
            "pareto_profiles": [profile("unsafe", .9, 10, 300).to_dict()],
            "bandwidth_intervals": [{
                "interval_id": 0, "x_range": [0.001, 1.0],
                "model_optimal_profile_id": "unsafe",
                "candidate_profile_ids": ["unsafe"],
            }],
        }]
        with open(self.path, "w") as f:
            json.dump(data, f)
        p, ctx = self.select(
            OnlineController(ProfileLibrary(self.path), epsilon=0), 100, quality=.99
        )
        self.assertIsNone(p)
        self.assertEqual(ctx["action"], "raw")

    def test_invalid_bandwidth_safe_fallback(self):
        ctl = OnlineController(self.library, epsilon=0)
        for b in (0, -1, float("nan"), float("inf")):
            p, ctx = self.select(ctl, b)
            self.assertIsNone(p)
            self.assertEqual(ctx["reason"], "invalid_service_context")
        self.assertEqual(
            self.select(ctl, 1e8)[1]["reason"], "no_matching_bandwidth_interval"
        )

    def test_bandit_residual_changes_policy(self):
        ctl = OnlineController(self.library, epsilon=0, alpha=1.0)
        p, ctx = self.select(ctl, 100)
        self.assertEqual(p.profile_id, "fast")
        ctl.update(ctx, 1200)
        self.assertEqual(self.select(ctl, 100)[0].profile_id, "compact")
        self.assertEqual(ctl.get_statistics()["update_count"], 1)

    def test_slo_and_workload_safety_even_when_exploring(self):
        ctl = OnlineController(self.library, epsilon=1, seed=2)
        for _ in range(20):
            p, _ = self.select(ctl, 100, quality=.98, slo=800, workload="agent")
            self.assertEqual(p.profile_id, "fast")
        self.assertIsNone(self.select(ctl, 100, workload="unknown")[0])
        self.assertIsNone(self.select(ctl, 100, quality=.95, slo=100)[0])

    def test_slo_cooldown(self):
        ctl = OnlineController(
            self.library, epsilon=0, alpha=.0001,
            violation_threshold=3, violation_window=4, cooldown_s=60
        )
        for _ in range(3):
            p, ctx = self.select(ctl, 100, slo=1000)
            self.assertEqual(p.profile_id, "fast")
            ctl.update(ctx, 1100, actual_jct=True)
        self.assertEqual(self.select(ctl, 100, slo=1000)[0].profile_id, "compact")
        self.assertEqual(ctl.get_statistics()["cooldown_count"], 1)

    def test_send_proxy_does_not_count_slo_violation(self):
        ctl = OnlineController(
            self.library, epsilon=0, alpha=.0001,
            violation_threshold=1, violation_window=3
        )
        _, ctx = self.select(ctl, 100, slo=1000)
        ctl.update(ctx, 1100, actual_jct=False)
        self.assertEqual(ctl.get_statistics()["cooldown_count"], 0)

    def test_pareto_candidate_budget(self):
        frontier = pareto_frontier([Profile.from_dict(x) for x in measurements()])
        self.assertEqual({p.profile_id for p in frontier}, {"fast", "compact"})
        for bucket in self.library.accuracy_buckets:
            for interval in bucket["bandwidth_intervals"]:
                ids = interval["candidate_profile_ids"]
                self.assertLessEqual(len(ids), 3)
                self.assertIn(interval["model_optimal_profile_id"], ids)

    def test_bandwidth_ewma(self):
        meter = BandwidthEstimator(500, alpha=1, min_samples=2)
        self.assertEqual(meter.get(), 500)
        self.assertFalse(meter.observe(10, .1))
        self.assertFalse(meter.observe(10 << 20, float("nan")))
        meter.observe(10 << 20, .1)
        self.assertEqual(meter.get(), 500)
        meter.observe(10 << 20, .1)
        self.assertAlmostEqual(meter.get(), 100)
        meter.observe(10 << 20, 1.0)
        self.assertAlmostEqual(meter.get(), 25)
        meter.observe(10 << 20, 1.0)
        self.assertAlmostEqual(meter.get(), 10)

    def test_bandit_state_roundtrip(self):
        ctl = OnlineController(self.library, epsilon=0)
        _, ctx = self.select(ctl, 100)
        ctl.update(ctx, 800)
        path = os.path.join(self.tmp.name, "bandit.json")
        ctl.save_state(path)
        restored = OnlineController(self.library, epsilon=0)
        restored.load_state(path)
        key = (ctx["bucket_id"], ctx["interval_id"], "fast")
        self.assertEqual(restored.bandit.get_stats(*key), ctl.bandit.get_stats(*key))

    def test_generator_rejects_unknown_quality_and_pipeline(self):
        p = measurements()[0]
        del p["quality_metrics"]["accuracy"]
        with self.assertRaises(KeyError):
            build_profile_library([p])
        p = measurements()[0]
        p["compression_config"]["pipeline"] = ["quantize", "codec"]
        with self.assertRaises(ValueError):
            build_profile_library([p])


if __name__ == "__main__":
    unittest.main()
