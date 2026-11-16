"""CPU-only tests for controller-mode adapter/transport-observation contracts.

The actual CUDA NCCL path remains an integration test for GPU machines.
"""
import importlib
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from kvserve_v1.compression.controller.build_library import build_profile_library
from test_online_controller import measurements


class FakeKV:
    ndim = 6
    dtype = "float16"
    shape = (1, 2, 2, 256, 4, 128)

    def contiguous(self):
        return self

    def numel(self):
        return 50 * 1024**2

    def element_size(self):
        return 2


class FakeManager:
    def compress_all_layers(self, **kwargs):
        return types.SimpleNamespace(metadata=dict(kwargs["metadata"]))


class AdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The adapter imports torch and a vLLM logger; neither is needed by
        # the pure selection path exercised by this CPU mock.
        torch = types.ModuleType("torch")
        torch.Tensor = FakeKV
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []
        logger = types.ModuleType("vllm.logger")
        logger.init_logger = logging.getLogger
        cls.patcher = patch.dict(
            sys.modules, {"torch": torch, "vllm": vllm, "vllm.logger": logger}
        )
        cls.patcher.start()
        cls.adapter_module = importlib.import_module(
            "kvserve_v1.compression.manager"
        )

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "library.json")
        with open(self.path, "w") as f:
            json.dump(build_profile_library(
                measurements(), b_min=1, b_max=1000
            ), f)

    def tearDown(self):
        self.tmp.cleanup()

    def create_adapter(self, feedback_mode="manual"):
        spec = {
            "mode": "controller", "library_path": self.path, "epsilon": 0,
            "feedback_mode": feedback_mode,
            "service_config": {
                "bandwidth_mbps": 100,
                "slo_ms": 20000,
                "accuracy_requirement": .95,
                "workload_id": "agent",
            },
        }
        adapter = self.adapter_module.KVCompressionAdapter(
            spec, num_kv_heads=4, head_size=128, model_name="M",
        )
        adapter._get_cached_manager = lambda config: (FakeManager(), None)
        return adapter

    def test_completed_send_measurements_reselect_profile(self):
        adapter = self.create_adapter()
        first = adapter.compress(FakeKV(), "a")
        self.assertEqual(first.metadata["ctrl_profile_id"], "fast")
        self.assertEqual(adapter.get_controller_stats()["pending_feedback"], 1)

        adapter.update_controller("a", 800, actual_jct=True)
        self.assertEqual(adapter.get_controller_stats()["update_count"], 1)
        self.assertEqual(adapter.get_controller_stats()["pending_feedback"], 0)

        # Two completed, 10 MiB / 1 s NCCL measurements move B to 10 MiB/s.
        for _ in range(2):
            adapter.observe_transport({
                "request_id": "transfer-test", "payload_bytes": 10 << 20,
                "nccl_s": 1.0,
            })
        second = adapter.compress(FakeKV(), "b")
        self.assertEqual(second.metadata["ctrl_profile_id"], "compact")
        self.assertTrue(adapter.get_controller_stats()["bandwidth_dynamic"])

    def test_sender_proxy_feedback_uses_completed_event_only(self):
        adapter = self.create_adapter(feedback_mode="send_proxy")
        adapter.compress(FakeKV(), "a")
        self.assertEqual(adapter.get_controller_stats()["update_count"], 0)
        adapter.observe_transport({
            "payload_bytes": 10 << 20, "nccl_s": .2,
        }, request_id="a")
        self.assertEqual(adapter.get_controller_stats()["update_count"], 1)
        self.assertEqual(adapter.get_controller_stats()["pending_feedback"], 0)
        self.assertEqual(adapter.get_controller_stats()["cooldown_count"], 0)


if __name__ == "__main__":
    unittest.main()
