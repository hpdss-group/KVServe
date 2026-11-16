# KVServe service-aware online controller (algorithm prototype)

This implementation follows the paper's **two-tier profile selection** approach:
a precomputed 3D Pareto / lower-envelope shortlist, an analytical latency
prediction, and an ε-greedy residual-learning bandit. Runtime operations are
CPU-only O(1) lookups across **at most three** preselected profiles per request.

It is an **algorithm prototype integrated with the vLLM KV connector**, not a
claim of fully calibrated production JCT control. The online learning target
and instrumentation are documented below.

## 1. Calibrate measured profiles

Create a JSON array of **measured, executable** profiles. Each entry uses:

~~~json
{
  "profile_id": "profile-A",
  "metadata": {
    "model_name": "your-model",
    "dataset": "your-workload",
    "quality_metric": "accuracy"
  },
  "compression_config": {
    "pipeline": ["quantizer"],
    "quantizer_config": {
      "model_name": "your-model"
    }
  },
  "performance_metrics": {
    "compression_ratio": 3.2,
    "encode_speed_mbps": 1800.0,
    "decode_speed_mbps": 2300.0,
    "harmonic_speed_mbps": 1009.7561
  },
  "quality_metrics": {"accuracy": 0.96, "metric_name": "accuracy"}
}
~~~

**Values above illustrate the schema only; they are not KVServe benchmark
results.** The quantizer_config block must be replaced with actual, validated
pipeline parameters. Do not use the demo configs for real P/D compression.
Do not use measurements from different model/workload/quality metrics in one
library; hardware, TP rank, tensor layout, and compiler versions should match
the deployment.

The compression ratio is defined as **uncompressed bytes / transferred bytes**.
Throughput is in **MiB/s** (despite legacy field names ending in "mbps").
The effective two-stage codec throughput used by the analytical model is

    1 / S = 1 / S_encode + 1 / S_decode

It is **not** the conventional harmonic mean with a factor of two. The
library builder validates this when both stage throughputs are present.
A codec without individual stage metrics can still be selected, but cannot
train the sender-side proxy feedback mode automatically.

Quality is a fraction in [0, 1], calibrated for a given task/metric. Missing
quality or incompatible pipeline fields cause a build-time error.

## 2. Build the offline profile library

From the repository root:

~~~bash
python3 -m kvserve_v1.compression.controller.build_library \
  --profiles /path/to/measured_profiles.json \
  --output /path/to/controller_profiles.json \
  --bandwidth-min 1 --bandwidth-max 100000
~~~

The compiler:
1. Removes dominated profiles in (quality, compression ratio, throughput).
2. Constructs quality buckets with conservative boundaries.
3. Finds **exact pairwise intersections** of the affine model in 1/B:
   T_p/V = 1/S_p + (1/B)/CR_p.
4. Writes the model optimum and up to two nearby candidates for each interval.
5. Validates the resulting library when loaded by the controller.

The builder and controller are pure Python, with no GPU dependency. Unlike a
runtime exhaustive search, only the few offline-shortlisted candidates are
evaluated on each request. Profiles should be generated from real offline
benchmarks, not from invented compression/accuracy numbers.

## 3. Run the controller through the vLLM connector

Use the same library on prefill and decode nodes. The sender selects a profile
for each request; its executable compression configuration is embedded into
the wire metadata, so the receiver does not have to make the same decision.

~~~python
kv_connector_extra_config = {
    "compression": {
        "mode": "controller",
        "library_path": "/path/to/controller_profiles.json",
        "epsilon": 0.1,
        "alpha": 0.2,
        "seed": 2026,
        "feedback_mode": "send_proxy",
        "violation_threshold": 3,
        "violation_window": 10,
        "cooldown_s": 60.0,
        "bandwidth_ewma_alpha": 0.25,
        "bandwidth_min_samples": 2,
        "service_config": {
            "bandwidth_mbps": 12000.0,          # initial MiB/s
            "slo_ms": 2500.0,                   # same JCT metric as model
            "accuracy_requirement": 0.95,
            "t_model_ms": 100.0,
            "workload_id": "your-workload",
        },
    }
}
~~~

The initial bandwidth acts as a fallback until enough completed NCCL sends
have been observed, or when samples become stale. A background completion
callback updates an EWMA from payload bytes / measured NCCL GPU duration;
the main request path does not synchronize the GPU to measure bandwidth.

An invalid context, missing library coverage, insufficient quality, predicted
SLO violation, lack of speedup, or temporary SLO cooldown selects **raw
lossless KV** rather than a potentially unsafe lossy profile. Raw may itself
miss the SLO; fallback only preserves quality, not a guaranteed SLO.

### Online bandit feedback (important distinction)

- **send_proxy** (default for this prototype): once NCCL completes, calculate
  sender-side compress/queue/send elapsed time + *offline-estimated* decode
  time + supplied T_model. Feed that proxy into the analytical residual EWMA.
  This captures online sender-side drift without another network round trip.
  It **is not measured request JCT**, so it does *not* trigger the paper's
  JCT-based SLO cooldown. It only works for profiles with real
  decode_speed_mbps measurements.
- **manual**: do not learn from sender proxies. Feed true, completed request
  JCT from a serving/router completion callback:

~~~python
compressor.update_controller(
    request_id, observed_jct_ms, actual_jct=True
)
~~~

True completed-JCT feedback updates both the model residual and SLO cooldown.
The context is removed after its first update to avoid duplicate learning.
For direct algorithm simulations, call
OnlineController.update(decision_context, observed_jct_ms).

Because the router's request lifecycle is not part of KVServe's external
connector API, this prototype does not implement a cross-node full-JCT
telemetry service. Likewise, independent TP workers may have independent
bandit states and do not perform a new distributed policy broadcast. Both
are intentionally left for a production-grade integration.

## 4. Tests and CPU-only demonstration

~~~bash
python3 -m unittest discover -s tests/controller -p 'test_*.py' -v
python3 examples/controller_demo.py
~~~

The demo generates temporary **synthetic** profiles, shows how bandwidth
switches between raw / fast / compact strategies, and measures CPU-only
selection overhead. It does not validate GPU compression correctness.

Before shipping to production, run P/D GPU end-to-end tests (TP=1/2),
bandwidth throttling, codec/quality checks, and SLO/TTFT/JCT comparisons
against fixed configurations. The raw/compressed round-trip and NCCL event
path were not executed by the CPU-only tests.
