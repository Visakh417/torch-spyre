# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pytest
import unittest
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    skipIfTorchDynamo,
    TemporaryFileName,
    TestCase,
)


Test_spyre = None
if hasattr(torch, "spyre"):
    Test_spyre = torch.spyre.is_available()
else:
    Test_spyre = False


class TestSpyreProfiler(TestCase):
    @unittest.skipUnless(Test_spyre, "requires spyre device")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_basic_profile(self):
        device = "spyre"
        x = torch.randn(4, device=device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            with_stack=True,
        ) as prof:
            x *= 2

        names = [e.name for e in prof.events()]
        self.assertTrue("aten::mul_" in names)

    @unittest.skipUnless(Test_spyre, "require spyre device")
    def test_event_list(self):
        device = torch.device("spyre")
        x, y = (torch.rand((4, 4), dtype=torch.float16).to(device) for _ in range(2))

        with profile(with_stack=True) as prof:
            z = torch.add(x, y)
            z = F.gelu(z)
            z = torch.sum(z)

        event_list = torch.autograd.profiler_util.EventList(prof.events())

        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            with open(fname) as f:
                json.load(f)

        event_list.table()

    @unittest.skipIf(not Test_spyre, "spyre device required")
    def test_profiler_timestamp_consistency(self):
        """Verify that FunctionEvent timestamps can reconstruct Chrome trace ts values."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            x = torch.randn(32, 32, device="spyre")
            torch.add(x, x)

        trace_start_ns = prof.profiler.kineto_results.trace_start_ns()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            base_time_ns = j.get("baseTimeNanoseconds", 0)

            fe_mm = next((e for e in prof.events() if e.name == "aten::add"), None)
            json_mm = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e["name"] == "aten::add" and e["ph"] == "X"
                ),
                None,
            )

            absolute_ns = int(fe_mm.time_range.start * 1000) + trace_start_ns
            recovered_ts = (absolute_ns - base_time_ns) / 1000
            self.assertEqual(
                recovered_ts,
                json_mm["ts"],
                msg="Recovered Chrome trace ts doesn't match Json for aten::add",
            )


def test_package_importable():
    """
    Verify that the torch_spyre.profiler package can be imported
    without requiring Spyre hardware.
    """
    import torch_spyre.profiler  # noqa: F401


def test_chrome_trace_is_valid_json(tmp_path):
    """
    Verify that export_chrome_trace() produces valid JSON with at least one event.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity

    trace_file = tmp_path / "spyre_trace.json"

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        x = torch.randn(10, 10)
        _ = torch.matmul(x, x)

    prof.export_chrome_trace(str(trace_file))

    # Ensure the file exists and contains valid JSON
    assert trace_file.exists(), "Chrome trace file was not created"

    with open(trace_file, "r") as f:
        data = json.load(f)

    # Chrome traces typically contain a "traceEvents" list
    assert isinstance(data, dict), "Trace JSON must be a dictionary"
    assert "traceEvents" in data, "Trace JSON must contain 'traceEvents'"
    assert len(data["traceEvents"]) > 0, "Trace JSON must contain at least one event"


@pytest.mark.requires_spyre_profiler
def test_synchronize_callable():
    """
    Ensure that torch.spyre.synchronize() is callable without error.
    This test requires Spyre hardware and USE_SPYRE_PROFILER=1.
    """
    import torch

    # Verify the attribute exists
    assert hasattr(torch, "spyre"), "torch.spyre namespace is missing"
    assert hasattr(torch.spyre, "synchronize"), "torch.spyre.synchronize() is missing"

    x = torch.randn(64, 64, device="spyre")
    y = torch.randn(64, 64, device="spyre")

    z = torch.matmul(x, y)

    torch.spyre.synchronize()

    result = z.cpu()

    assert result.numel() == 64 * 64
    assert torch.isfinite(result).all()


@pytest.mark.requires_spyre_profiler
def test_kineto_memcpy_and_memset_events_captured():
    """
    Confirm that H2D memcpy, D2H memcpy, and memset events are captured
    in the kineto-spyre Chrome trace when profiling with PrivateUse1.

    Triggered operations:
      - H2D: cpu_tensor.to("spyre")
      - memset: torch.zeros(..., device="spyre")
      - D2H: device_tensor.cpu()

    Note: P2P (device-to-device) transfers are out of scope for this test.
    """
    cpu_src = torch.randn(64, 64, dtype=torch.float16)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        device_tensor = cpu_src.to("spyre")
        _ = torch.zeros(64, 64, dtype=torch.float16, device="spyre")
        _ = device_tensor.cpu()
        torch.spyre.synchronize()

    with TemporaryFileName(mode="w+") as fname:
        prof.export_chrome_trace(fname)
        with open(fname) as f:
            trace = json.load(f)

    assert "traceEvents" in trace, (
        "Chrome trace is missing 'traceEvents' key — export may have failed"
    )
    events = trace["traceEvents"]

    # "gpu_memcpy" / "gpu_memset" are emitted by libkineto's ActivityType::type_string()
    # (upstream kineto, not kineto-spyre-specific) and have been stable across all kineto
    # versions used by torch-spyre. kineto-spyre maps Spyre memory activities to these
    # standard ActivityType values; the Chrome trace writer produces these category strings.
    h2d_events = [
        e
        for e in events
        if e.get("cat") == "gpu_memcpy" and "HtoD" in e.get("name", "")
    ]
    assert h2d_events, (
        "Expected at least one H2D memcpy event in the kineto-spyre trace"
    )

    d2h_events = [
        e
        for e in events
        if e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "")
    ]
    assert d2h_events, (
        "Expected at least one D2H memcpy event in the kineto-spyre trace"
    )

    memset_events = [e for e in events if e.get("cat") == "gpu_memset"]
    assert memset_events, "Expected at least one memset event in the kineto-spyre trace"


def test_no_zero_timestamp_or_duration(trace_dir: str) -> None:
    """Check every Chrome trace file in ``trace_dir`` for zero timestamps or durations.

    Args:
        trace_dir: Path to a directory containing one or more Chrome trace JSON
            files produced by the profiler.

    For each file the function checks three categories of complete ('X') events:

    * **cpu_op / all complete events** — zero ``ts`` across all events.
    * **gpu_memcpy** — HtoD and DtoH transfers with ``dur == 0``.
    * **kernel** — compute kernel events with ``dur == 0``.

    Both ``ts`` and ``dur`` are in nanoseconds as declared by ``displayTimeUnit``.
    """
    trace_files = sorted(
        os.path.join(trace_dir, f)
        for f in os.listdir(trace_dir)
        if f.endswith(".json")
    )
    assert trace_files, f"No .json trace files found in '{trace_dir}'"

    for trace_file in trace_files:
        print(f"\n[{os.path.basename(trace_file)}]")

        with open(trace_file) as f:
            data = json.load(f)

        if "traceEvents" not in data:
            print("  SKIP — no 'traceEvents' key")
            continue

        complete_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        if not complete_events:
            print("  SKIP — no complete ('X') events")
            continue

        # --- zero timestamps (all complete events) ---
        zero_ts = [e for e in complete_events if e.get("ts", 1) == 0]
        if zero_ts:
            print(
                f"  FAIL — {len(zero_ts)} event(s) have ts == 0: "
                + ", ".join(e.get("name", "<unnamed>") for e in zero_ts)
            )
        else:
            print("  OK   — no zero timestamps")

        # --- gpu_memcpy: HtoD and DtoH with dur == 0 ---
        memcpy_events = [e for e in complete_events if e.get("cat") == "gpu_memcpy"]
        htod_zero = [
            e for e in memcpy_events
            if "HtoD" in e.get("name", "") and e.get("dur", 1) == 0
        ]
        dtoh_zero = [
            e for e in memcpy_events
            if "DtoH" in e.get("name", "") and e.get("dur", 1) == 0
        ]
        if htod_zero:
            print(f"  FAIL — {len(htod_zero)} HtoD memcpy event(s) have dur == 0: "
                  + ", ".join(e.get("name", "<unnamed>") for e in htod_zero))
        else:
            print("  OK   — no zero-duration HtoD memcpy events")

        if dtoh_zero:
            print(f"  FAIL — {len(dtoh_zero)} DtoH memcpy event(s) have dur == 0: "
                  + ", ".join(e.get("name", "<unnamed>") for e in dtoh_zero))
        else:
            print("  OK   — no zero-duration DtoH memcpy events")

        # --- kernel: compute kernel events with dur == 0 ---
        kernel_events = [e for e in complete_events if e.get("cat") == "kernel"]
        kernel_zero_dur = [e for e in kernel_events if e.get("dur", 1) == 0]
        if kernel_zero_dur:
            print(
                f"  FAIL — {len(kernel_zero_dur)} kernel event(s) have dur == 0: "
                + ", ".join(e.get("name", "<unnamed>") for e in kernel_zero_dur)
            )
        else:
            print("  OK   — no zero-duration kernel events")
