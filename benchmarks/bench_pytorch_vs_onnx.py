#!/usr/bin/env python3
"""
Benchmark: EasyOCR (PyTorch) vs EasyOCR-ONNX (ONNX Runtime)

Measures and compares:
  1. Cold-start initialization time (model loading)
  2. Warm-start inference latency (detection, recognition, end-to-end)
  3. Throughput (images/sec)
  4. Peak memory usage (RSS)
  5. Scaling across image sizes

Designed for Apple Silicon (M1 Air) — CPU-only, single-threaded comparison.

Usage:
    uv run python benchmarks/bench_pytorch_vs_onnx.py
    uv run python benchmarks/bench_pytorch_vs_onnx.py --rounds 20 --warmup 5
    uv run python benchmarks/bench_pytorch_vs_onnx.py --quick          # fewer rounds for a fast check
    uv run python benchmarks/bench_pytorch_vs_onnx.py --output results.json

Requirements:
    - Dev dependencies installed: uv sync --group dev
    - ONNX models present in models/  (run notebooks/convert.py first)
    - PyTorch models present in models/ (for easyocr)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_MODELS_DIR = _PROJECT_ROOT / "models"

DETECTOR_ONNX = str(_MODELS_DIR / "craft_mlt_25k.onnx")
RECOGNIZER_ONNX = str(_MODELS_DIR / "english_g2.onnx")


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class TimingSample:
    """Raw timing samples for a single benchmark scenario."""

    label: str
    backend: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) >= 2 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        sorted_t = sorted(self.times_ms)
        idx = int(len(sorted_t) * 0.95)
        return sorted_t[min(idx, len(sorted_t) - 1)]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "backend": self.backend,
            "rounds": len(self.times_ms),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "stdev_ms": round(self.stdev_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
        }


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------


def make_text_image(
    width: int, height: int, texts: list[str] | None = None
) -> np.ndarray:
    """Create a white-background RGB image with black text rendered via OpenCV."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    if texts is None:
        texts = ["Hello World", "Test 123", "Benchmark"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font relative to image height
    scale = max(0.5, height / 300)
    thickness = max(1, int(height / 150))
    y_step = height // (len(texts) + 1)

    for i, text in enumerate(texts):
        y = y_step * (i + 1)
        x = max(10, width // 20)
        cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img


def make_dense_text_image(width: int = 1280, height: int = 960) -> np.ndarray:
    """Create an image with many lines of text — stresses both detection and recognition."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "The quick brown fox jumps over the lazy dog",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump",
        "The five boxing wizards jump quickly",
        "Sphinx of black quartz judge my vow",
        "Two driven jocks help fax my big quiz",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789 !@#$%^&*() Test",
        "EasyOCR ONNX Runtime Benchmark",
    ]
    scale = max(0.4, height / 600)
    thickness = max(1, int(height / 400))
    y_step = height // (len(lines) + 1)

    for i, line in enumerate(lines):
        y = y_step * (i + 1)
        cv2.putText(img, line, (20, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------


def get_peak_rss_mb() -> float:
    """Get peak RSS in MB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports in bytes, Linux in KB
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


def get_current_rss_mb() -> float:
    """Approximate current RSS via /proc or resource."""
    try:
        # Linux
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    # Fallback: use peak (imprecise but available everywhere)
    return get_peak_rss_mb()


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _force_gc():
    """Force garbage collection and give the system a moment."""
    gc.collect()
    gc.collect()
    gc.collect()


def bench_init_onnx(rounds: int = 5) -> tuple[TimingSample, Any]:
    """Benchmark ONNX Reader initialization (cold-start)."""
    from easyocr_onnx import Reader

    sample = TimingSample(label="init", backend="onnx")
    reader = None

    for _ in range(rounds):
        _force_gc()
        t0 = time.perf_counter()
        reader = Reader(
            detector_onnx_path=DETECTOR_ONNX,
            recognizer_onnx_path=RECOGNIZER_ONNX,
        )
        t1 = time.perf_counter()
        sample.times_ms.append((t1 - t0) * 1000)

    return sample, reader


def bench_init_pytorch(rounds: int = 5) -> tuple[TimingSample, Any]:
    """Benchmark EasyOCR (PyTorch) initialization (cold-start)."""
    import easyocr

    sample = TimingSample(label="init", backend="pytorch")
    reader = None

    for _ in range(rounds):
        _force_gc()
        t0 = time.perf_counter()
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=str(_MODELS_DIR),
            download_enabled=False,
        )
        t1 = time.perf_counter()
        sample.times_ms.append((t1 - t0) * 1000)

    return sample, reader


def bench_readtext_onnx(
    reader,
    image: np.ndarray,
    warmup: int = 3,
    rounds: int = 10,
    label: str = "readtext",
) -> TimingSample:
    """Benchmark end-to-end readtext for ONNX reader."""
    sample = TimingSample(label=label, backend="onnx")

    # Warmup
    for _ in range(warmup):
        reader.readtext(image)

    for _ in range(rounds):
        _force_gc()
        t0 = time.perf_counter()
        reader.readtext(image)
        t1 = time.perf_counter()
        sample.times_ms.append((t1 - t0) * 1000)

    return sample


def bench_readtext_pytorch(
    reader,
    image: np.ndarray,
    warmup: int = 3,
    rounds: int = 10,
    label: str = "readtext",
) -> TimingSample:
    """Benchmark end-to-end readtext for PyTorch reader."""
    sample = TimingSample(label=label, backend="pytorch")

    # Warmup
    for _ in range(warmup):
        reader.readtext(image)

    for _ in range(rounds):
        _force_gc()
        t0 = time.perf_counter()
        reader.readtext(image)
        t1 = time.perf_counter()
        sample.times_ms.append((t1 - t0) * 1000)

    return sample


def bench_detect_onnx(
    reader, image: np.ndarray, warmup: int = 3, rounds: int = 10, label: str = "detect"
) -> TimingSample:
    """Benchmark detection-only for ONNX reader."""
    from easyocr_onnx.reader import reformat_input

    img_rgb, _ = reformat_input(image)
    sample = TimingSample(label=label, backend="onnx")

    for _ in range(warmup):
        reader.detect(img_rgb)

    for _ in range(rounds):
        _force_gc()
        t0 = time.perf_counter()
        reader.detect(img_rgb)
        t1 = time.perf_counter()
        sample.times_ms.append((t1 - t0) * 1000)

    return sample


def bench_detect_pytorch(
    reader, image: np.ndarray, warmup: int = 3, rounds: int = 10, label: str = "detect"
) -> TimingSample:
    """Benchmark detection-only for PyTorch reader (via readtext with recognition disabled is not straightforward, so we time the full pipeline and note it)."""
    # EasyOCR doesn't expose detect() as a clean public API in the same way,
    # but reader.detect() exists. Let's use it.
    sample = TimingSample(label=label, backend="pytorch")

    # easyocr expects the image to be loaded; it handles BGR/RGB internally
    for _ in range(warmup):
        reader.detect(image)

    for _ in range(rounds):
        _force_gc()
        t0 = time.perf_counter()
        reader.detect(image)
        t1 = time.perf_counter()
        sample.times_ms.append((t1 - t0) * 1000)

    return sample


def bench_throughput(
    onnx_reader, pytorch_reader, images: list[np.ndarray], warmup: int = 2
) -> tuple[TimingSample, TimingSample]:
    """Measure throughput: process N images sequentially, report total time."""
    n = len(images)

    # Warmup both
    for img in images[:warmup]:
        onnx_reader.readtext(img)
        pytorch_reader.readtext(img)

    # ONNX
    _force_gc()
    t0 = time.perf_counter()
    for img in images:
        onnx_reader.readtext(img)
    t1 = time.perf_counter()
    onnx_total_ms = (t1 - t0) * 1000
    onnx_sample = TimingSample(label=f"throughput_{n}_images", backend="onnx")
    onnx_sample.times_ms = [onnx_total_ms]

    # PyTorch
    _force_gc()
    t0 = time.perf_counter()
    for img in images:
        pytorch_reader.readtext(img)
    t1 = time.perf_counter()
    pytorch_total_ms = (t1 - t0) * 1000
    pytorch_sample = TimingSample(label=f"throughput_{n}_images", backend="pytorch")
    pytorch_sample.times_ms = [pytorch_total_ms]

    return onnx_sample, pytorch_sample


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

# ANSI colors for terminal
_BOLD = "\033[1m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _speedup_str(pytorch_ms: float, onnx_ms: float) -> str:
    if onnx_ms <= 0 or pytorch_ms <= 0:
        return "N/A"
    ratio = pytorch_ms / onnx_ms
    if ratio >= 1.0:
        return f"{_GREEN}{ratio:.2f}x faster{_RESET}"
    else:
        return f"{_RED}{1 / ratio:.2f}x slower{_RESET}"


def print_comparison_table(pairs: list[tuple[TimingSample, TimingSample]]):
    """Print a side-by-side comparison table."""
    # Header
    header = f"{'Benchmark':<35} │ {'PyTorch (ms)':>14} │ {'ONNX (ms)':>14} │ {'Speedup':>20}"
    sep = "─" * 35 + "─┼─" + "─" * 14 + "─┼─" + "─" * 14 + "─┼─" + "─" * 20
    print()
    print(f"{_BOLD}{header}{_RESET}")
    print(sep)

    for pytorch_s, onnx_s in pairs:
        label = pytorch_s.label
        pt_med = pytorch_s.median_ms
        ox_med = onnx_s.median_ms
        speedup = _speedup_str(pt_med, ox_med)
        pt_str = f"{pt_med:>10.1f} ±{pytorch_s.stdev_ms:>4.0f}"
        ox_str = f"{ox_med:>10.1f} ±{onnx_s.stdev_ms:>4.0f}"
        print(f"{label:<35} │ {pt_str:>14} │ {ox_str:>14} │ {speedup:>32}")

    print(sep)
    print()


def print_section(title: str):
    print()
    print(f"{_BOLD}{_CYAN}{'═' * 70}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * 70}{_RESET}")


def print_system_info():
    print_section("System Information")
    print(f"  Platform:       {platform.platform()}")
    print(f"  Machine:        {platform.machine()}")
    print(f"  Python:         {platform.python_version()}")
    print(f"  CPU:            {platform.processor() or 'N/A'}")

    try:
        import onnxruntime as ort

        print(f"  ONNX Runtime:   {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  ORT Providers:  {', '.join(providers)}")
    except ImportError:
        print("  ONNX Runtime:   not installed")

    try:
        import torch

        print(f"  PyTorch:        {torch.__version__}")
        print(f"  MPS available:  {torch.backends.mps.is_available()}")
    except ImportError:
        print("  PyTorch:        not installed")

    try:
        import easyocr

        print(f"  EasyOCR:        {easyocr.__version__}")
    except (ImportError, AttributeError):
        pass

    print(f"  Peak RSS:       {get_peak_rss_mb():.1f} MB")
    print()


# ---------------------------------------------------------------------------
# Correctness check (sanity)
# ---------------------------------------------------------------------------


def verify_output_parity(onnx_reader, pytorch_reader, image: np.ndarray) -> bool:
    """Quick sanity check that both readers produce the same text."""
    onnx_result = onnx_reader.readtext(image, detail=0)
    pytorch_result = pytorch_reader.readtext(image, detail=0)

    print(f"  {_DIM}ONNX texts:    {onnx_result}{_RESET}")
    print(f"  {_DIM}PyTorch texts: {pytorch_result}{_RESET}")

    if len(onnx_result) != len(pytorch_result):
        print(
            f"  {_YELLOW}⚠ Different number of detections: ONNX={len(onnx_result)}, PyTorch={len(pytorch_result)}{_RESET}"
        )
        return False

    all_match = True
    for ot, pt in zip(onnx_result, pytorch_result):
        if ot != pt:
            print(f"  {_YELLOW}⚠ Text mismatch: ONNX='{ot}' vs PyTorch='{pt}'{_RESET}")
            all_match = False

    if all_match:
        print(f"  {_GREEN}✓ Output text matches between backends{_RESET}")

    return all_match


# ---------------------------------------------------------------------------
# Main benchmark suite
# ---------------------------------------------------------------------------


def run_benchmarks(rounds: int = 10, warmup: int = 3, quick: bool = False):
    """Run the full benchmark suite."""

    if quick:
        rounds = 5
        warmup = 2

    print_system_info()

    # Check models exist
    if not os.path.isfile(DETECTOR_ONNX) or not os.path.isfile(RECOGNIZER_ONNX):
        print(f"{_RED}ERROR: ONNX models not found in {_MODELS_DIR}{_RESET}")
        print("Run: cd notebooks && uv run python convert.py")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Initialization benchmark
    # -----------------------------------------------------------------------
    print_section("1. Initialization Time (Cold Start)")
    print(f"  Rounds: {rounds}")

    rss_before = get_peak_rss_mb()

    init_onnx, onnx_reader = bench_init_onnx(rounds=rounds)
    rss_after_onnx = get_peak_rss_mb()
    print(
        f"  ONNX init:    median {init_onnx.median_ms:.1f} ms  (peak RSS: {rss_after_onnx:.1f} MB, Δ{rss_after_onnx - rss_before:+.1f} MB)"
    )

    rss_before_pt = get_peak_rss_mb()
    init_pytorch, pytorch_reader = bench_init_pytorch(rounds=rounds)
    rss_after_pytorch = get_peak_rss_mb()
    print(
        f"  PyTorch init: median {init_pytorch.median_ms:.1f} ms  (peak RSS: {rss_after_pytorch:.1f} MB, Δ{rss_after_pytorch - rss_before_pt:+.1f} MB)"
    )

    speedup = _speedup_str(init_pytorch.median_ms, init_onnx.median_ms)
    print(f"  → ONNX is {speedup} at initialization")

    all_pairs: list[tuple[TimingSample, TimingSample]] = [(init_pytorch, init_onnx)]

    # -----------------------------------------------------------------------
    # 2. Sanity check — output parity
    # -----------------------------------------------------------------------
    print_section("2. Output Parity Check")
    sanity_img = make_text_image(640, 480, ["Hello World", "Test 123"])
    verify_output_parity(onnx_reader, pytorch_reader, sanity_img)

    # -----------------------------------------------------------------------
    # 3. Inference benchmarks at multiple image sizes
    # -----------------------------------------------------------------------
    sizes = [
        ("small_320x240", 320, 240),
        ("medium_640x480", 640, 480),
        ("large_1280x960", 1280, 960),
        ("xlarge_1920x1080", 1920, 1080),
    ]
    if quick:
        sizes = sizes[:2]  # only small and medium in quick mode

    print_section("3. End-to-End Inference (readtext)")
    print(f"  Warmup: {warmup} | Rounds: {rounds}")

    for size_label, w, h in sizes:
        img = make_text_image(w, h)
        label = f"readtext [{size_label}]"
        onnx_s = bench_readtext_onnx(
            onnx_reader, img, warmup=warmup, rounds=rounds, label=label
        )
        pytorch_s = bench_readtext_pytorch(
            pytorch_reader, img, warmup=warmup, rounds=rounds, label=label
        )
        all_pairs.append((pytorch_s, onnx_s))
        speedup = _speedup_str(pytorch_s.median_ms, onnx_s.median_ms)
        print(
            f"  {label:<35} ONNX {onnx_s.median_ms:>8.1f} ms | PyTorch {pytorch_s.median_ms:>8.1f} ms | {speedup}"
        )

    # -----------------------------------------------------------------------
    # 4. Detection-only benchmark
    # -----------------------------------------------------------------------
    print_section("4. Detection Only")
    print(f"  Warmup: {warmup} | Rounds: {rounds}")

    for size_label, w, h in sizes:
        img = make_text_image(w, h)
        label = f"detect [{size_label}]"
        onnx_s = bench_detect_onnx(
            onnx_reader, img, warmup=warmup, rounds=rounds, label=label
        )
        pytorch_s = bench_detect_pytorch(
            pytorch_reader, img, warmup=warmup, rounds=rounds, label=label
        )
        all_pairs.append((pytorch_s, onnx_s))
        speedup = _speedup_str(pytorch_s.median_ms, onnx_s.median_ms)
        print(
            f"  {label:<35} ONNX {onnx_s.median_ms:>8.1f} ms | PyTorch {pytorch_s.median_ms:>8.1f} ms | {speedup}"
        )

    # -----------------------------------------------------------------------
    # 5. Dense text image (many detections / recognitions)
    # -----------------------------------------------------------------------
    print_section("5. Dense Text Image (stress test)")
    dense_img = make_dense_text_image(1280, 960)
    label = "readtext [dense_1280x960]"
    dense_rounds = max(3, rounds // 2) if not quick else 3

    onnx_dense = bench_readtext_onnx(
        onnx_reader, dense_img, warmup=warmup, rounds=dense_rounds, label=label
    )
    pytorch_dense = bench_readtext_pytorch(
        pytorch_reader, dense_img, warmup=warmup, rounds=dense_rounds, label=label
    )
    all_pairs.append((pytorch_dense, onnx_dense))

    speedup = _speedup_str(pytorch_dense.median_ms, onnx_dense.median_ms)
    print(
        f"  {label:<35} ONNX {onnx_dense.median_ms:>8.1f} ms | PyTorch {pytorch_dense.median_ms:>8.1f} ms | {speedup}"
    )

    # How many text boxes were detected?
    onnx_results = onnx_reader.readtext(dense_img)
    pytorch_results = pytorch_reader.readtext(dense_img)
    print(f"  Detections:  ONNX={len(onnx_results)}, PyTorch={len(pytorch_results)}")

    # -----------------------------------------------------------------------
    # 6. Throughput benchmark
    # -----------------------------------------------------------------------
    print_section("6. Throughput (sequential batch)")
    n_images = 20 if not quick else 8
    throughput_images = [
        make_text_image(
            np.random.randint(320, 1024),
            np.random.randint(240, 768),
            [f"Image {i}", f"Sample text {i * 7}"],
        )
        for i in range(n_images)
    ]
    onnx_tp, pytorch_tp = bench_throughput(
        onnx_reader, pytorch_reader, throughput_images, warmup=min(2, warmup)
    )
    all_pairs.append((pytorch_tp, onnx_tp))

    onnx_ips = n_images / (onnx_tp.times_ms[0] / 1000) if onnx_tp.times_ms[0] > 0 else 0
    pytorch_ips = (
        n_images / (pytorch_tp.times_ms[0] / 1000) if pytorch_tp.times_ms[0] > 0 else 0
    )

    print(f"  {n_images} images processed:")
    print(
        f"    ONNX:    {onnx_tp.times_ms[0]:>8.0f} ms total  ({onnx_ips:.2f} img/sec)"
    )
    print(
        f"    PyTorch: {pytorch_tp.times_ms[0]:>8.0f} ms total  ({pytorch_ips:.2f} img/sec)"
    )
    speedup = _speedup_str(pytorch_tp.times_ms[0], onnx_tp.times_ms[0])
    print(f"    → ONNX is {speedup}")

    # -----------------------------------------------------------------------
    # 7. Memory summary
    # -----------------------------------------------------------------------
    print_section("7. Memory Usage")
    peak_rss = get_peak_rss_mb()
    print(f"  Peak RSS (entire process): {peak_rss:.1f} MB")
    print(f"  {_DIM}(Includes both PyTorch + ONNX loaded in same process.{_RESET}")
    print(
        f"  {_DIM} For isolated measurements, run each backend in a separate process.){_RESET}"
    )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print_section("Summary Table")
    print_comparison_table(all_pairs)

    # -----------------------------------------------------------------------
    # Collect results for JSON export
    # -----------------------------------------------------------------------
    results = {
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "peak_rss_mb": round(peak_rss, 1),
        },
        "config": {
            "rounds": rounds,
            "warmup": warmup,
            "quick": quick,
        },
        "benchmarks": [
            {
                "pytorch": pytorch_s.summary_dict(),
                "onnx": onnx_s.summary_dict(),
                "speedup": round(pytorch_s.median_ms / onnx_s.median_ms, 3)
                if onnx_s.median_ms > 0
                else None,
            }
            for pytorch_s, onnx_s in all_pairs
        ],
    }

    return results


# ---------------------------------------------------------------------------
# Isolated memory benchmark (fork a subprocess for each backend)
# ---------------------------------------------------------------------------


def run_isolated_memory_bench():
    """
    Measure memory for each backend in isolation using a subprocess.
    This gives accurate per-backend RSS without cross-contamination.
    """
    import subprocess
    import textwrap

    print_section("Isolated Memory Measurement (subprocess)")

    # ONNX memory script
    onnx_script = textwrap.dedent(f"""\
        import resource, sys, time
        sys.path.insert(0, "{_PROJECT_ROOT / "src"}")
        import numpy as np
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        from easyocr_onnx import Reader
        reader = Reader(
            detector_onnx_path="{DETECTOR_ONNX}",
            recognizer_onnx_path="{RECOGNIZER_ONNX}",
        )
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        reader.readtext(img)
        t1 = time.perf_counter()
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1024*1024 if sys.platform == "darwin" else 1024
        print(f"rss_mb={{rss1/scale:.1f}}")
        print(f"delta_mb={{(rss1-rss0)/scale:.1f}}")
        print(f"time_ms={{(t1-t0)*1000:.1f}}")
    """)

    # PyTorch memory script
    pytorch_script = textwrap.dedent(f"""\
        import resource, sys, time
        import numpy as np
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        import easyocr
        reader = easyocr.Reader(
            ["en"], gpu=False,
            model_storage_directory="{_MODELS_DIR}",
            download_enabled=False,
        )
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        reader.readtext(img)
        t1 = time.perf_counter()
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1024*1024 if sys.platform == "darwin" else 1024
        print(f"rss_mb={{rss1/scale:.1f}}")
        print(f"delta_mb={{(rss1-rss0)/scale:.1f}}")
        print(f"time_ms={{(t1-t0)*1000:.1f}}")
    """)

    python_exe = sys.executable  # Use the same Python (important for venv)

    for label, script in [
        ("ONNX Runtime", onnx_script),
        ("PyTorch/EasyOCR", pytorch_script),
    ]:
        try:
            result = subprocess.run(
                [python_exe, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(_PROJECT_ROOT),
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                data = {}
                for line in lines:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        data[k] = float(v)
                rss = data.get("rss_mb", 0)
                delta = data.get("delta_mb", 0)
                time_ms = data.get("time_ms", 0)
                print(
                    f"  {label:<20}  Peak RSS: {rss:>7.1f} MB  (Δ{delta:>+7.1f} MB)  Import+init+infer: {time_ms:.0f} ms"
                )
            else:
                print(
                    f"  {label:<20}  {_RED}ERROR{_RESET}: {result.stderr.strip()[:200]}"
                )
        except subprocess.TimeoutExpired:
            print(f"  {label:<20}  {_YELLOW}TIMEOUT (>120s){_RESET}")
        except Exception as e:
            print(f"  {label:<20}  {_RED}FAILED: {e}{_RESET}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EasyOCR (PyTorch) vs EasyOCR-ONNX (ONNX Runtime)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python benchmarks/bench_pytorch_vs_onnx.py
  uv run python benchmarks/bench_pytorch_vs_onnx.py --quick
  uv run python benchmarks/bench_pytorch_vs_onnx.py --rounds 20 --warmup 5
  uv run python benchmarks/bench_pytorch_vs_onnx.py --output results.json
  uv run python benchmarks/bench_pytorch_vs_onnx.py --memory-only
        """,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of timed iterations per benchmark (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations before timing (default: 3)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer rounds, smaller images"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Save results to JSON file"
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only run isolated memory measurement (subprocess)",
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="Skip isolated memory measurement"
    )

    args = parser.parse_args()

    print()
    print(
        f"{_BOLD}╔══════════════════════════════════════════════════════════════════════╗{_RESET}"
    )
    print(
        f"{_BOLD}║   EasyOCR (PyTorch) vs EasyOCR-ONNX — Performance Benchmark        ║{_RESET}"
    )
    print(
        f"{_BOLD}╚══════════════════════════════════════════════════════════════════════╝{_RESET}"
    )

    if args.memory_only:
        run_isolated_memory_bench()
        return

    results = run_benchmarks(rounds=args.rounds, warmup=args.warmup, quick=args.quick)

    if not args.no_memory:
        run_isolated_memory_bench()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"{_GREEN}Results saved to {output_path}{_RESET}")

    print(f"{_BOLD}Done!{_RESET}")
    print()


if __name__ == "__main__":
    main()
