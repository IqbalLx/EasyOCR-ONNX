#!/usr/bin/env python3
"""
Benchmark: EasyOCR (PyTorch) vs EasyOCR-ONNX (ONNX Runtime)

Measures and compares:
  1. Cold-start initialization time (model loading)
  2. Warm-start inference latency (detection, recognition, end-to-end)
  3. Throughput (images/sec)
  4. Peak memory usage (RSS)
  5. Scaling across image sizes
  6. Multi-provider comparison (CPU, CoreML, etc.)

Designed for Apple Silicon (M1 Air) — supports CPU and CoreML execution providers.

Usage:
    uv run python benchmarks/bench_pytorch_vs_onnx.py
    uv run python benchmarks/bench_pytorch_vs_onnx.py --rounds 20 --warmup 5
    uv run python benchmarks/bench_pytorch_vs_onnx.py --quick
    uv run python benchmarks/bench_pytorch_vs_onnx.py --output results.json
    uv run python benchmarks/bench_pytorch_vs_onnx.py --provider coreml
    uv run python benchmarks/bench_pytorch_vs_onnx.py --provider cpu coreml   # compare both
    uv run python benchmarks/bench_pytorch_vs_onnx.py --provider all          # auto-detect all

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
from dataclasses import dataclass, field
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
# Provider name mapping
# ---------------------------------------------------------------------------

_PROVIDER_ALIASES: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "azure": "AzureExecutionProvider",
}

# Short display names for table headers
_PROVIDER_SHORT_NAMES: dict[str, str] = {
    "CPUExecutionProvider": "ONNX-CPU",
    "CoreMLExecutionProvider": "ONNX-CoreML",
    "CUDAExecutionProvider": "ONNX-CUDA",
    "TensorrtExecutionProvider": "ONNX-TRT",
    "DmlExecutionProvider": "ONNX-DML",
    "OpenVINOExecutionProvider": "ONNX-OV",
    "ROCMExecutionProvider": "ONNX-ROCm",
    "MIGraphXExecutionProvider": "ONNX-MIGraphX",
    "AzureExecutionProvider": "ONNX-Azure",
}


def _resolve_provider(alias: str) -> str:
    """Resolve a short alias like 'coreml' to 'CoreMLExecutionProvider'."""
    lower = alias.lower().strip()
    if lower in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[lower]
    # Maybe they passed the full name already
    if alias.endswith("ExecutionProvider"):
        return alias
    raise ValueError(
        f"Unknown provider alias '{alias}'. "
        f"Known aliases: {', '.join(sorted(_PROVIDER_ALIASES.keys()))}"
    )


def _short_name(provider: str) -> str:
    return _PROVIDER_SHORT_NAMES.get(
        provider, provider.replace("ExecutionProvider", "")
    )


def _get_available_providers() -> list[str]:
    """Return list of ONNX Runtime providers available on this machine."""
    import onnxruntime as ort

    return ort.get_available_providers()


def _resolve_provider_list(raw_args: list[str]) -> list[str]:
    """
    Given CLI args like ['cpu', 'coreml'] or ['all'], return resolved provider list.
    Filters out providers not actually available on this system.
    """
    available = _get_available_providers()

    if len(raw_args) == 1 and raw_args[0].lower() == "all":
        # Auto-detect: return all available providers (excluding Azure which is not
        # a real compute provider)
        exclude = {"AzureExecutionProvider"}
        return [p for p in available if p not in exclude]

    resolved = []
    for alias in raw_args:
        provider = _resolve_provider(alias)
        if provider not in available:
            print(
                f"  \033[93m⚠ Provider '{alias}' ({provider}) not available on this system.\033[0m"
            )
            print(f"    Available: {', '.join(available)}")
            continue
        if provider not in resolved:
            resolved.append(provider)

    return resolved


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
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


def get_current_rss_mb() -> float:
    """Approximate current RSS via /proc or resource."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return get_peak_rss_mb()


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _force_gc():
    """Force garbage collection and give the system a moment."""
    gc.collect()
    gc.collect()
    gc.collect()


def bench_init_onnx(
    rounds: int = 5, providers: list[str] | None = None
) -> tuple[TimingSample, Any]:
    """Benchmark ONNX Reader initialization (cold-start)."""
    from easyocr_onnx import Reader

    if providers is None:
        providers = ["CPUExecutionProvider"]

    backend_name = _short_name(providers[0])
    sample = TimingSample(label="init", backend=backend_name)
    reader = None

    for _ in range(rounds):
        _force_gc()
        t0 = time.perf_counter()
        reader = Reader(
            detector_onnx_path=DETECTOR_ONNX,
            recognizer_onnx_path=RECOGNIZER_ONNX,
            providers=providers,
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


def bench_readtext(
    reader,
    image: np.ndarray,
    warmup: int = 3,
    rounds: int = 10,
    label: str = "readtext",
    backend: str = "onnx",
) -> TimingSample:
    """Benchmark end-to-end readtext for any reader."""
    sample = TimingSample(label=label, backend=backend)

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
    reader,
    image: np.ndarray,
    warmup: int = 3,
    rounds: int = 10,
    label: str = "detect",
    backend: str = "onnx",
) -> TimingSample:
    """Benchmark detection-only for ONNX reader."""
    from easyocr_onnx.reader import reformat_input

    img_rgb, _ = reformat_input(image)
    sample = TimingSample(label=label, backend=backend)

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
    reader,
    image: np.ndarray,
    warmup: int = 3,
    rounds: int = 10,
    label: str = "detect",
) -> TimingSample:
    """Benchmark detection-only for PyTorch reader."""
    sample = TimingSample(label=label, backend="pytorch")

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
    readers: dict[str, Any], images: list[np.ndarray], warmup: int = 2
) -> dict[str, TimingSample]:
    """Measure throughput for all readers: process N images sequentially."""
    n = len(images)
    results: dict[str, TimingSample] = {}

    # Warmup all readers
    for reader in readers.values():
        for img in images[:warmup]:
            reader.readtext(img)

    for backend_name, reader in readers.items():
        _force_gc()
        t0 = time.perf_counter()
        for img in images:
            reader.readtext(img)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        sample = TimingSample(label=f"throughput_{n}_images", backend=backend_name)
        sample.times_ms = [total_ms]
        results[backend_name] = sample

    return results


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _speedup_str(baseline_ms: float, candidate_ms: float) -> str:
    if candidate_ms <= 0 or baseline_ms <= 0:
        return "N/A"
    ratio = baseline_ms / candidate_ms
    if ratio >= 1.0:
        return f"{_GREEN}{ratio:.2f}x faster{_RESET}"
    else:
        return f"{_RED}{1 / ratio:.2f}x slower{_RESET}"


def _speedup_str_plain(baseline_ms: float, candidate_ms: float) -> str:
    """Speedup string without ANSI codes (for measuring width)."""
    if candidate_ms <= 0 or baseline_ms <= 0:
        return "N/A"
    ratio = baseline_ms / candidate_ms
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1 / ratio:.2f}x slower"


def print_multi_comparison_table(
    rows: list[dict[str, TimingSample]],
    backend_order: list[str],
    baseline: str = "pytorch",
):
    """
    Print a comparison table with N backends.

    Each row is a dict of {backend_name: TimingSample} for a single benchmark label.
    backend_order controls column ordering.
    baseline is the backend name used as the reference for speedup calculation.
    """
    if not rows:
        return

    # Determine column widths
    col_width = 16
    label_width = 35
    speedup_width = 16

    # Build header
    header_parts = [f"{'Benchmark':<{label_width}}"]
    for b in backend_order:
        header_parts.append(f"{b:>{col_width}}")
    # Add speedup columns for non-baseline backends (vs baseline)
    has_baseline = baseline in backend_order
    if has_baseline:
        for b in backend_order:
            if b != baseline:
                header_parts.append(f"{'vs ' + baseline:>{speedup_width}}")

    header = " │ ".join(header_parts)
    sep_parts = ["─" * label_width]
    for b in backend_order:
        sep_parts.append("─" * col_width)
    if has_baseline:
        for b in backend_order:
            if b != baseline:
                sep_parts.append("─" * speedup_width)
    sep = "─┼─".join(sep_parts)

    print()
    print(f"{_BOLD}{header}{_RESET}")
    print(sep)

    for row in rows:
        label = ""
        cells = []

        # Label from the first sample we find
        for b in backend_order:
            if b in row:
                label = row[b].label
                break

        cells.append(f"{label:<{label_width}}")

        # Timing columns
        for b in backend_order:
            if b in row:
                s = row[b]
                cell = f"{s.median_ms:>8.1f} ±{s.stdev_ms:>4.0f}"
            else:
                cell = f"{'—':>{col_width}}"
            cells.append(f"{cell:>{col_width}}")

        # Speedup columns
        if has_baseline and baseline in row:
            baseline_ms = row[baseline].median_ms
            for b in backend_order:
                if b != baseline:
                    if b in row:
                        candidate_ms = row[b].median_ms
                        su = _speedup_str(baseline_ms, candidate_ms)
                        # Pad with invisible chars accounting for ANSI codes
                        plain_len = len(_speedup_str_plain(baseline_ms, candidate_ms))
                        pad = speedup_width - plain_len
                        cells.append(" " * max(0, pad) + su)
                    else:
                        cells.append(f"{'—':>{speedup_width}}")

        print(" │ ".join(cells))

    print(sep)
    print()


def print_section(title: str):
    print()
    print(f"{_BOLD}{_CYAN}{'═' * 70}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * 70}{_RESET}")


def print_system_info(onnx_providers_requested: list[str]):
    print_section("System Information")
    print(f"  Platform:       {platform.platform()}")
    print(f"  Machine:        {platform.machine()}")
    print(f"  Python:         {platform.python_version()}")
    print(f"  CPU:            {platform.processor() or 'N/A'}")

    try:
        import onnxruntime as ort

        print(f"  ONNX Runtime:   {ort.__version__}")
        available = ort.get_available_providers()
        print(f"  ORT Providers:  {', '.join(available)}")
        print(
            f"  Benchmarking:   {', '.join(_short_name(p) for p in onnx_providers_requested)}"
        )
    except ImportError:
        print("  ONNX Runtime:   not installed")

    try:
        import torch

        print(f"  PyTorch:        {torch.__version__}")
        if hasattr(torch.backends, "mps"):
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


def verify_output_parity(readers: dict[str, Any], image: np.ndarray) -> bool:
    """Quick sanity check that all readers produce the same text."""
    results: dict[str, list[str]] = {}
    for name, reader in readers.items():
        results[name] = reader.readtext(image, detail=0)

    for name, texts in results.items():
        print(f"  {_DIM}{name:>16} texts: {texts}{_RESET}")

    # Compare all against the first
    names = list(results.keys())
    reference_name = names[0]
    reference = results[reference_name]
    all_match = True

    for name in names[1:]:
        other = results[name]
        if len(reference) != len(other):
            print(
                f"  {_YELLOW}⚠ Different detection count: "
                f"{reference_name}={len(reference)}, {name}={len(other)}{_RESET}"
            )
            all_match = False
            continue
        for rt, ot in zip(reference, other):
            if rt != ot:
                print(
                    f"  {_YELLOW}⚠ Text mismatch: "
                    f"{reference_name}='{rt}' vs {name}='{ot}'{_RESET}"
                )
                all_match = False

    if all_match:
        print(f"  {_GREEN}✓ Output text matches across all backends{_RESET}")

    return all_match


# ---------------------------------------------------------------------------
# Main benchmark suite
# ---------------------------------------------------------------------------


def run_benchmarks(
    rounds: int = 10,
    warmup: int = 3,
    quick: bool = False,
    onnx_providers: list[str] | None = None,
):
    """Run the full benchmark suite."""

    if quick:
        rounds = 5
        warmup = 2

    if onnx_providers is None:
        onnx_providers = ["CPUExecutionProvider"]

    print_system_info(onnx_providers)

    # Check models exist
    if not os.path.isfile(DETECTOR_ONNX) or not os.path.isfile(RECOGNIZER_ONNX):
        print(f"{_RED}ERROR: ONNX models not found in {_MODELS_DIR}{_RESET}")
        print("Run: cd notebooks && uv run python convert.py")
        sys.exit(1)

    # Track all readers: {display_name: reader_instance}
    all_readers: dict[str, Any] = {}
    # Track all init samples: {display_name: TimingSample}
    init_samples: dict[str, TimingSample] = {}

    # backend_order for table columns: pytorch first, then ONNX providers
    backend_order: list[str] = ["pytorch"]

    # All table rows (each row is a dict of backend -> TimingSample)
    table_rows: list[dict[str, TimingSample]] = []

    # -----------------------------------------------------------------------
    # 1. Initialization benchmark
    # -----------------------------------------------------------------------
    print_section("1. Initialization Time (Cold Start)")
    print(f"  Rounds: {rounds}")
    print(f"  ONNX providers: {', '.join(_short_name(p) for p in onnx_providers)}")
    print()

    # Init ONNX readers (one per provider)
    for provider in onnx_providers:
        short = _short_name(provider)
        backend_order.append(short)

        # For CoreML and similar providers, also include CPU as fallback for
        # ops not supported by the accelerator. This is standard ONNX RT practice.
        if provider == "CPUExecutionProvider":
            providers_list = ["CPUExecutionProvider"]
        else:
            providers_list = [provider, "CPUExecutionProvider"]

        rss_before = get_peak_rss_mb()
        sample, reader = bench_init_onnx(rounds=rounds, providers=providers_list)
        sample.backend = short  # fix backend name
        rss_after = get_peak_rss_mb()

        all_readers[short] = reader
        init_samples[short] = sample

        print(
            f"  {short:<16} init: median {sample.median_ms:>8.1f} ms  "
            f"(peak RSS: {rss_after:.1f} MB, Δ{rss_after - rss_before:+.1f} MB)"
        )

    # Init PyTorch reader
    rss_before = get_peak_rss_mb()
    init_pytorch, pytorch_reader = bench_init_pytorch(rounds=rounds)
    rss_after = get_peak_rss_mb()
    all_readers["pytorch"] = pytorch_reader
    init_samples["pytorch"] = init_pytorch

    print(
        f"  {'pytorch':<16} init: median {init_pytorch.median_ms:>8.1f} ms  "
        f"(peak RSS: {rss_after:.1f} MB, Δ{rss_after - rss_before:+.1f} MB)"
    )

    # Speedups for init
    print()
    for provider in onnx_providers:
        short = _short_name(provider)
        su = _speedup_str(init_pytorch.median_ms, init_samples[short].median_ms)
        print(f"  → {short} is {su} than PyTorch at initialization")

    table_rows.append(init_samples)

    # -----------------------------------------------------------------------
    # 2. Sanity check — output parity
    # -----------------------------------------------------------------------
    print_section("2. Output Parity Check")
    sanity_img = make_text_image(640, 480, ["Hello World", "Test 123"])
    verify_output_parity(all_readers, sanity_img)

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
        sizes = sizes[:2]

    print_section("3. End-to-End Inference (readtext)")
    print(f"  Warmup: {warmup} | Rounds: {rounds}")
    print()

    for size_label, w, h in sizes:
        img = make_text_image(w, h)
        label = f"readtext [{size_label}]"

        row: dict[str, TimingSample] = {}

        # PyTorch
        pt_s = bench_readtext(
            pytorch_reader,
            img,
            warmup=warmup,
            rounds=rounds,
            label=label,
            backend="pytorch",
        )
        row["pytorch"] = pt_s

        # ONNX providers
        for provider in onnx_providers:
            short = _short_name(provider)
            reader = all_readers[short]
            s = bench_readtext(
                reader,
                img,
                warmup=warmup,
                rounds=rounds,
                label=label,
                backend=short,
            )
            row[short] = s

        table_rows.append(row)

        # Print inline summary
        parts = [f"  {label:<35}"]
        for b in backend_order:
            if b in row:
                parts.append(f"{b} {row[b].median_ms:>7.1f}ms")
        print(" | ".join(parts))

        # Print speedups
        for provider in onnx_providers:
            short = _short_name(provider)
            if short in row:
                su = _speedup_str(pt_s.median_ms, row[short].median_ms)
                print(f"    {short} vs pytorch: {su}")

    # -----------------------------------------------------------------------
    # 4. Detection-only benchmark
    # -----------------------------------------------------------------------
    print_section("4. Detection Only")
    print(f"  Warmup: {warmup} | Rounds: {rounds}")
    print()

    for size_label, w, h in sizes:
        img = make_text_image(w, h)
        label = f"detect [{size_label}]"

        row = {}

        # PyTorch
        pt_s = bench_detect_pytorch(
            pytorch_reader,
            img,
            warmup=warmup,
            rounds=rounds,
            label=label,
        )
        row["pytorch"] = pt_s

        # ONNX providers
        for provider in onnx_providers:
            short = _short_name(provider)
            reader = all_readers[short]
            s = bench_detect_onnx(
                reader,
                img,
                warmup=warmup,
                rounds=rounds,
                label=label,
                backend=short,
            )
            row[short] = s

        table_rows.append(row)

        parts = [f"  {label:<35}"]
        for b in backend_order:
            if b in row:
                parts.append(f"{b} {row[b].median_ms:>7.1f}ms")
        print(" | ".join(parts))

        for provider in onnx_providers:
            short = _short_name(provider)
            if short in row:
                su = _speedup_str(pt_s.median_ms, row[short].median_ms)
                print(f"    {short} vs pytorch: {su}")

    # -----------------------------------------------------------------------
    # 5. Dense text image (many detections / recognitions)
    # -----------------------------------------------------------------------
    print_section("5. Dense Text Image (stress test)")
    dense_img = make_dense_text_image(1280, 960)
    label = "readtext [dense_1280x960]"
    dense_rounds = max(3, rounds // 2) if not quick else 3

    row = {}

    pt_dense = bench_readtext(
        pytorch_reader,
        dense_img,
        warmup=warmup,
        rounds=dense_rounds,
        label=label,
        backend="pytorch",
    )
    row["pytorch"] = pt_dense

    for provider in onnx_providers:
        short = _short_name(provider)
        reader = all_readers[short]
        s = bench_readtext(
            reader,
            dense_img,
            warmup=warmup,
            rounds=dense_rounds,
            label=label,
            backend=short,
        )
        row[short] = s

    table_rows.append(row)

    parts = [f"  {label:<35}"]
    for b in backend_order:
        if b in row:
            parts.append(f"{b} {row[b].median_ms:>7.1f}ms")
    print(" | ".join(parts))

    for provider in onnx_providers:
        short = _short_name(provider)
        if short in row:
            su = _speedup_str(pt_dense.median_ms, row[short].median_ms)
            print(f"    {short} vs pytorch: {su}")

    # Detection counts
    print()
    for name, reader in all_readers.items():
        det_count = len(reader.readtext(dense_img))
        print(f"  {name:>16} detections: {det_count}")

    # -----------------------------------------------------------------------
    # 6. Throughput benchmark
    # -----------------------------------------------------------------------
    print_section("6. Throughput (sequential batch)")
    n_images = 20 if not quick else 8
    rng = np.random.RandomState(42)
    throughput_images = [
        make_text_image(
            rng.randint(320, 1024),
            rng.randint(240, 768),
            [f"Image {i}", f"Sample text {i * 7}"],
        )
        for i in range(n_images)
    ]

    tp_results = bench_throughput(all_readers, throughput_images, warmup=min(2, warmup))

    row = {}
    print(f"  {n_images} images processed:")
    for b in backend_order:
        if b in tp_results:
            s = tp_results[b]
            row[b] = s
            total_ms = s.times_ms[0]
            ips = n_images / (total_ms / 1000) if total_ms > 0 else 0
            print(f"    {b:<16} {total_ms:>8.0f} ms total  ({ips:.2f} img/sec)")

    table_rows.append(row)

    # Speedup vs pytorch
    if "pytorch" in tp_results:
        pt_total = tp_results["pytorch"].times_ms[0]
        for provider in onnx_providers:
            short = _short_name(provider)
            if short in tp_results:
                su = _speedup_str(pt_total, tp_results[short].times_ms[0])
                print(f"    → {short} is {su}")

    # -----------------------------------------------------------------------
    # 7. Memory summary
    # -----------------------------------------------------------------------
    print_section("7. Memory Usage")
    peak_rss = get_peak_rss_mb()
    print(f"  Peak RSS (entire process): {peak_rss:.1f} MB")
    print(f"  {_DIM}(Includes all backends loaded in same process.{_RESET}")
    print(f"  {_DIM} For isolated measurements, run with --memory-only.){_RESET}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print_section("Summary Table")
    print_multi_comparison_table(table_rows, backend_order, baseline="pytorch")

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
            "onnx_providers": [_short_name(p) for p in onnx_providers],
        },
        "benchmarks": [],
    }

    for row in table_rows:
        entry: dict[str, Any] = {}
        for b in backend_order:
            if b in row:
                entry[b] = row[b].summary_dict()
        # Compute speedups vs pytorch
        if "pytorch" in row:
            pt_med = row["pytorch"].median_ms
            speedups = {}
            for b in backend_order:
                if b != "pytorch" and b in row and row[b].median_ms > 0:
                    speedups[b] = round(pt_med / row[b].median_ms, 3)
            entry["speedup_vs_pytorch"] = speedups
        results["benchmarks"].append(entry)

    return results


# ---------------------------------------------------------------------------
# Isolated memory benchmark (fork a subprocess for each backend)
# ---------------------------------------------------------------------------


def run_isolated_memory_bench(onnx_providers: list[str] | None = None):
    """
    Measure memory for each backend in isolation using a subprocess.
    This gives accurate per-backend RSS without cross-contamination.
    """
    import subprocess
    import textwrap

    if onnx_providers is None:
        onnx_providers = ["CPUExecutionProvider"]

    print_section("Isolated Memory Measurement (subprocess)")

    python_exe = sys.executable

    # Build scripts for each ONNX provider
    onnx_scripts: list[tuple[str, str]] = []
    for provider in onnx_providers:
        short = _short_name(provider)
        if provider == "CPUExecutionProvider":
            providers_arg = '["CPUExecutionProvider"]'
        else:
            providers_arg = f'["{provider}", "CPUExecutionProvider"]'

        script = textwrap.dedent(f"""\
            import resource, sys, time
            sys.path.insert(0, "{_PROJECT_ROOT / "src"}")
            import numpy as np
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            t0 = time.perf_counter()
            from easyocr_onnx import Reader
            reader = Reader(
                detector_onnx_path="{DETECTOR_ONNX}",
                recognizer_onnx_path="{RECOGNIZER_ONNX}",
                providers={providers_arg},
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
        onnx_scripts.append((short, script))

    # PyTorch script
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

    all_scripts = onnx_scripts + [("PyTorch/EasyOCR", pytorch_script)]

    for label, script in all_scripts:
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
                    f"  {label:<20}  Peak RSS: {rss:>7.1f} MB  "
                    f"(Δ{delta:>+7.1f} MB)  Import+init+infer: {time_ms:.0f} ms"
                )
            else:
                stderr_snippet = result.stderr.strip()[:300]
                print(f"  {label:<20}  {_RED}ERROR{_RESET}: {stderr_snippet}")
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
Provider aliases:
  cpu       → CPUExecutionProvider
  coreml    → CoreMLExecutionProvider    (macOS / Apple Silicon)
  cuda      → CUDAExecutionProvider      (NVIDIA GPU)
  tensorrt  → TensorrtExecutionProvider  (NVIDIA TensorRT)
  dml       → DmlExecutionProvider       (Windows DirectML)
  openvino  → OpenVINOExecutionProvider  (Intel)
  all       → auto-detect all available providers

Examples:
  uv run python benchmarks/bench_pytorch_vs_onnx.py
  uv run python benchmarks/bench_pytorch_vs_onnx.py --quick
  uv run python benchmarks/bench_pytorch_vs_onnx.py --provider coreml
  uv run python benchmarks/bench_pytorch_vs_onnx.py --provider cpu coreml
  uv run python benchmarks/bench_pytorch_vs_onnx.py --provider all
  uv run python benchmarks/bench_pytorch_vs_onnx.py --rounds 20 --warmup 5
  uv run python benchmarks/bench_pytorch_vs_onnx.py --output results.json
  uv run python benchmarks/bench_pytorch_vs_onnx.py --memory-only --provider coreml
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
        "--provider",
        nargs="+",
        default=["cpu"],
        metavar="PROVIDER",
        help=(
            "ONNX Runtime execution provider(s) to benchmark. "
            "Use short aliases: cpu, coreml, cuda, tensorrt, dml, openvino, "
            "or 'all' to auto-detect. Can specify multiple: --provider cpu coreml. "
            "(default: cpu)"
        ),
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

    # Resolve providers
    onnx_providers = _resolve_provider_list(args.provider)
    if not onnx_providers:
        print(f"{_RED}ERROR: No valid ONNX providers available.{_RESET}")
        sys.exit(1)

    print(f"  Providers: {', '.join(_short_name(p) for p in onnx_providers)}")

    if args.memory_only:
        run_isolated_memory_bench(onnx_providers)
        return

    results = run_benchmarks(
        rounds=args.rounds,
        warmup=args.warmup,
        quick=args.quick,
        onnx_providers=onnx_providers,
    )

    if not args.no_memory:
        run_isolated_memory_bench(onnx_providers)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"{_GREEN}Results saved to {output_path}{_RESET}")

    print(f"{_BOLD}Done!{_RESET}")
    print()


if __name__ == "__main__":
    main()
