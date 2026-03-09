# AGENT.md

> This file provides structured context for AI coding assistants working on this project.

## Project Summary

EasyOCR-ONNX is a lightweight OCR engine that runs EasyOCR's text detection and recognition models using ONNX Runtime instead of PyTorch. The production runtime has zero PyTorch and zero Pillow dependency — only `numpy`, `onnxruntime`, and `opencv-python-headless`. The project is a publishable PyPI package (`easyocr-onnx`) with a `src/` layout, console script, and full test suite. It consists of a model conversion pipeline (dev-only, requires PyTorch) and a pure NumPy/OpenCV inference reader.

## Tech Stack

- **Language:** Python 3.12+
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build backend:** [hatchling](https://hatch.pypa.io/)
- **Production deps:** numpy, onnxruntime, opencv-python-headless
- **Optional deps:** `easyocr-onnx[pil]` adds Pillow (only needed if passing `PIL.Image` objects as input)
- **Dev deps:** easyocr (brings PyTorch + torchvision), onnx, onnxscript, ipykernel, matplotlib, Pillow, pytest
- **Config:** `pyproject.toml` (PEP 621 + hatchling build system)
- **Test framework:** pytest (tests use programmatically generated dummy images — no test fixtures needed)

## Directory Structure

```
EasyOCR-ONNX/
├── src/
│   └── easyocr_onnx/                # Published package (src layout)
│       ├── __init__.py              #   Public API: Reader, CTCLabelConverter, etc.
│       ├── reader.py                #   Core ONNX-based OCR reader (~1100 lines)
│       ├── cli.py                   #   Console script entry point (easyocr-onnx)
│       └── py.typed                 #   PEP 561 typing marker
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures: dummy images, temp files, model paths
│   ├── test_preprocessing.py        # Unit tests (no ONNX models needed, 62 tests)
│   └── test_integration.py          # Integration tests (require ONNX models, 41 tests)
├── models/                          # Model weights (gitignored, ~93 MB total)
│   ├── craft_mlt_25k.pth           #   CRAFT detector PyTorch weights (dev only)
│   ├── craft_mlt_25k.onnx          #   CRAFT detector ONNX model
│   ├── craft_mlt_25k.onnx.data     #   ONNX external data (large tensor storage)
│   ├── english_g2.pth              #   English G2 recognizer PyTorch weights (dev only)
│   └── english_g2.onnx             #   English G2 recognizer ONNX model
├── notebooks/
│   └── convert.py                   # PyTorch → ONNX conversion (Jupyter percent-script)
├── main.py                          # Legacy demo (delegates to easyocr_onnx.cli)
├── pyproject.toml                   # Build config, deps, metadata (hatchling)
├── CONTRIBUTING.md                  # Developer guide with architecture + gotchas
├── README.md                        # User-facing documentation
└── AGENT.md                         # This file
```

## Package Layout

The project uses the **src layout** required by hatchling:

- **Package name on PyPI:** `easyocr-onnx`
- **Python import name:** `easyocr_onnx`
- **Package root:** `src/easyocr_onnx/`
- **Console script:** `easyocr-onnx` → `easyocr_onnx.cli:main`

```python
# Public API
from easyocr_onnx import Reader
from easyocr_onnx import CTCLabelConverter, ENGLISH_G2_CHARACTERS, reformat_input
```

## Key Files

### `src/easyocr_onnx/__init__.py` — Public API

Exports the public interface: `Reader`, `CTCLabelConverter`, `ENGLISH_G2_CHARACTERS`, `reformat_input`. Defines `__all__` for clean star-imports.

### `src/easyocr_onnx/reader.py` — Production inference (THE core file)

Single-file ONNX-based OCR reader. Zero torch imports, zero Pillow imports. Contains:

- **`Reader` class** (line ~740): Main public API with `readtext()`, `detect()`, `recognize()` methods. Holds two `onnxruntime.InferenceSession` instances (detector + recognizer).
- **Detection preprocessing** (lines ~103–140): `_normalize_mean_variance()`, `_resize_aspect_ratio()` — ImageNet normalization, pad to multiples of 32.
- **Detection postprocessing** (lines ~148–400): `_get_det_boxes()`, `_adjust_result_coordinates()`, `_group_text_box()` — threshold score maps, find connected components, extract bounding boxes, group into text lines.
- **Recognition preprocessing** (lines ~515–590): `_normalize_pad()`, `_prepare_recognizer_input()` — resize grayscale crops to height 64 using `cv2.resize` with `INTER_CUBIC`, normalize to [-1,1], right-pad to batch width.
- **CTC decoder** (lines ~598–655): `CTCLabelConverter.decode_greedy()` — collapse CTC blanks and repeated characters.
- **Recognition inference** (lines ~670–720): `_recognizer_predict()` — run ONNX session, softmax, greedy decode, compute confidence.
- **Character set** (lines ~722–732): `ENGLISH_G2_CHARACTERS` — the 96-character set for the English G2 model (must be byte-for-byte identical to training config).
- **PIL Image support** — handled via lazy `_try_import_pil()` in `reformat_input()`. PIL Images are accepted as input if Pillow is installed, but it is not a required dependency.

### `src/easyocr_onnx/cli.py` — Console script

Argparse-based CLI installed as `easyocr-onnx`. Supports `--detector`, `--recognizer`, `--detail`, `--min-size`, `--canvas-size`, `--text-threshold`, `--low-text`, `--link-threshold` arguments. Lazy-imports `Reader` so `--help` is fast.

### `tests/conftest.py` — Shared test fixtures

Provides:
- Programmatically generated dummy images (RGB, grayscale, with rendered text via `cv2.putText`)
- Temporary image files on disk (PNG, JPEG) — auto-cleaned after tests
- ONNX model path fixtures with `pytest.skip` when models are absent
- `skip_no_models` marker for integration tests

### `tests/test_preprocessing.py` — Unit tests (62 tests)

Tests all preprocessing, postprocessing, and utility functions **without** ONNX models:
- `reformat_input()` — all input types (str, bytes, ndarray, PIL Image)
- Detector preprocessing (`_normalize_mean_variance`, `_resize_aspect_ratio`)
- Detector postprocessing (`_get_det_boxes`, `_adjust_result_coordinates`, `_group_text_box`)
- Geometry helpers (`_four_point_transform`, `_calculate_ratio`, `_compute_ratio_and_resize`)
- Recognizer preprocessing (`_normalize_pad`, `_prepare_recognizer_input`, contrast adjustment)
- CTC decoding (`CTCLabelConverter`)
- Softmax, custom mean, character set validation
- **`TestNoPillowRequired`** — verifies no top-level PIL import in `reader.py` via AST parsing

### `tests/test_integration.py` — Integration tests (41 tests)

Requires ONNX model files in `models/`; auto-skipped if missing:
- `TestReaderInit` — session setup, input names, converter
- `TestDetection` — blank images, text images, box format, thresholds, canvas sizes
- `TestRecognition` — known boxes, blank regions, detail/output formats
- `TestReadtext` — end-to-end pipeline with all input types (ndarray, file, bytes)
- `TestRecognitionAccuracy` — smoke tests for digits, uppercase, lowercase, mixed, punctuation
- `TestEdgeCases` — tiny/wide/tall/noisy images, determinism, min_size filtering

### `notebooks/convert.py` — Model conversion (dev only)

Jupyter-compatible percent-script that:
1. Loads PyTorch models via `easyocr.Reader`
2. Exports CRAFT detector directly with `torch.onnx.export(dynamo=False)`
3. Wraps recognizer in `RecognizerONNXWrapper` to replace `AdaptiveAvgPool2d((None, 1))` with `torch.mean()`, then exports
4. Verifies numerical equivalence between PyTorch and ONNX Runtime outputs
5. Tests dynamic input sizes

### `main.py` — Legacy demo

Delegates directly to `easyocr_onnx.cli:main`. Kept for backward compatibility.

## Common Commands

```sh
# Install production deps only
uv sync

# Install all deps including dev (easyocr, torch, onnx, pytest, etc.)
uv sync --group dev

# Run OCR on an image (via installed console script)
easyocr-onnx path/to/image.jpg

# Run OCR with explicit model paths
easyocr-onnx photo.jpg --detector models/craft_mlt_25k.onnx --recognizer models/english_g2.onnx

# Run OCR via main.py (legacy)
uv run python main.py path/to/image.jpg

# Run all tests (103 total)
uv run pytest

# Run unit tests only (no ONNX models needed, 62 tests)
uv run pytest tests/test_preprocessing.py -v

# Run integration tests only (requires ONNX models in models/, 41 tests)
uv run pytest tests/test_integration.py -v

# Run a specific test class or test
uv run pytest tests/test_integration.py::TestReadtext -v
uv run pytest tests/test_preprocessing.py::TestCTCLabelConverter::test_decode_greedy_simple -v

# Build the package (source dist + wheel)
uv build

# Build without uv-specific source overrides (recommended before publishing)
uv build --no-sources

# Bump version
uv version --bump patch   # 0.1.0 → 0.1.1
uv version --bump minor   # 0.1.0 → 0.2.0
uv version --bump major   # 0.1.0 → 1.0.0

# Publish to PyPI
uv publish --token $PYPI_TOKEN

# Publish to TestPyPI first
uv publish --index testpypi

# Run model conversion (dev only, from notebooks/ directory)
cd notebooks && uv run python convert.py

# Quick comparison test against easyocr
uv run python -c "
import easyocr
from easyocr_onnx import Reader
torch_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models', download_enabled=False)
onnx_reader = Reader(detector_onnx_path='models/craft_mlt_25k.onnx', recognizer_onnx_path='models/english_g2.onnx')
for (_, tt, tc), (_, ot, oc) in zip(torch_reader.readtext('test.jpg'), onnx_reader.readtext('test.jpg')):
    print(f'{tt:30s} torch={tc:.4f} onnx={oc:.4f} match={tt==ot}')
"
```

## Model Architecture

### CRAFT Detector (`craft_mlt_25k.onnx`, ~79 MB)

- **Type:** Fully convolutional (no Linear layers → accepts any image size)
- **Backbone:** VGG16 with batch norm, split into 5 slices
- **Head:** U-Net style upsampling with skip connections → 2-channel output
- **Input:** `(batch, 3, H, W)` float32 — RGB, H and W must be multiples of 32
- **Output:** `(batch, H/2, W/2, 2)` float32 — per-pixel [text_score, link_score] heatmaps
- **Second output:** `(batch, 32, H/2, W/2)` float32 — intermediate features (unused in standard pipeline)
- **Dynamic axes:** batch, height, width are all dynamic in the ONNX model

### English G2 Recognizer (`english_g2.onnx`, ~14 MB)

- **Type:** VGG feature extractor → BiLSTM × 2 → Linear classifier (CTC-based)
- **Input:** `(batch, 1, 64, W)` float32 — grayscale, height fixed at 64, width dynamic
- **Output:** `(batch, T, 97)` float32 — per-timestep logits over 97 classes
- **Classes:** `[blank]` + 96 characters (digits, punctuation, space, €, A-Z, a-z)
- **Note:** The original model's `text` input parameter was pruned during ONNX export because it's unused during inference
- **Dynamic axes:** batch and width are dynamic

## Critical Gotchas

### 1. CTC Blank Token (Index 0) Must NOT Be Zeroed Out

The `ignore_idx` list in recognition post-processing is for user-specified characters to ignore (e.g., `ignore_char='@#'`). It must **never** include index 0 (the CTC blank token). The blank is essential for CTC decoding — without it, the argmax picks random high-probability characters at every timestep and produces garbage output. This was the most time-consuming bug during development.

### 2. Character Set Must Be Exact

`ENGLISH_G2_CHARACTERS` in `reader.py` must be byte-for-byte identical to `recognition_models['gen2']['english_g2']['characters']` from easyocr's config. The mapping is positional — character at index `i` maps to model output index `i+1`. The English G2 set includes the `€` symbol which is easy to miss.

### 3. RecognizerONNXWrapper Is Required for Export

The recognizer uses `AdaptiveAvgPool2d((None, 1))` which the TorchScript ONNX exporter cannot handle with dynamic dimensions. The `RecognizerONNXWrapper` in `convert.py` replaces this with `torch.mean(x, dim=3, keepdim=True)` — mathematically identical, ONNX-exportable.

### 4. Legacy Exporter Required

Both models must be exported with `torch.onnx.export(dynamo=False)` (legacy TorchScript exporter). The newer dynamo exporter fails on the LSTM's `flatten_parameters()` and on opset version downconversion for the Resize operator.

### 5. Quantization Differences

EasyOCR applies `torch.quantization.quantize_dynamic` on CPU. Our ONNX models are non-quantized, so confidence scores differ by up to ~0.03 vs EasyOCR CPU mode. The recognized text is identical.

### 6. Image Channel Order

Detection expects RGB. OpenCV's `imread` returns BGR. The `reformat_input()` function handles conversion, but if you pass numpy arrays directly to `detect()`, ensure they're RGB.

### 7. No Pillow in Production

`reader.py` has **no top-level Pillow import**. All image resizing uses `cv2.resize` with `cv2.INTER_CUBIC`. PIL Image inputs are still accepted via a lazy `_try_import_pil()` helper in `reformat_input()`, but Pillow is an optional dependency (`easyocr-onnx[pil]`) — it is only required if you pass `PIL.Image.Image` objects as input. The test `TestNoPillowRequired.test_no_top_level_pil_import` enforces this via AST parsing.

### 8. Import Paths Changed (src layout)

The package moved from `src/reader.py` (flat `from src.reader import Reader`) to a proper src-layout package. All imports must use:
```python
from easyocr_onnx import Reader                    # public API
from easyocr_onnx.reader import _some_internal_fn   # internal access in tests
```
**Never** use `from src.reader import ...` — that was the pre-packaging import path.

## Current Limitations / TODO

- **English only** — only the English G2 recognizer is converted. Adding other languages requires extracting their character sets and exporting their models (see CONTRIBUTING.md § Extension Guide).
- **Greedy decoding only** — beam search and word beam search from easyocr are not yet ported.
- **No batched detection** — detection processes one image at a time (recognition does handle batches within a single image's crops).
- **DBNet detector not supported** — only CRAFT is converted. EasyOCR also supports `dbnet18`.
- **Models not in package** — `.onnx` and `.pth` files are gitignored and not included in the PyPI distribution due to size. They must be generated via `convert.py` or copied manually.

## Pipeline Flow (for understanding the code)

```
readtext(image)
  │
  ├─ reformat_input(image) → (img_rgb, img_grey)
  │
  ├─ detect(img_rgb)
  │    ├─ _resize_aspect_ratio() → pad to 32x multiples
  │    ├─ _normalize_mean_variance() → ImageNet norm
  │    ├─ detector_session.run() → score maps
  │    ├─ _get_det_boxes() → threshold + connected components + rectangles
  │    ├─ _adjust_result_coordinates() → scale back to original size
  │    └─ _group_text_box() → merge into lines
  │    → (horizontal_list, free_list)
  │
  └─ recognize(img_grey, horizontal_list, free_list)
       ├─ for each box:
       │    ├─ _get_image_list() → crop + resize to height 64
       │    └─ _recognize_crops()
       │         ├─ _prepare_recognizer_input() → cv2.resize + normalize + pad → (B, 1, 64, W)
       │         ├─ _recognizer_predict()
       │         │    ├─ recognizer_session.run() → (B, T, 97) logits
       │         │    ├─ _softmax() → probabilities
       │         │    ├─ argmax → greedy indices
       │         │    └─ converter.decode_greedy() → text strings
       │         ├─ [optional] second pass with contrast adjustment
       │         └─ merge results, pick higher confidence
       → [(bbox, text, confidence), ...]
```

## Test Architecture

Tests use **programmatically generated images** (no external fixtures):

- **Dummy images:** Created with `np.full()` + `cv2.putText()` to render text like "Hello World", "Test 123"
- **Unit tests** (`test_preprocessing.py`): Exercise all pure functions in isolation — no ONNX models required
- **Integration tests** (`test_integration.py`): Run the full pipeline — auto-skipped via `skip_no_models` marker if ONNX models are absent
- **Accuracy smoke tests:** Render known text, run OCR, check that at least some characters are recognised (not strict equality — OCR on rendered OpenCV fonts is inherently fuzzy)
- **Edge case tests:** 1×1 images, very wide/tall images, random noise, RGBA input — verify no crashes

## Build & Publish

The project uses [hatchling](https://hatch.pypa.io/) as the PEP 517 build backend, configured in `pyproject.toml`:

- **`[build-system]`** — declares `hatchling` as the build requirement
- **`[tool.hatch.build.targets.wheel]`** — `packages = ["src/easyocr_onnx"]` tells hatch where the package source lives
- **`[project.scripts]`** — registers `easyocr-onnx` console script → `easyocr_onnx.cli:main`
- **`[project.optional-dependencies]`** — `pil = ["Pillow>=11.0.0"]` for optional PIL support
- **`[project.urls]`** — homepage, repository, and issues links

Build artifacts:
- `dist/easyocr_onnx-{version}.tar.gz` — source distribution
- `dist/easyocr_onnx-{version}-py3-none-any.whl` — pure-Python wheel

Always run `uv build --no-sources` before publishing to ensure the package builds correctly without `tool.uv.sources` overrides.

## Code Style

- Type hints used throughout (`from __future__ import annotations`)
- Private functions prefixed with `_`
- No classes except `Reader` and `CTCLabelConverter`
- All preprocessing/postprocessing functions are pure (no side effects, no state)
- Comments reference the original easyocr source file and function names
- PEP 561 `py.typed` marker included for downstream type checking