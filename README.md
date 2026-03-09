# EasyOCR-ONNX

A lightweight OCR engine that runs [EasyOCR](https://github.com/JaidedAI/EasyOCR) models using [ONNX Runtime](https://onnxruntime.ai/) — no PyTorch required at runtime.

The original EasyOCR library depends on PyTorch (~2 GB installed), which is heavy for production deployments. This project converts the PyTorch models to ONNX format and provides a pure NumPy/OpenCV/ONNX Runtime inference pipeline that produces **identical results** with a fraction of the dependency footprint. No Pillow required.

## Features

- **No PyTorch at runtime** — only `numpy`, `onnxruntime`, `opencv-python-headless`
- **Same accuracy** — numerically verified against the original EasyOCR output
- **Same API** — `reader.readtext(image)` works just like EasyOCR
- **Dynamic input sizes** — both models support variable image dimensions
- **CLI included** — `easyocr-onnx` console command installed automatically
- **English text** — currently ships with the CRAFT detector + English G2 recognizer

## Installation

### From PyPI

```sh
pip install easyocr-onnx
```

If you need to pass `PIL.Image` objects as input (optional):

```sh
pip install easyocr-onnx[pil]
```

### From source

```sh
# Requires Python 3.12+
git clone https://github.com/iqbalmaulana/easyocr-onnx.git
cd easyocr-onnx

# Install production deps
uv sync

# Or install with dev deps (easyocr, torch, pytest, etc.)
uv sync --group dev
```

## Getting the ONNX Models

The ONNX model files are not included in the package due to their size (~93 MB total). You need to provide them separately.

**Option A — Convert from PyTorch (requires dev dependencies):**

```sh
uv sync --group dev

cd notebooks
uv run python convert.py
```

This places `craft_mlt_25k.onnx` and `english_g2.onnx` in the `models/` directory.

**Option B — Copy pre-converted models:**

If someone on your team has already run the conversion, copy their `*.onnx` files into a `models/` directory (or any path you prefer).

## Quickstart

### CLI

```sh
# Uses models/ in current directory by default
easyocr-onnx photo.jpg

# Specify model paths explicitly
easyocr-onnx photo.jpg --detector /path/to/craft_mlt_25k.onnx --recognizer /path/to/english_g2.onnx

# Text-only output (no bounding boxes)
easyocr-onnx photo.jpg --detail 0

# Adjust detection thresholds
easyocr-onnx photo.jpg --text-threshold 0.5 --low-text 0.3
```

Run `easyocr-onnx --help` for all available options.

### Python API

```python
from easyocr_onnx import Reader

reader = Reader(
    detector_onnx_path="models/craft_mlt_25k.onnx",
    recognizer_onnx_path="models/english_g2.onnx",
)

results = reader.readtext("photo.jpg")

for bbox, text, confidence in results:
    print(f"{text} ({confidence:.2f})")
```

## API Reference

### `Reader`

```python
from easyocr_onnx import Reader

Reader(
    detector_onnx_path: str,       # Path to CRAFT detector ONNX model
    recognizer_onnx_path: str,     # Path to recognizer ONNX model
    character: str | None = None,  # Character set (defaults to English G2)
    imgH: int = 64,                # Recognizer input height
    providers: list[str] | None = None,  # ONNX Runtime providers (default: CPU)
)
```

#### `reader.readtext(image, ...) -> list`

Main entry point. Detects and recognizes text in an image.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes`, `ndarray`, `PIL.Image` | — | Image input (file path, raw bytes, numpy array, or PIL image if Pillow installed) |
| `detail` | `int` | `1` | `0` = text only, `1` = (bbox, text, confidence) |
| `decoder` | `str` | `"greedy"` | Decoding strategy. Only `"greedy"` is currently supported |
| `min_size` | `int` | `20` | Minimum text region size in pixels |
| `text_threshold` | `float` | `0.7` | Text confidence threshold |
| `low_text` | `float` | `0.4` | Text low-bound score |
| `link_threshold` | `float` | `0.4` | Link confidence threshold |
| `canvas_size` | `int` | `2560` | Maximum image dimension for detection |
| `mag_ratio` | `float` | `1.0` | Image magnification ratio |
| `contrast_ths` | `float` | `0.1` | Contrast threshold for second-pass recognition |
| `adjust_contrast` | `float` | `0.5` | Target contrast for low-confidence crops |
| `filter_ths` | `float` | `0.003` | Minimum confidence to include in results |
| `output_format` | `str` | `"standard"` | `"standard"` (list of tuples) or `"dict"` |

**Returns** (when `detail=1`):

```python
[
    ([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "detected text", 0.95),
    ...
]
```

#### `reader.detect(img, ...) -> (horizontal_list, free_list)`

Run text detection only. Returns grouped bounding boxes.

#### `reader.recognize(img_grey, horizontal_list, free_list, ...) -> list`

Run text recognition only on pre-detected regions.

### Using GPU

Pass ONNX Runtime execution providers to enable GPU acceleration:

```python
reader = Reader(
    detector_onnx_path="models/craft_mlt_25k.onnx",
    recognizer_onnx_path="models/english_g2.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

See [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/) for available options (CUDA, TensorRT, CoreML, DirectML, etc.).

## Project Structure

```
EasyOCR-ONNX/
├── src/
│   └── easyocr_onnx/           # Published package
│       ├── __init__.py          #   Public API: Reader, reformat_input, ...
│       ├── reader.py            #   Core ONNX-based OCR reader (~1100 lines)
│       ├── cli.py               #   Console script entry point
│       └── py.typed             #   PEP 561 typing marker
├── tests/
│   ├── conftest.py              # Shared fixtures: dummy images, model paths
│   ├── test_preprocessing.py    # Unit tests — no ONNX models needed (62 tests)
│   └── test_integration.py      # Integration tests — require ONNX models (41 tests)
├── models/                      # Model weights (not in git, ~93 MB total)
│   ├── craft_mlt_25k.onnx      #   CRAFT detector ONNX model
│   ├── english_g2.onnx         #   English G2 recognizer ONNX model
│   ├── craft_mlt_25k.pth       #   PyTorch weights (dev only)
│   └── english_g2.pth          #   PyTorch weights (dev only)
├── notebooks/
│   └── convert.py               # PyTorch → ONNX conversion (dev only)
├── main.py                      # Legacy demo (delegates to easyocr_onnx.cli)
├── pyproject.toml               # Build config, deps, metadata (hatchling)
├── README.md
├── CONTRIBUTING.md
└── AGENT.md
```

## Models

### CRAFT Text Detector (`craft_mlt_25k.onnx`)

- **Architecture:** VGG16-BN backbone + U-Net style upsampling
- **Input:** `(batch, 3, H, W)` — RGB image, H/W must be multiples of 32
- **Output:** `(batch, H/2, W/2, 2)` — per-pixel text score + link score heatmaps
- **Fully convolutional** — accepts any image size (no fixed input dimensions)
- **Size:** ~79 MB

### English G2 Recognizer (`english_g2.onnx`)

- **Architecture:** VGG feature extractor → BiLSTM → Linear (CTC-based)
- **Input:** `(batch, 1, 64, W)` — grayscale text crop, height fixed at 64, width dynamic
- **Output:** `(batch, T, 97)` — per-timestep logits over 97 character classes
- **Characters:** digits, punctuation, symbols, space, `€`, uppercase A-Z, lowercase a-z
- **Size:** ~14 MB

## Comparison with EasyOCR

| | EasyOCR | EasyOCR-ONNX |
|---|---|---|
| Runtime deps | PyTorch + torchvision + Pillow (~2 GB) | onnxruntime + numpy + opencv (~50 MB) |
| GPU support | CUDA via PyTorch | CUDA, TensorRT, CoreML, DirectML via ONNX Runtime |
| Accuracy | Baseline | Identical (numerically verified) |
| Languages | 80+ languages | English (extensible) |
| Beam search | ✅ | ❌ (greedy only, for now) |
| Installable from PyPI | ✅ | ✅ (`pip install easyocr-onnx`) |

## Testing

Tests use programmatically generated images (no external test fixtures required).

```sh
# Install dev dependencies (includes pytest)
uv sync --group dev

# Run all tests (103 total)
uv run pytest

# Unit tests only — no ONNX models needed (62 tests)
uv run pytest tests/test_preprocessing.py -v

# Integration tests only — requires ONNX models in models/ (41 tests)
uv run pytest tests/test_integration.py -v
```

**Unit tests** (`test_preprocessing.py`) exercise all preprocessing, postprocessing, CTC decoding, and utility functions without any model files. They also verify that `reader.py` has no top-level Pillow import.

**Integration tests** (`test_integration.py`) run the full detect → recognize pipeline on dummy images with rendered text. They are automatically skipped if ONNX models are not found in `models/`.

## Building & Publishing

This project uses [hatchling](https://hatch.pypa.io/) as the build backend and [uv](https://docs.astral.sh/uv/) for package management.

```sh
# Build source distribution and wheel
uv build

# Build without uv-specific source overrides (recommended before publishing)
uv build --no-sources

# Publish to PyPI (requires a PyPI API token)
uv publish --token $PYPI_TOKEN

# Publish to TestPyPI first
uv publish --index testpypi

# Bump version before publishing
uv version --bump patch   # 0.1.0 → 0.1.1
uv version --bump minor   # 0.1.0 → 0.2.0
uv version --bump major   # 0.1.0 → 1.0.0
```

Built artifacts are placed in `dist/`:
- `easyocr_onnx-{version}.tar.gz` — source distribution
- `easyocr_onnx-{version}-py3-none-any.whl` — wheel (pure Python)

## License

The model weights originate from [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) (Apache 2.0).