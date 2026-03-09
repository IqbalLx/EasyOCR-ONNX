# Contributing to EasyOCR-ONNX

This document is for developers who want to understand, maintain, or extend this codebase. It covers the architecture, the porting decisions made, known gotchas, and how to add support for new languages or models.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Deep-Dive](#pipeline-deep-dive)
  - [Detection Pipeline](#detection-pipeline)
  - [Recognition Pipeline](#recognition-pipeline)
- [Porting Decisions](#porting-decisions)
  - [Why the Legacy TorchScript Exporter](#why-the-legacy-torchscript-exporter)
  - [The RecognizerONNXWrapper](#the-recognizeconnxwrapper)
  - [The `text` Input Pruning](#the-text-input-pruning)
  - [Preprocessing Without torch](#preprocessing-without-torch)
- [Known Gotchas](#known-gotchas)
  - [CTC Blank Token Must Not Be Zeroed Out](#ctc-blank-token-must-not-be-zeroed-out)
  - [Character Set Must Match Exactly](#character-set-must-match-exactly)
  - [Image Channel Order](#image-channel-order)
  - [Quantized vs Non-Quantized Models](#quantized-vs-non-quantized-models)
- [Development Setup](#development-setup)
- [Re-Exporting Models](#re-exporting-models)
- [Extension Guide](#extension-guide)
  - [Adding a New Language](#adding-a-new-language)
  - [Adding Beam Search Decoding](#adding-beam-search-decoding)
  - [Swapping the Detector](#swapping-the-detector)
- [Code Map](#code-map)
- [Testing Against EasyOCR](#testing-against-easyocr)

---

## Architecture Overview

The project has a clean separation between **model conversion** (dev-only, requires PyTorch) and **inference** (production, PyTorch-free).

```
┌──────────────────────────────────────────────────────────┐
│                   Dev / Conversion                        │
│                                                          │
│  notebooks/convert.py                                    │
│    ├─ Loads PyTorch weights via easyocr                  │
│    ├─ Wraps recognizer to fix AdaptiveAvgPool2d          │
│    ├─ Exports both models with torch.onnx.export()       │
│    └─ Verifies numerical equivalence with onnxruntime    │
│                                                          │
│  Requires: easyocr, torch, onnx, onnxruntime             │
└──────────────────────────┬───────────────────────────────┘
                           │ produces .onnx files
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  Production / Inference                   │
│                                                          │
│  src/reader.py                                           │
│    ├─ Reader class (main API)                            │
│    ├─ Detection pre/post-processing (pure numpy/cv2)     │
│    ├─ Recognition pre/post-processing (pure numpy/PIL)   │
│    ├─ CTC greedy decoder (pure numpy)                    │
│    └─ ONNX Runtime sessions for inference                │
│                                                          │
│  Requires: numpy, onnxruntime, opencv-python-headless,   │
│            Pillow                                        │
└──────────────────────────────────────────────────────────┘
```

The key insight is that EasyOCR's pre/post-processing is almost entirely NumPy and OpenCV already. Only three things required PyTorch:

1. The model forward pass → replaced with `onnxruntime.InferenceSession.run()`
2. `torchvision.transforms.ToTensor()` + tensor normalization → replaced with numpy arithmetic
3. `torch.FloatTensor` padding in `NormalizePAD` → replaced with `np.zeros` + slicing

## Pipeline Deep-Dive

### Detection Pipeline

```
Input Image (any size, RGB)
    │
    ▼
resize_aspect_ratio()          Resize to fit canvas_size, pad to multiples of 32
    │
    ▼
normalize_mean_variance()      ImageNet normalization: (x - mean*255) / (var*255)
    │
    ▼
np.transpose (HWC → CHW)      Rearrange axes for the conv network
    │
    ▼
CRAFT ONNX model               Input: (1, 3, H, W) float32
    │                           Output: (1, H/2, W/2, 2) — [text_score, link_score]
    ▼
get_det_boxes()                Threshold → connected components → min-area rectangles
    │
    ▼
adjust_result_coordinates()    Scale boxes back to original image dimensions
    │
    ▼
group_text_box()               Merge nearby boxes into text lines
    │
    ▼
(horizontal_list, free_list)   Axis-aligned boxes and rotated boxes
```

**CRAFT model details:**
- Fully convolutional (VGG16-BN backbone + U-Net decoder), no Linear layers
- Accepts any H×W as long as both are multiples of 32
- Output spatial dims are exactly half the input (due to the U-Net design)
- Two output channels: text region heatmap and character link heatmap

### Recognition Pipeline

```
Detected region (from detection)
    │
    ▼
Crop from grayscale image       Using bounding box coordinates
    │
    ▼
Resize to (imgH, W')            Maintain aspect ratio, imgH=64
    │
    ▼
NormalizePAD                    ToTensor equivalent → normalize to [-1, 1] → right-pad
    │
    ▼
Recognizer ONNX model           Input: (B, 1, 64, W) float32
    │                            Output: (B, T, 97) — logits per timestep per class
    ▼
softmax → argmax                Greedy CTC decoding
    │
    ▼
CTC collapse                    Remove blanks, merge repeated characters
    │
    ▼
(text, confidence)              Final recognized string + per-character confidence
```

**Recognizer model details:**
- VGG feature extractor → AdaptiveAvgPool → BiLSTM × 2 → Linear
- Fixed height of 64 pixels, variable width
- 97 output classes: [blank] + 96 characters (digits, punctuation, €, A-Z, a-z)
- Uses CTC (Connectionist Temporal Classification) alignment — no attention

**Two-pass recognition:** If a crop's confidence falls below `contrast_ths` (default 0.1), a second pass runs with contrast-adjusted preprocessing. The higher-confidence result is kept.

## Porting Decisions

### Why the Legacy TorchScript Exporter

We use `torch.onnx.export(dynamo=False)` — the legacy TorchScript-based exporter — for both models. The reasons:

1. **Detector (CRAFT):** The newer dynamo exporter works but tries to downconvert to opset 17, which fails on the `Resize` operator. The legacy exporter handles it fine.

2. **Recognizer:** The dynamo exporter fails entirely because `torch.export` cannot trace through the LSTM's `flatten_parameters()` call. The legacy exporter handles LSTMs natively via its `symbolic_opset9` LSTM handler.

We use **opset 18** as the target version, which is the minimum recommended for PyTorch 2.x exports.

### The RecognizerONNXWrapper

The original recognizer's forward pass contains:

```python
visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
```

`AdaptiveAvgPool2d((None, 1))` means "keep the first spatial dimension as-is, pool the second down to 1." When the first dimension is dynamic (depends on input width), the TorchScript exporter cannot convert this to ONNX because ONNX's `AveragePool` needs static output sizes.

The wrapper replaces this with the mathematically equivalent:

```python
visual_feature = torch.mean(visual_feature, dim=3, keepdim=True)
```

This is a simple reduction that ONNX supports natively with the `ReduceMean` operator. The wrapper is only used during conversion — at runtime, the ONNX model is self-contained.

**Numerical verification:** The wrapper produces results within float32 epsilon (~1e-5 max absolute diff) of the original.

### The `text` Input Pruning

The recognizer's `forward(self, input, text)` takes a `text` parameter that exists for CTC training but is **never used during inference**. During ONNX export, the TorchScript tracer detects this dead input and prunes it from the graph. The resulting ONNX model has only one input (`input`), not two.

The runtime code handles this gracefully by checking `session.get_inputs()` for actual input names rather than hardcoding them.

### Preprocessing Without torch

Three torch-dependent preprocessing operations were replaced:

| Original (torch) | Replacement (numpy/PIL) | Location |
|---|---|---|
| `transforms.ToTensor()` | `np.array(img) / 255.0` | `_normalize_pad()` |
| `tensor.sub_(0.5).div_(0.5)` | `(arr - 0.5) / 0.5` | `_normalize_pad()` |
| `torch.FloatTensor(*size).fill_(0)` + slice assignment | `np.zeros(size)` + slice assignment | `_normalize_pad()` |
| `torch.cat([t.unsqueeze(0) ...], 0)` | `np.stack(arrays, axis=0)` | `_prepare_recognizer_input()` |
| `F.softmax(preds, dim=2)` | `_softmax()` (manual impl) | `_recognizer_predict()` |

## Known Gotchas

### CTC Blank Token Must Not Be Zeroed Out

**This is the most dangerous bug if you're modifying the recognition pipeline.**

EasyOCR's `recognizer_predict` zeroes out probabilities at `ignore_idx` positions before argmax decoding. The `ignore_idx` list is built from user-specified `ignore_char` — characters the user wants to exclude from results.

**Critically, index 0 (the CTC blank token `[blank]`) is NOT in this list by default.** The blank token is essential for CTC decoding:

- It separates repeated characters: `[H, H, blank, e, l, l, l, blank, o]` → `"Hello"`
- Without blanks, argmax would pick the highest non-blank class at every timestep, producing garbage

During development, we initially set `ignore_idx = [0]` (thinking blank should be ignored), which caused the recognizer to output random characters despite the model weights being correct. The fix was `ignore_idx = []`.

### Character Set Must Match Exactly

The `ENGLISH_G2_CHARACTERS` string in `reader.py` must be **byte-for-byte identical** to the character list used during model training. The mapping is positional: character at index `i` in the list corresponds to logit output index `i+1` (index 0 is reserved for `[blank]`).

The English G2 character set is 96 characters and includes the `€` symbol. If you're adding a new language, extract the character list from `easyocr.config.recognition_models[generation][model_name]['characters']`.

### Image Channel Order

The detection pipeline expects **RGB** input (matching EasyOCR's `loadImage` which uses `skimage.io.imread`). The `reformat_input()` function handles conversion from various input formats.

The recognition pipeline expects **grayscale** input. The `readtext()` method handles this split automatically.

OpenCV's `imread` returns BGR by default — the code accounts for this in `reformat_input()`.

### Quantized vs Non-Quantized Models

EasyOCR applies `torch.quantization.quantize_dynamic` to the recognizer when running on CPU. Our ONNX models are exported from the **non-quantized** weights, so there are minor confidence score differences (typically <0.03) compared to EasyOCR's CPU mode. The recognized text is identical.

If you need exact numerical parity with EasyOCR CPU output, you could export from the quantized model — but ONNX quantization is a separate concern (see ONNX Runtime's quantization tools).

## Development Setup

```sh
# Clone the repo
git clone <repo-url>
cd EasyOCR-ONNX

# Install all dependencies (production + dev)
uv sync --group dev

# This installs:
#   Production: numpy, onnxruntime, opencv-python-headless, Pillow
#   Dev: easyocr (pulls in torch, torchvision), onnx, onnxscript,
#        ipykernel, matplotlib, opencv-python
```

The dev group installs `easyocr` which transitively installs PyTorch. This is only needed for:
- Running `notebooks/convert.py` (model export)
- Running comparison tests against the original EasyOCR

### Running the conversion

```sh
cd notebooks
uv run python convert.py
```

This will:
1. Load PyTorch models via easyocr from `models/*.pth`
2. Export to `models/*.onnx`
3. Verify numerical equivalence
4. Print model sizes

The `.pth` files must be present in `models/`. If they're missing, run easyocr once with `download_enabled=True` to fetch them.

## Re-Exporting Models

If EasyOCR updates their model weights, re-export:

```sh
# 1. Update easyocr
uv add --group dev easyocr@latest

# 2. Delete old models
rm models/*.pth models/*.onnx

# 3. Download new weights (easyocr will auto-download)
uv run python -c "
import easyocr
easyocr.Reader(['en'], gpu=False, model_storage_directory='./models', download_enabled=True)
"

# 4. Re-export
cd notebooks && uv run python convert.py

# 5. Verify the reader still works
cd .. && uv run python main.py <test_image>
```

## Extension Guide

### Adding a New Language

To add support for a new language (e.g., Japanese):

1. **Find the model config** in easyocr:

   ```python
   from easyocr.config import recognition_models
   model = recognition_models['gen2']['japanese_g2']
   print(model['characters'])  # character set
   print(model['filename'])    # weight file name
   ```

2. **Download the weights:**

   ```python
   easyocr.Reader(['ja'], model_storage_directory='./models', download_enabled=True)
   ```

3. **Check the model architecture.** EasyOCR has two generations:
   - `generation1` → `easyocr.model.model.Model` (ResNet feature extractor)
   - `generation2` → `easyocr.model.vgg_model.Model` (VGG feature extractor)

   Both share the same `AdaptiveAvgPool2d((None, 1))` issue and need the `RecognizerONNXWrapper`.

4. **Check the network params:**

   ```python
   # gen1 models use:
   {'input_channel': 1, 'output_channel': 512, 'hidden_size': 512}
   # gen2 models use:
   {'input_channel': 1, 'output_channel': 256, 'hidden_size': 256}
   ```

5. **Export the model** by modifying `convert.py` — load the new model, wrap it, and export. The wrapper works for both gen1 and gen2 architectures.

6. **Add the character set** to `reader.py`:

   ```python
   JAPANESE_G2_CHARACTERS = "..."  # from step 1
   ```

7. **Update `Reader.__init__`** to accept a `lang` parameter or let users pass the character set directly (which is already supported via the `character` parameter).

### Adding Beam Search Decoding

The current implementation only supports greedy decoding. To add beam search:

1. Port the `ctcBeamSearch` function from `easyocr/utils.py` — it's already pure Python/NumPy.

2. Wire it into `_recognizer_predict()` alongside the greedy path:

   ```python
   if decoder == "beamsearch":
       preds_str = ctc_beam_search(preds_prob, converter.character, ...)
   ```

3. The `decode_wordbeamsearch` variant also exists and uses dictionary-based constraints — this requires the language dictionary files from `easyocr/dict/`.

### Swapping the Detector

EasyOCR also supports `dbnet18` as an alternative detector. To add it:

1. The DBNet model is at `easyocr/DBNet/` — it's a different architecture but the export process is similar.
2. The post-processing is different (in `easyocr/detection_db.py`), so you'd need to port that as well.
3. The `Reader.detect()` method would need a flag to switch between CRAFT and DBNet post-processing.

## Code Map

### `src/reader.py` (~1100 lines)

The single production source file. Organized into sections:

| Lines | Section | Description |
|---|---|---|
| 35–83 | Image loading | `_load_image()`, `reformat_input()` — accept various input types |
| 91–128 | Detector preprocessing | `_normalize_mean_variance()`, `_resize_aspect_ratio()` |
| 136–247 | Detector postprocessing | `_get_det_boxes()`, `_adjust_result_coordinates()` — threshold, connected components, box extraction |
| 255–386 | Box grouping | `_group_text_box()` — merge nearby detections into text lines |
| 394–493 | Crop extraction | `_four_point_transform()`, `_get_image_list()` — cut text regions from the image |
| 501–575 | Recognizer preprocessing | `_normalize_pad()`, `_prepare_recognizer_input()` — resize, normalize, pad to batch |
| 583–639 | CTC decoder | `CTCLabelConverter` — greedy CTC decoding (collapse blanks + repeats) |
| 647–704 | Recognizer inference | `_recognizer_predict()` — run ONNX session + decode |
| 706–716 | Character set | `ENGLISH_G2_CHARACTERS` constant |
| 725–1094 | `Reader` class | Public API: `__init__`, `detect()`, `recognize()`, `readtext()` |

### `notebooks/convert.py` (~290 lines)

Dev-only conversion script, structured as a Python percent-script (Jupyter-compatible cells):

| Cell | Description |
|---|---|
| §1 | Load models via `easyocr.Reader` |
| §2 | Export CRAFT detector to ONNX |
| §3 | Define `RecognizerONNXWrapper`, export recognizer to ONNX |
| §4 | Verify ONNX outputs match PyTorch (including dynamic size tests) |
| §5 | Print model file sizes |

## Testing Against EasyOCR

There's no formal test suite yet. To verify correctness manually:

```python
import easyocr
from easyocr_onnx import Reader

# PyTorch baseline
torch_reader = easyocr.Reader(
    ['en'], gpu=False, model_storage_directory='./models', download_enabled=False
)
torch_results = torch_reader.readtext('test.jpg')

# ONNX implementation
onnx_reader = Reader(
    detector_onnx_path='models/craft_mlt_25k.onnx',
    recognizer_onnx_path='models/english_g2.onnx',
)
onnx_results = onnx_reader.readtext('test.jpg')

# Compare
for (tb, tt, tc), (ob, ot, oc) in zip(torch_results, onnx_results):
    assert tt == ot, f"Text mismatch: {tt!r} vs {ot!r}"
    print(f"✅ {tt:30s}  torch={tc:.4f}  onnx={oc:.4f}")
```

Expected: identical text, identical bounding boxes, confidence within ~0.03 (due to quantization differences on CPU).