# %% [markdown]
# # EasyOCR PyTorch → ONNX Conversion
#
# This notebook converts the two EasyOCR models to ONNX format:
# 1. **CRAFT** — the text detector (input: RGB image, output: text heatmaps)
# 2. **English G2** — the text recognizer (input: grayscale text crop, output: character predictions)

# %%
import os

# %% [markdown]
# ## 1. Load models via EasyOCR
# %%
import easyocr
import numpy as np
import onnx
import torch
import torch.nn as nn

reader = easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory="../models",
    download_enabled=False,
)

# %% [markdown]
# ## 2. Export the CRAFT Detector to ONNX
#
# CRAFT is a fully convolutional network:
# - **Input:** `(batch, 3, H, W)` — RGB image, normalized with ImageNet mean/variance.
#   H and W must be multiples of 32 (due to 4x MaxPool2d stride=2 in the VGG backbone).
#   EasyOCR pads images to multiples of 32 before inference.
# - **Output:** `(batch, H/2, W/2, 2)` — per-pixel text score and link score heatmaps.
#
# Since CRAFT is fully convolutional, H and W are dynamic.

# %%
# Unwrap DataParallel if present
detector = reader.detector
if isinstance(detector, torch.nn.DataParallel):
    detector = detector.module

detector.eval()

# Create a dummy input — (batch=1, channels=3, H=480, W=640)
# Any multiple-of-32 H, W works; we just need a concrete example for tracing.
detector_dummy_input = torch.randn(1, 3, 480, 640)

detector_onnx_path = os.path.join("..", "models", "craft_mlt_25k.onnx")

torch.onnx.export(
    detector,
    (detector_dummy_input,),
    detector_onnx_path,
    dynamo=False,
    opset_version=18,
    input_names=["input"],
    output_names=["output", "feature"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 1: "height_half", 2: "width_half"},
        "feature": {0: "batch", 2: "height_half", 3: "width_half"},
    },
)

print(f"Detector exported to: {detector_onnx_path}")

# Validate the exported model
detector_onnx_model = onnx.load(detector_onnx_path)
onnx.checker.check_model(detector_onnx_model)
print("Detector ONNX model is valid!")

# %% [markdown]
# ## 3. Export the Recognizer to ONNX
#
# The English G2 recognizer architecture (`vgg_model.Model`):
# - **Input 1 (`input`):** `(batch, 1, 64, W)` — grayscale text crop, height fixed at 64,
#   width is variable (right-padded to the longest crop in the batch).
# - **Input 2 (`text`):** `(batch, max_length+1)` — placeholder tensor of zeros used
#   for CTC-based prediction. Not actually consumed by the forward pass in a meaningful way
#   during inference, but required by the model signature.
# - **Output:** `(batch, T, num_classes)` — per-timestep character logits.
#
# The recognizer uses `AdaptiveAvgPool2d((None, 1))` + bidirectional LSTM, so width is dynamic.
#
# **Problem:** `AdaptiveAvgPool2d((None, 1))` with a dynamic first dimension (`None`)
# is not supported by the TorchScript ONNX exporter. We solve this by creating a thin
# wrapper that replaces the adaptive pool with an explicit `torch.mean` over axis 3,
# which produces the exact same result but is ONNX-exportable.

# %%


class RecognizerONNXWrapper(nn.Module):
    """Wrapper around the EasyOCR recognizer that replaces AdaptiveAvgPool2d((None, 1))
    with an explicit mean reduction so ONNX export works with dynamic width.

    The original forward does:
        visual_feature = self.FeatureExtraction(input)                # (B, C_out, H_feat, W_feat)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # (B, W_feat, C_out, 1)
        visual_feature = visual_feature.squeeze(3)                    # (B, W_feat, C_out)
        contextual_feature = self.SequenceModeling(visual_feature)    # (B, W_feat, hidden)
        prediction = self.Prediction(contextual_feature)              # (B, W_feat, num_class)

    AdaptiveAvgPool2d((None, 1)) keeps the first spatial dim (W_feat) and pools the
    second spatial dim (C_out-axis, which after permute is at dim=3) down to 1.
    This is equivalent to torch.mean(..., dim=3, keepdim=True).
    """

    def __init__(self, model):
        super().__init__()
        self.FeatureExtraction = model.FeatureExtraction
        self.SequenceModeling = model.SequenceModeling
        self.Prediction = model.Prediction

    def forward(self, input, text):
        # Feature extraction: (B, 1, 64, W) -> (B, C_out, H_feat, W_feat)
        visual_feature = self.FeatureExtraction(input)

        # Permute to (B, W_feat, C_out, H_feat)
        visual_feature = visual_feature.permute(0, 3, 1, 2)

        # Replace AdaptiveAvgPool2d((None, 1)) with explicit mean over last dim
        # (B, W_feat, C_out, H_feat) -> mean over dim=3 -> (B, W_feat, C_out, 1)
        visual_feature = torch.mean(visual_feature, dim=3, keepdim=True)

        # Squeeze the pooled dimension: (B, W_feat, C_out, 1) -> (B, W_feat, C_out)
        visual_feature = visual_feature.squeeze(3)

        # Sequence modeling (biLSTM): (B, W_feat, C_out) -> (B, W_feat, hidden)
        contextual_feature = self.SequenceModeling(visual_feature)

        # Prediction head: (B, W_feat, hidden) -> (B, W_feat, num_class)
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction


# %%
recognizer = reader.recognizer
if isinstance(recognizer, torch.nn.DataParallel):
    recognizer = recognizer.module

recognizer.eval()

# Wrap the recognizer for ONNX-safe export
recognizer_wrapper = RecognizerONNXWrapper(recognizer)
recognizer_wrapper.eval()

# Verify wrapper produces identical output to original
example_imgW = 200
batch_max_length = int(example_imgW / 10)

test_img = torch.randn(1, 1, 64, example_imgW)
test_txt = torch.zeros(1, batch_max_length + 1, dtype=torch.long)

with torch.no_grad():
    orig_out = recognizer(test_img, test_txt)
    wrap_out = recognizer_wrapper(test_img, test_txt)

np.testing.assert_allclose(orig_out.numpy(), wrap_out.numpy(), rtol=1e-4, atol=1e-5)
print("✅ Wrapper produces identical output to original model")

# %%
recognizer_dummy_image = torch.randn(1, 1, 64, example_imgW)
recognizer_dummy_text = torch.zeros(1, batch_max_length + 1, dtype=torch.long)

recognizer_onnx_path = os.path.join("..", "models", "english_g2.onnx")

torch.onnx.export(
    recognizer_wrapper,
    (recognizer_dummy_image, recognizer_dummy_text),
    recognizer_onnx_path,
    dynamo=False,
    opset_version=18,
    input_names=["input", "text"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 3: "width"},
        "text": {0: "batch", 1: "max_length"},
        "output": {0: "batch", 1: "sequence_length"},
    },
)

print(f"Recognizer exported to: {recognizer_onnx_path}")

# Validate the exported model
recognizer_onnx_model = onnx.load(recognizer_onnx_path)
onnx.checker.check_model(recognizer_onnx_model)
print("Recognizer ONNX model is valid!")

# %% [markdown]
# ## 4. Verify ONNX outputs match PyTorch outputs
#
# Run both PyTorch and ONNX Runtime on the same input and compare numerically.

# %%
import onnxruntime as ort


def verify_detector():
    """Compare CRAFT detector outputs between PyTorch and ONNX Runtime."""
    test_input = torch.randn(1, 3, 480, 640)

    # PyTorch inference
    with torch.no_grad():
        torch_output, torch_feature = detector(test_input)

    # ONNX Runtime inference
    session = ort.InferenceSession(
        detector_onnx_path, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {"input": test_input.numpy()}
    ort_output, ort_feature = session.run(None, ort_inputs)

    # Compare
    np.testing.assert_allclose(torch_output.numpy(), ort_output, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(torch_feature.numpy(), ort_feature, rtol=1e-4, atol=1e-5)
    print("✅ Detector verification passed!")
    print(f"   Input shape:   {list(test_input.shape)}")
    print(f"   Output shape:  {list(ort_output.shape)}")
    print(f"   Feature shape: {list(ort_feature.shape)}")

    # Also test with a different (dynamic) size
    test_input2 = torch.randn(1, 3, 320, 960)
    with torch.no_grad():
        torch_output2, _ = detector(test_input2)
    ort_output2, _ = session.run(None, {"input": test_input2.numpy()})
    np.testing.assert_allclose(torch_output2.numpy(), ort_output2, rtol=1e-4, atol=1e-5)
    print(f"   Dynamic size test ({list(test_input2.shape)}) also passed!")


def verify_recognizer():
    """Compare recognizer outputs between PyTorch and ONNX Runtime."""
    # Test with the same width used during export
    test_image = torch.randn(1, 1, 64, example_imgW)
    test_text = torch.zeros(1, batch_max_length + 1, dtype=torch.long)

    # PyTorch inference (use original recognizer as ground truth)
    with torch.no_grad():
        torch_output = recognizer(test_image, test_text)

    # ONNX Runtime inference
    session = ort.InferenceSession(
        recognizer_onnx_path, providers=["CPUExecutionProvider"]
    )

    # Discover actual input names — the `text` input may have been pruned
    # by the ONNX exporter since the model doesn't actually use it.
    onnx_input_names = [inp.name for inp in session.get_inputs()]
    print(f"   ONNX input names: {onnx_input_names}")

    ort_inputs = {"input": test_image.numpy()}
    if "text" in onnx_input_names:
        ort_inputs["text"] = test_text.numpy()

    ort_output = session.run(None, ort_inputs)[0]

    # Compare
    np.testing.assert_allclose(torch_output.numpy(), ort_output, rtol=1e-4, atol=1e-5)
    print("✅ Recognizer verification passed!")
    print(f"   Image shape:  {list(test_image.shape)}")
    print(f"   Output shape: {list(ort_output.shape)}")

    # Also test with a different (dynamic) width
    test_imgW2 = 320
    test_image2 = torch.randn(1, 1, 64, test_imgW2)
    test_text2 = torch.zeros(1, int(test_imgW2 / 10) + 1, dtype=torch.long)
    with torch.no_grad():
        torch_output2 = recognizer(test_image2, test_text2)

    ort_inputs2 = {"input": test_image2.numpy()}
    if "text" in onnx_input_names:
        ort_inputs2["text"] = test_text2.numpy()

    ort_output2 = session.run(None, ort_inputs2)[0]
    np.testing.assert_allclose(torch_output2.numpy(), ort_output2, rtol=1e-4, atol=1e-5)
    print(f"   Dynamic width test (W={test_imgW2}) also passed!")


verify_detector()
print()
verify_recognizer()

# %% [markdown]
# ## 5. Print model sizes

# %%


def print_model_size(path, name):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"{name}: {size_mb:.1f} MB")


print()
print("--- Model Sizes ---")
print_model_size(os.path.join("..", "models", "craft_mlt_25k.pth"), "CRAFT PyTorch")
print_model_size(detector_onnx_path, "CRAFT ONNX")
print_model_size(os.path.join("..", "models", "english_g2.pth"), "English G2 PyTorch")
print_model_size(recognizer_onnx_path, "English G2 ONNX")
