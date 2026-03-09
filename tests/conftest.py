"""
Shared pytest fixtures for EasyOCR-ONNX tests.

Provides:
- Programmatically generated dummy images (RGB, grayscale, with rendered text)
- Temporary image files on disk (PNG, JPEG)
- ONNX model path fixtures with automatic skip if models are missing
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

DETECTOR_ONNX = os.path.join(_MODELS_DIR, "craft_mlt_25k.onnx")
RECOGNIZER_ONNX = os.path.join(_MODELS_DIR, "english_g2.onnx")

models_available = os.path.isfile(DETECTOR_ONNX) and os.path.isfile(RECOGNIZER_ONNX)

skip_no_models = pytest.mark.skipif(
    not models_available,
    reason="ONNX model files not found in models/ — run notebooks/convert.py first",
)


# ---------------------------------------------------------------------------
# Model path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector_onnx_path() -> str:
    """Return path to the CRAFT detector ONNX model (skips if missing)."""
    if not os.path.isfile(DETECTOR_ONNX):
        pytest.skip(f"Detector model not found: {DETECTOR_ONNX}")
    return DETECTOR_ONNX


@pytest.fixture
def recognizer_onnx_path() -> str:
    """Return path to the English G2 recognizer ONNX model (skips if missing)."""
    if not os.path.isfile(RECOGNIZER_ONNX):
        pytest.skip(f"Recognizer model not found: {RECOGNIZER_ONNX}")
    return RECOGNIZER_ONNX


# ---------------------------------------------------------------------------
# Dummy image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_rgb_image() -> np.ndarray:
    """A 640×480 RGB uint8 image with random noise."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_grey_image() -> np.ndarray:
    """A 640×480 single-channel uint8 image with random noise."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640), dtype=np.uint8)


@pytest.fixture
def white_image_rgb() -> np.ndarray:
    """A 640×480 solid white RGB image (useful as a blank canvas)."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def white_image_grey() -> np.ndarray:
    """A 640×480 solid white grayscale image."""
    return np.full((480, 640), 255, dtype=np.uint8)


@pytest.fixture
def image_with_text() -> np.ndarray:
    """
    A 640×480 RGB image with clearly rendered text for integration tests.
    White background, black text: "Hello World" and "Test 123".
    """
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    # Use a thick font so detection has clear signal
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Hello World", (50, 150), font, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, "Test 123", (50, 350), font, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    return img


@pytest.fixture
def image_with_text_grey() -> np.ndarray:
    """Grayscale version of image_with_text."""
    img = np.full((480, 640), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Hello World", (50, 150), font, 2.0, 0, 4, cv2.LINE_AA)
    cv2.putText(img, "Test 123", (50, 350), font, 2.0, 0, 4, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Temporary file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_png_path(image_with_text) -> str:
    """Write image_with_text to a temporary PNG file and yield its path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    # OpenCV expects BGR for imwrite
    bgr = cv2.cvtColor(image_with_text, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_jpg_path(image_with_text) -> str:
    """Write image_with_text to a temporary JPEG file and yield its path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
    bgr = cv2.cvtColor(image_with_text, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_grey_png_path(image_with_text_grey) -> str:
    """Write a grayscale image to a temporary PNG file and yield its path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    cv2.imwrite(path, image_with_text_grey)
    yield path
    os.unlink(path)
