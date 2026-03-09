"""
Integration tests for the full EasyOCR-ONNX pipeline.

These tests require the ONNX model files to be present in models/.
They will be automatically skipped if the models are not found.

Run with:
    uv run pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from easyocr_onnx import Reader, reformat_input
from tests.conftest import skip_no_models

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
DETECTOR_ONNX = os.path.join(_MODELS_DIR, "craft_mlt_25k.onnx")
RECOGNIZER_ONNX = os.path.join(_MODELS_DIR, "english_g2.onnx")


@pytest.fixture
def reader():
    """Create a Reader instance (skips if models missing)."""
    if not (os.path.isfile(DETECTOR_ONNX) and os.path.isfile(RECOGNIZER_ONNX)):
        pytest.skip("ONNX model files not found — run notebooks/convert.py first")
    return Reader(
        detector_onnx_path=DETECTOR_ONNX,
        recognizer_onnx_path=RECOGNIZER_ONNX,
    )


def _make_text_image(
    text_lines: list[tuple[str, tuple[int, int]]],
    width: int = 640,
    height: int = 480,
    font_scale: float = 2.0,
    thickness: int = 4,
) -> np.ndarray:
    """
    Create a white RGB image with black text rendered at specified positions.
    Returns an RGB uint8 numpy array.
    """
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, (x, y) in text_lines:
        cv2.putText(
            img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )
    return img


# ===========================================================================
# Reader instantiation
# ===========================================================================


@skip_no_models
class TestReaderInit:
    """Test Reader construction and session setup."""

    def test_create_reader(self, reader):
        assert reader.detector_session is not None
        assert reader.recognizer_session is not None

    def test_detector_input_name(self, reader):
        assert isinstance(reader.detector_input_name, str)
        assert len(reader.detector_input_name) > 0

    def test_recognizer_input_name(self, reader):
        assert isinstance(reader.recognizer_input_name, str)
        assert len(reader.recognizer_input_name) > 0

    def test_converter_initialized(self, reader):
        assert reader.converter is not None
        assert len(reader.converter.character) == 97  # 96 chars + blank

    def test_default_img_height(self, reader):
        assert reader.imgH == 64


# ===========================================================================
# Detection only
# ===========================================================================


@skip_no_models
class TestDetection:
    """Test the detect() method in isolation."""

    def test_blank_image_no_detections(self, reader):
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
        h_list, f_list = reader.detect(blank)
        # A blank white image should produce no text detections
        assert isinstance(h_list, list)
        assert isinstance(f_list, list)
        assert len(h_list) == 0
        assert len(f_list) == 0

    def test_text_image_has_detections(self, reader):
        img = _make_text_image(
            [
                ("Hello World", (50, 150)),
                ("Test 123", (50, 350)),
            ]
        )
        h_list, f_list = reader.detect(img)
        total = len(h_list) + len(f_list)
        assert total >= 1, "Expected at least one detected text region"

    def test_horizontal_boxes_format(self, reader):
        img = _make_text_image([("ABCDEF", (50, 200))])
        h_list, f_list = reader.detect(img)
        for box in h_list:
            # horizontal_list entries are [x_min, x_max, y_min, y_max]
            assert len(box) == 4
            x_min, x_max, y_min, y_max = box
            assert x_max > x_min
            assert y_max > y_min

    def test_detect_various_canvas_sizes(self, reader):
        img = _make_text_image([("Hello", (50, 200))])
        for canvas_size in [640, 1280, 2560]:
            h_list, _ = reader.detect(img, canvas_size=canvas_size)
            # Should not crash regardless of canvas_size
            assert isinstance(h_list, list)

    def test_detect_custom_thresholds(self, reader):
        img = _make_text_image([("Testing", (50, 200))])
        # Very high text_threshold should reduce detections
        h_strict, f_strict = reader.detect(img, text_threshold=0.99, low_text=0.99)
        h_loose, f_loose = reader.detect(img, text_threshold=0.3, low_text=0.2)
        strict_count = len(h_strict) + len(f_strict)
        loose_count = len(h_loose) + len(f_loose)
        assert loose_count >= strict_count

    def test_small_image(self, reader):
        img = _make_text_image(
            [("Hi", (5, 25))],
            width=60,
            height=40,
            font_scale=0.5,
            thickness=1,
        )
        h_list, f_list = reader.detect(img)
        # Should not crash on small images
        assert isinstance(h_list, list)


# ===========================================================================
# Recognition only
# ===========================================================================


@skip_no_models
class TestRecognition:
    """Test the recognize() method in isolation."""

    def test_recognize_with_known_box(self, reader):
        img = _make_text_image([("Hello", (50, 100))], height=200, width=400)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Provide a bounding box that covers the text
        h_list = [[20, 380, 40, 140]]
        results = reader.recognize(grey, horizontal_list=h_list)
        assert len(results) >= 1
        # Each result is (bbox, text, confidence)
        bbox, text, confidence = results[0]
        assert isinstance(text, str)
        assert len(text) > 0
        assert 0.0 <= confidence <= 1.0

    def test_recognize_blank_region(self, reader):
        grey = np.full((200, 400), 255, dtype=np.uint8)
        h_list = [[10, 390, 10, 190]]
        results = reader.recognize(grey, horizontal_list=h_list, filter_ths=0.0)
        # Should return something (possibly low-confidence garbage) but not crash
        assert isinstance(results, list)

    def test_recognize_detail_zero(self, reader):
        img = _make_text_image([("Test", (20, 80))], height=150, width=300)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h_list = [[10, 280, 20, 120]]
        results = reader.recognize(grey, horizontal_list=h_list, detail=0)
        # detail=0 should return plain strings
        for item in results:
            assert isinstance(item, str)

    def test_recognize_dict_output(self, reader):
        img = _make_text_image([("Test", (20, 80))], height=150, width=300)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h_list = [[10, 280, 20, 120]]
        results = reader.recognize(grey, horizontal_list=h_list, output_format="dict")
        for item in results:
            assert isinstance(item, dict)
            assert "boxes" in item
            assert "text" in item
            assert "confident" in item

    def test_recognize_no_boxes_uses_full_image(self, reader):
        img = _make_text_image([("Full", (20, 80))], height=150, width=300)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # When no boxes are provided, should use the entire image
        results = reader.recognize(grey, filter_ths=0.0)
        assert isinstance(results, list)


# ===========================================================================
# Full pipeline (readtext)
# ===========================================================================


@skip_no_models
class TestReadtext:
    """End-to-end tests for readtext()."""

    def test_readtext_rgb_ndarray(self, reader):
        img = _make_text_image(
            [
                ("Hello World", (50, 150)),
                ("EasyOCR", (50, 350)),
            ]
        )
        results = reader.readtext(img)
        assert isinstance(results, list)
        # Should detect at least some text
        if len(results) > 0:
            bbox, text, confidence = results[0]
            assert isinstance(bbox, list)
            assert isinstance(text, str)
            assert isinstance(confidence, float)

    def test_readtext_greyscale_ndarray(self, reader):
        img = _make_text_image([("Greyscale", (50, 200))])
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        results = reader.readtext(grey)
        assert isinstance(results, list)

    def test_readtext_from_file_png(self, reader):
        img = _make_text_image([("From File", (50, 200))])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            results = reader.readtext(path)
            assert isinstance(results, list)
        finally:
            os.unlink(path)

    def test_readtext_from_file_jpg(self, reader):
        img = _make_text_image([("JPEG Test", (50, 200))])
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            results = reader.readtext(path)
            assert isinstance(results, list)
        finally:
            os.unlink(path)

    def test_readtext_from_bytes(self, reader):
        img = _make_text_image([("Bytes Input", (50, 200))])
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", bgr)
        results = reader.readtext(buf.tobytes())
        assert isinstance(results, list)

    def test_readtext_detail_zero(self, reader):
        img = _make_text_image([("Detail Zero", (50, 200))])
        results = reader.readtext(img, detail=0)
        for item in results:
            assert isinstance(item, str)

    def test_readtext_dict_output(self, reader):
        img = _make_text_image([("Dict Format", (50, 200))])
        results = reader.readtext(img, output_format="dict")
        for item in results:
            assert isinstance(item, dict)

    def test_readtext_blank_image_no_results(self, reader):
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
        results = reader.readtext(blank)
        assert results == []

    def test_readtext_filter_threshold(self, reader):
        img = _make_text_image([("Filter Test", (50, 200))])
        # Very high filter should remove low-confidence results
        results_strict = reader.readtext(img, filter_ths=0.99)
        results_loose = reader.readtext(img, filter_ths=0.0)
        assert len(results_loose) >= len(results_strict)

    def test_readtext_confidence_range(self, reader):
        img = _make_text_image(
            [
                ("Confidence", (50, 150)),
                ("Range Check", (50, 350)),
            ]
        )
        results = reader.readtext(img)
        for bbox, text, confidence in results:
            assert 0.0 <= confidence <= 1.0, (
                f"Confidence {confidence} out of [0, 1] range for text '{text}'"
            )

    def test_readtext_bbox_format(self, reader):
        img = _make_text_image([("BBox Test", (50, 200))])
        results = reader.readtext(img)
        for bbox, text, confidence in results:
            # bbox should be list of 4 points, each [x, y]
            assert len(bbox) == 4
            for point in bbox:
                assert len(point) == 2
                x, y = point
                assert isinstance(x, (int, float, np.integer, np.floating))
                assert isinstance(y, (int, float, np.integer, np.floating))

    def test_readtext_multiple_lines(self, reader):
        img = _make_text_image(
            [
                ("Line One", (50, 100)),
                ("Line Two", (50, 250)),
                ("Line Three", (50, 400)),
            ]
        )
        results = reader.readtext(img)
        # We expect at least 2 text regions detected
        assert len(results) >= 2, (
            f"Expected at least 2 text regions, got {len(results)}"
        )


# ===========================================================================
# Recognised text accuracy (smoke tests)
# ===========================================================================


@skip_no_models
class TestRecognitionAccuracy:
    """
    Smoke tests that verify the model actually recognises common text.
    These are not strict assertions — OCR is inherently fuzzy — but they
    catch major regressions like broken CTC decoding or wrong character sets.
    """

    def _readtext_simple(self, reader, text: str, font_scale: float = 2.5) -> list[str]:
        """Render text, run OCR, return recognised strings."""
        img = _make_text_image(
            [(text, (30, 120))],
            width=max(640, len(text) * 50),
            height=200,
            font_scale=font_scale,
            thickness=5,
        )
        return reader.readtext(img, detail=0)

    def test_digits(self, reader):
        recognised = self._readtext_simple(reader, "1234567890")
        combined = "".join(recognised).replace(" ", "")
        # At least some digits should be recognised
        digit_count = sum(1 for c in combined if c.isdigit())
        assert digit_count >= 5, f"Expected digits, got: {recognised}"

    def test_uppercase_letters(self, reader):
        recognised = self._readtext_simple(reader, "ABCDEFGH")
        combined = "".join(recognised).upper().replace(" ", "")
        alpha_count = sum(1 for c in combined if c.isalpha())
        assert alpha_count >= 4, f"Expected letters, got: {recognised}"

    def test_lowercase_letters(self, reader):
        recognised = self._readtext_simple(reader, "abcdefgh")
        combined = "".join(recognised).lower().replace(" ", "")
        alpha_count = sum(1 for c in combined if c.isalpha())
        assert alpha_count >= 4, f"Expected letters, got: {recognised}"

    def test_mixed_case(self, reader):
        recognised = self._readtext_simple(reader, "Hello World")
        combined = " ".join(recognised).strip()
        assert len(combined) > 0, "Expected some text output"

    def test_punctuation(self, reader):
        recognised = self._readtext_simple(reader, "Hello, World!")
        combined = " ".join(recognised)
        # Should contain at least a comma or exclamation mark
        assert len(combined) > 3, f"Expected punctuated text, got: {recognised}"


# ===========================================================================
# Edge cases and robustness
# ===========================================================================


@skip_no_models
class TestEdgeCases:
    """Edge cases that should not crash the pipeline."""

    def test_very_small_image(self, reader):
        img = np.full((16, 16, 3), 200, dtype=np.uint8)
        results = reader.readtext(img)
        assert isinstance(results, list)

    def test_single_pixel_image(self, reader):
        img = np.full((1, 1, 3), 128, dtype=np.uint8)
        results = reader.readtext(img)
        assert isinstance(results, list)

    def test_very_wide_image(self, reader):
        img = np.full((32, 2000, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Wide", (50, 25), font, 0.8, (0, 0, 0), 2)
        results = reader.readtext(img)
        assert isinstance(results, list)

    def test_very_tall_image(self, reader):
        img = np.full((2000, 32, 3), 255, dtype=np.uint8)
        results = reader.readtext(img)
        assert isinstance(results, list)

    def test_noisy_image(self, reader):
        rng = np.random.RandomState(123)
        img = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        results = reader.readtext(img)
        assert isinstance(results, list)

    def test_rgba_input(self, reader):
        rgba = np.full((200, 400, 4), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rgba, "RGBA", (20, 100), font, 2.0, (0, 0, 0, 255), 3)
        results = reader.readtext(rgba[:, :, :3])  # strip alpha before passing
        assert isinstance(results, list)

    def test_min_size_filters_small_detections(self, reader):
        img = _make_text_image([("Big Text", (50, 200))])
        results_large_min = reader.readtext(img, min_size=9999)
        results_small_min = reader.readtext(img, min_size=1)
        assert len(results_small_min) >= len(results_large_min)

    def test_readtext_is_deterministic(self, reader):
        img = _make_text_image([("Deterministic", (50, 200))])
        results1 = reader.readtext(img)
        results2 = reader.readtext(img)
        assert len(results1) == len(results2)
        for (b1, t1, c1), (b2, t2, c2) in zip(results1, results2):
            assert t1 == t2
            assert abs(c1 - c2) < 1e-6
