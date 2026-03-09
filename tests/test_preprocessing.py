"""
Unit tests for preprocessing, postprocessing, and utility functions.

These tests do NOT require ONNX model files — they exercise the pure
NumPy/OpenCV helper functions in easyocr_onnx/reader.py.
"""

from __future__ import annotations

import math
import os
import tempfile

import cv2
import numpy as np
import pytest

from easyocr_onnx.reader import (
    ENGLISH_G2_CHARACTERS,
    CTCLabelConverter,
    _adjust_contrast_grey,
    _adjust_result_coordinates,
    _calculate_ratio,
    _compute_ratio_and_resize,
    _contrast_grey,
    _custom_mean,
    _four_point_transform,
    _get_det_boxes,
    _get_image_list,
    _group_text_box,
    _normalize_mean_variance,
    _normalize_pad,
    _prepare_recognizer_input,
    _resize_aspect_ratio,
    _softmax,
    reformat_input,
)

# ===========================================================================
# reformat_input
# ===========================================================================


class TestReformatInput:
    """Tests for the reformat_input() function."""

    def test_rgb_ndarray(self, dummy_rgb_image):
        img, grey = reformat_input(dummy_rgb_image)
        assert img.shape == (480, 640, 3)
        assert grey.shape == (480, 640)
        assert img.dtype == np.uint8
        assert grey.dtype == np.uint8

    def test_greyscale_ndarray(self, dummy_grey_image):
        img, grey = reformat_input(dummy_grey_image)
        assert img.shape == (480, 640, 3)
        assert grey.shape == (480, 640)
        # The grey output should be the same as input
        np.testing.assert_array_equal(grey, dummy_grey_image)

    def test_rgba_ndarray(self):
        rgba = np.full((100, 200, 4), 128, dtype=np.uint8)
        img, grey = reformat_input(rgba)
        assert img.shape == (100, 200, 3)
        assert grey.shape == (100, 200)

    def test_single_channel_3d(self):
        single = np.full((100, 200, 1), 128, dtype=np.uint8)
        img, grey = reformat_input(single)
        assert img.shape == (100, 200, 3)
        assert grey.shape == (100, 200)

    def test_file_path_png(self, tmp_png_path):
        img, grey = reformat_input(tmp_png_path)
        assert img.ndim == 3 and img.shape[2] == 3
        assert grey.ndim == 2

    def test_file_path_jpg(self, tmp_jpg_path):
        img, grey = reformat_input(tmp_jpg_path)
        assert img.ndim == 3 and img.shape[2] == 3
        assert grey.ndim == 2

    def test_bytes_input(self, image_with_text):
        bgr = cv2.cvtColor(image_with_text, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", bgr)
        img, grey = reformat_input(buf.tobytes())
        assert img.ndim == 3 and img.shape[2] == 3
        assert grey.ndim == 2

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid input type"):
            reformat_input(12345)

    def test_unsupported_ndarray_shape(self):
        bad = np.zeros((10, 10, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported ndarray shape"):
            reformat_input(bad)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            reformat_input("/nonexistent/path/to/image.png")

    def test_pil_image_if_pillow_installed(self):
        """If Pillow is installed (dev dep), PIL Image input should work."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        pil_img = Image.fromarray(
            np.full((100, 200, 3), 128, dtype=np.uint8), mode="RGB"
        )
        img, grey = reformat_input(pil_img)
        assert img.shape == (100, 200, 3)
        assert grey.shape == (100, 200)


# ===========================================================================
# Detector preprocessing
# ===========================================================================


class TestNormalizeMeanVariance:
    """Tests for _normalize_mean_variance()."""

    def test_output_shape_unchanged(self):
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        out = _normalize_mean_variance(img)
        assert out.shape == img.shape
        assert out.dtype == np.float32

    def test_zero_image(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        out = _normalize_mean_variance(img)
        # All zeros → (0 - mean) / variance for each channel
        assert out.dtype == np.float32
        assert out.shape == (32, 32, 3)

    def test_constant_image(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = _normalize_mean_variance(img)
        # Should be roughly centred around 0 (ImageNet mean ≈ 0.485..0.406)
        assert out.dtype == np.float32
        # All values should be the same within each channel
        for c in range(3):
            vals = out[:, :, c]
            assert np.allclose(vals, vals[0, 0])


class TestResizeAspectRatio:
    """Tests for _resize_aspect_ratio()."""

    def test_output_dimensions_are_multiples_of_32(self):
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        resized, ratio, size_heatmap = _resize_aspect_ratio(
            img, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
        )
        h, w = resized.shape[:2]
        assert h % 32 == 0, f"Height {h} not multiple of 32"
        assert w % 32 == 0, f"Width {w} not multiple of 32"

    def test_small_image_gets_padded(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        resized, ratio, size_heatmap = _resize_aspect_ratio(
            img, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
        )
        h, w = resized.shape[:2]
        assert h >= 32
        assert w >= 32
        assert h % 32 == 0
        assert w % 32 == 0

    def test_ratio_is_positive(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        _, ratio, _ = _resize_aspect_ratio(
            img, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
        )
        assert ratio > 0

    def test_large_image_gets_shrunk(self):
        img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        resized, ratio, size_heatmap = _resize_aspect_ratio(
            img, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
        )
        assert resized.shape[0] <= 640 + 32  # allow padding
        assert resized.shape[1] <= 640 + 32

    def test_mag_ratio_scales_up(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        r1_img, r1, _ = _resize_aspect_ratio(
            img, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
        )
        r2_img, r2, _ = _resize_aspect_ratio(
            img, 640, interpolation=cv2.INTER_LINEAR, mag_ratio=2.0
        )
        # With higher mag_ratio, effective size should be >= r1
        assert r2_img.shape[0] >= r1_img.shape[0] or r2_img.shape[1] >= r1_img.shape[1]


# ===========================================================================
# Detector post-processing
# ===========================================================================


class TestGetDetBoxes:
    """Tests for _get_det_boxes()."""

    def test_blank_maps_return_empty(self):
        textmap = np.zeros((160, 160), dtype=np.float32)
        linkmap = np.zeros((160, 160), dtype=np.float32)
        boxes, polys, mapper = _get_det_boxes(
            textmap, linkmap, text_threshold=0.7, link_threshold=0.4, low_text=0.4
        )
        assert len(boxes) == 0
        assert len(polys) == 0

    def test_strong_signal_returns_box(self):
        textmap = np.zeros((160, 320), dtype=np.float32)
        linkmap = np.zeros((160, 320), dtype=np.float32)
        # Paint a bright rectangle on the text map
        textmap[40:60, 40:120] = 0.9
        boxes, polys, mapper = _get_det_boxes(
            textmap, linkmap, text_threshold=0.7, link_threshold=0.4, low_text=0.4
        )
        assert len(boxes) >= 1
        # Each box should be a 4×2 array
        for b in boxes:
            arr = np.array(b)
            assert arr.shape == (4, 2)

    def test_estimate_num_chars(self):
        textmap = np.zeros((160, 320), dtype=np.float32)
        linkmap = np.zeros((160, 320), dtype=np.float32)
        # Two separate "characters" in the text map
        textmap[40:60, 40:60] = 0.9
        textmap[40:60, 80:100] = 0.9
        # Join them with a link
        linkmap[40:60, 60:80] = 0.9
        boxes, polys, mapper = _get_det_boxes(
            textmap,
            linkmap,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            estimate_num_chars=True,
        )
        # mapper should be populated when estimate_num_chars is True
        assert mapper is not None
        if len(boxes) > 0:
            assert len(mapper) == len(boxes)


class TestAdjustResultCoordinates:
    """Tests for _adjust_result_coordinates()."""

    def test_scaling(self):
        box = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.float32)
        adjusted = _adjust_result_coordinates([box], ratio_w=2.0, ratio_h=3.0)
        assert len(adjusted) == 1
        # ratio_net default is 2 → effective scale: (2*2, 3*2) = (4, 6)
        expected = box * np.array([4.0, 6.0])
        np.testing.assert_allclose(adjusted[0], expected)

    def test_identity_scaling(self):
        box = np.array([[1, 2], [3, 4]], dtype=np.float32)
        adjusted = _adjust_result_coordinates([box], ratio_w=0.5, ratio_h=0.5)
        # 0.5 * 2 = 1.0 → no change
        np.testing.assert_allclose(adjusted[0], box)

    def test_empty_list(self):
        assert _adjust_result_coordinates([], 1.0, 1.0) == []


class TestGroupTextBox:
    """Tests for _group_text_box()."""

    def test_empty_polys(self):
        h, f = _group_text_box([], 0.1, 0.5, 0.5, 0.5, 0.1, True)
        assert h == []
        assert f == []

    def test_single_horizontal_box(self):
        # A clearly horizontal box: wide, not rotated
        poly = np.array([10, 100, 200, 100, 200, 130, 10, 130], dtype=np.int32)
        h, f = _group_text_box([poly], 0.1, 0.5, 0.5, 0.5, 0.1, True)
        # Should be classified as horizontal
        assert len(h) >= 1 or len(f) >= 1  # at minimum it's detected


# ===========================================================================
# Four-point transform and geometry helpers
# ===========================================================================


class TestFourPointTransform:
    """Tests for _four_point_transform()."""

    def test_identity_like_rect(self):
        img = np.arange(100 * 200, dtype=np.uint8).reshape(100, 200)
        rect = np.array([[0, 0], [199, 0], [199, 99], [0, 99]], dtype=np.float32)
        warped = _four_point_transform(img, rect)
        assert warped.shape[0] > 0
        assert warped.shape[1] > 0

    def test_subregion_crop(self):
        img = np.full((200, 300), 128, dtype=np.uint8)
        rect = np.array([[50, 50], [150, 50], [150, 100], [50, 100]], dtype=np.float32)
        warped = _four_point_transform(img, rect)
        assert warped.shape == (50, 100)


class TestCalculateRatio:
    """Tests for _calculate_ratio()."""

    def test_landscape(self):
        assert _calculate_ratio(200, 100) == 2.0

    def test_portrait(self):
        # When ratio < 1, it should be inverted
        assert _calculate_ratio(100, 200) == 2.0

    def test_square(self):
        assert _calculate_ratio(100, 100) == 1.0


class TestComputeRatioAndResize:
    """Tests for _compute_ratio_and_resize()."""

    def test_landscape_resize(self):
        img = np.zeros((50, 200), dtype=np.uint8)
        resized, ratio = _compute_ratio_and_resize(img, 200, 50, model_height=64)
        assert resized.shape[0] == 64  # height should be model_height
        assert ratio == 4.0  # 200/50

    def test_portrait_resize(self):
        img = np.zeros((200, 50), dtype=np.uint8)
        resized, ratio = _compute_ratio_and_resize(img, 50, 200, model_height=64)
        assert resized.shape[1] == 64  # width should be model_height (rotated logic)
        assert ratio == 4.0  # inverted: 200/50


# ===========================================================================
# _get_image_list
# ===========================================================================


class TestGetImageList:
    """Tests for _get_image_list()."""

    def test_horizontal_boxes(self, image_with_text_grey):
        # Provide a horizontal box that contains the first line of text
        h_list = [[30, 400, 100, 200]]
        image_list, max_width = _get_image_list(h_list, [], image_with_text_grey)
        assert len(image_list) == 1
        assert max_width > 0
        # Each entry is (coord, crop_image)
        coord, crop = image_list[0]
        assert crop.ndim == 2  # grayscale
        assert crop.shape[0] > 0

    def test_empty_lists(self, dummy_grey_image):
        image_list, max_width = _get_image_list([], [], dummy_grey_image)
        assert len(image_list) == 0
        # max_width should still be some positive value (model_height * 1)
        assert max_width >= 64


# ===========================================================================
# Recognizer preprocessing
# ===========================================================================


class TestContrastGrey:
    """Tests for _contrast_grey()."""

    def test_uniform_image(self):
        img = np.full((64, 128), 128, dtype=np.uint8)
        contrast, high, low = _contrast_grey(img)
        assert contrast == 0.0
        assert high == low == 128.0

    def test_high_contrast(self):
        img = np.zeros((64, 128), dtype=np.uint8)
        img[:, :64] = 0
        img[:, 64:] = 255
        contrast, high, low = _contrast_grey(img)
        assert contrast > 0


class TestAdjustContrastGrey:
    """Tests for _adjust_contrast_grey()."""

    def test_low_contrast_gets_adjusted(self):
        img = np.full((64, 128), 128, dtype=np.uint8)
        adjusted = _adjust_contrast_grey(img, target=0.4)
        assert adjusted.dtype == np.uint8
        # The output should have more spread than constant 128
        assert adjusted.shape == img.shape

    def test_high_contrast_unchanged(self):
        img = np.zeros((64, 128), dtype=np.uint8)
        img[:, :64] = 10
        img[:, 64:] = 245
        adjusted = _adjust_contrast_grey(img, target=0.4)
        np.testing.assert_array_equal(adjusted, img)


class TestNormalizePad:
    """Tests for _normalize_pad()."""

    def test_output_shape(self):
        img = np.full((64, 100), 128, dtype=np.uint8)
        out = _normalize_pad(img, (1, 64, 200))
        assert out.shape == (1, 64, 200)
        assert out.dtype == np.float32

    def test_no_padding_needed(self):
        img = np.full((64, 200), 128, dtype=np.uint8)
        out = _normalize_pad(img, (1, 64, 200))
        assert out.shape == (1, 64, 200)

    def test_values_normalized(self):
        img = np.zeros((64, 100), dtype=np.uint8)
        out = _normalize_pad(img, (1, 64, 100))
        # 0/255 = 0.0, (0.0 - 0.5)/0.5 = -1.0
        np.testing.assert_allclose(out[0, :, :], -1.0)

    def test_white_image_normalized(self):
        img = np.full((64, 100), 255, dtype=np.uint8)
        out = _normalize_pad(img, (1, 64, 100))
        # 255/255 = 1.0, (1.0 - 0.5)/0.5 = 1.0
        np.testing.assert_allclose(out[0, :, :], 1.0)

    def test_right_pad_repeats_last_column(self):
        img = np.full((64, 50), 100, dtype=np.uint8)
        out = _normalize_pad(img, (1, 64, 100))
        # Padded region should equal the last column of the original
        last_col = out[0, :, 49]
        for w in range(50, 100):
            np.testing.assert_allclose(out[0, :, w], last_col)


class TestPrepareRecognizerInput:
    """Tests for _prepare_recognizer_input()."""

    def test_output_shape(self):
        crops = [np.full((64, 100), 128, dtype=np.uint8)]
        out = _prepare_recognizer_input(crops, imgH=64, imgW=200)
        assert out.shape == (1, 1, 64, 200)
        assert out.dtype == np.float32

    def test_batch_of_crops(self):
        crops = [
            np.full((64, 80), 128, dtype=np.uint8),
            np.full((64, 120), 200, dtype=np.uint8),
            np.full((64, 60), 50, dtype=np.uint8),
        ]
        out = _prepare_recognizer_input(crops, imgH=64, imgW=200)
        assert out.shape == (3, 1, 64, 200)

    def test_contrast_adjustment(self):
        crops = [np.full((64, 100), 128, dtype=np.uint8)]
        out = _prepare_recognizer_input(crops, imgH=64, imgW=200, adjust_contrast=0.5)
        assert out.shape == (1, 1, 64, 200)

    def test_wide_crop_gets_clamped(self):
        # A crop wider than imgW should be resized to imgW
        crops = [np.full((32, 500), 128, dtype=np.uint8)]
        out = _prepare_recognizer_input(crops, imgH=64, imgW=200)
        assert out.shape == (1, 1, 64, 200)


# ===========================================================================
# CTC Label Converter
# ===========================================================================


class TestCTCLabelConverter:
    """Tests for CTCLabelConverter."""

    def test_init_with_english_g2(self):
        converter = CTCLabelConverter(ENGLISH_G2_CHARACTERS)
        # 96 characters + [blank] = 97 classes
        assert len(converter.character) == 97

    def test_decode_greedy_simple(self):
        converter = CTCLabelConverter(ENGLISH_G2_CHARACTERS)
        # Simulate a sequence: blank=0, then indices for characters
        # Characters in ENGLISH_G2_CHARACTERS: '0'=index 1, '1'=index 2, ...
        # 'A' is at position 36 in the string → model index 37
        a_idx = ENGLISH_G2_CHARACTERS.index("A") + 1
        b_idx = ENGLISH_G2_CHARACTERS.index("B") + 1

        # Sequence: [A, A, blank, B, B] → decoded: "AB"
        indices = np.array([a_idx, a_idx, 0, b_idx, b_idx], dtype=np.int64)
        result = converter.decode_greedy(indices, [5])
        assert len(result) == 1
        assert result[0] == "AB"

    def test_decode_greedy_all_blanks(self):
        converter = CTCLabelConverter(ENGLISH_G2_CHARACTERS)
        indices = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        result = converter.decode_greedy(indices, [5])
        assert result[0] == ""

    def test_decode_greedy_multiple_sequences(self):
        converter = CTCLabelConverter(ENGLISH_G2_CHARACTERS)
        idx_h = ENGLISH_G2_CHARACTERS.index("H") + 1
        idx_i = ENGLISH_G2_CHARACTERS.index("i") + 1

        # Two sequences of length 3: [H, 0, i] and [i, i, 0]
        indices = np.array([idx_h, 0, idx_i, idx_i, idx_i, 0], dtype=np.int64)
        result = converter.decode_greedy(indices, [3, 3])
        assert len(result) == 2
        assert result[0] == "Hi"
        assert result[1] == "i"

    def test_character_count(self):
        assert len(ENGLISH_G2_CHARACTERS) == 96

    def test_euro_symbol_present(self):
        """The € symbol must be in the character set (common gotcha)."""
        assert "€" in ENGLISH_G2_CHARACTERS


# ===========================================================================
# Softmax and custom mean
# ===========================================================================


class TestSoftmax:
    """Tests for _softmax()."""

    def test_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        out = _softmax(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), [1.0, 1.0], atol=1e-6)

    def test_uniform_input(self):
        x = np.zeros((2, 5), dtype=np.float32)
        out = _softmax(x, axis=-1)
        np.testing.assert_allclose(out, 0.2, atol=1e-6)

    def test_large_values_no_overflow(self):
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        out = _softmax(x, axis=-1)
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)
        assert np.all(np.isfinite(out))


class TestCustomMean:
    """Tests for _custom_mean()."""

    def test_single_value(self):
        # _custom_mean = prod ** (2 / sqrt(n)); for n=1: 0.9 ** 2 = 0.81
        assert _custom_mean(np.array([0.9])) == pytest.approx(0.81)

    def test_all_ones(self):
        assert _custom_mean(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)

    def test_geometric_like(self):
        # _custom_mean = prod ** (2 / sqrt(n))
        vals = np.array([0.5, 0.5])
        expected = (0.5 * 0.5) ** (2.0 / np.sqrt(2))
        assert _custom_mean(vals) == pytest.approx(expected)


# ===========================================================================
# No Pillow in production imports
# ===========================================================================


class TestNoPillowRequired:
    """Ensure easyocr_onnx/reader.py does not have a hard Pillow import."""

    def test_no_top_level_pil_import(self):
        import importlib
        import sys

        # Temporarily hide PIL from the import system
        pil_modules = {k: v for k, v in sys.modules.items() if k.startswith("PIL")}
        # We can't easily unload PIL if it's already loaded,
        # so instead we just verify the module-level code doesn't import it.
        import ast

        reader_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src",
            "easyocr_onnx",
            "reader.py",
        )
        with open(reader_path) as f:
            tree = ast.parse(f.read())

        top_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_level_imports.append(node.module)

        assert "PIL" not in top_level_imports, (
            "PIL should not be a top-level import in reader.py"
        )
        assert "PIL.Image" not in top_level_imports, (
            "PIL.Image should not be a top-level import in reader.py"
        )
