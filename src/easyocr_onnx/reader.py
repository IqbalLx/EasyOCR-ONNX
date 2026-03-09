"""
EasyOCR-ONNX Reader

Drop-in replacement for easyocr.Reader that uses ONNX Runtime instead of PyTorch.
Only the inference layer is swapped — all pre/post-processing logic is ported from
the original EasyOCR source as pure NumPy/OpenCV operations.

Production dependencies: numpy, onnxruntime, opencv-python-headless (no Pillow).

Usage:
    from easyocr_onnx import Reader

    reader = Reader(
        detector_onnx_path="models/craft_mlt_25k.onnx",
        recognizer_onnx_path="models/english_g2.onnx",
    )
    results = reader.readtext("image.jpg")
    # results: list of (bbox, text, confidence)
"""

from __future__ import annotations

import math
import os
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort


def _try_import_pil():
    """Lazily import PIL — only needed if a PIL Image is passed as input."""
    try:
        from PIL import Image as _Image

        return _Image
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Image loading / reformatting (from easyocr/utils.py + easyocr/imgproc.py)
# ---------------------------------------------------------------------------


def _load_image(image_path: str) -> np.ndarray:
    """Load an image file as RGB numpy array."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def reformat_input(image) -> tuple[np.ndarray, np.ndarray]:
    """
    Accept various input types and return (img_rgb, img_grey).
    Supports: file path (str), bytes, numpy array.
    """
    if isinstance(image, str):
        image = os.path.expanduser(image)
        img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img_cv_grey is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        img = _load_image(image)
    elif isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_cv_grey = np.squeeze(image)
            img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            img = image[:, :, :3]
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported ndarray shape: {image.shape}")
    else:
        # Check for PIL Image without hard dependency
        PILImage = _try_import_pil()
        if PILImage is not None and isinstance(image, PILImage.Image):
            image_array = np.array(image.convert("RGB"))
            img = image_array
            img_cv_grey = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(
                "Invalid input type. Supported: str (file path), bytes, numpy array"
                " (install Pillow to also accept PIL Images)"
            )
    return img, img_cv_grey


# ---------------------------------------------------------------------------
# Detector preprocessing (from easyocr/imgproc.py)
# ---------------------------------------------------------------------------


def _normalize_mean_variance(
    in_img: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    variance: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    img = in_img.copy().astype(np.float32)
    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def _resize_aspect_ratio(
    img: np.ndarray,
    square_size: int,
    interpolation: int,
    mag_ratio: float = 1.0,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width, channel = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # Pad to multiples of 32
    target_h32 = target_h if target_h % 32 == 0 else target_h + (32 - target_h % 32)
    target_w32 = target_w if target_w % 32 == 0 else target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32
    size_heatmap = (int(target_w / 2), int(target_h / 2))
    return resized, ratio, size_heatmap


# ---------------------------------------------------------------------------
# Detector post-processing (from easyocr/craft_utils.py)
# ---------------------------------------------------------------------------


def _get_det_boxes(
    textmap: np.ndarray,
    linkmap: np.ndarray,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    poly: bool = False,
    estimate_num_chars: bool = False,
) -> tuple[list, list, list | None]:
    """
    Binarise score maps and find connected components → bounding boxes.
    Simplified from easyocr/craft_utils.py getDetBoxes.
    """
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    # Thresholding
    _, text_score = cv2.threshold(textmap, low_text, 1, cv2.THRESH_BINARY)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, cv2.THRESH_BINARY)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    boxes = []
    polys = []
    mapper = []

    for k in range(1, n_labels):
        # Size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # Thresholding
        seg_map = np.zeros(textmap.shape, dtype=np.uint8)
        seg_map[labels == k] = 255

        if estimate_num_chars:
            _, character_locs = cv2.threshold(
                textmap, text_threshold, 1, cv2.THRESH_BINARY
            )
            character_locs = character_locs.astype(np.uint8)
            character_locs = cv2.bitwise_and(
                character_locs, character_locs, mask=seg_map
            )
            n_chars, _, _, _ = cv2.connectedComponentsWithStats(
                character_locs, connectivity=4
            )
            n_chars = max(1, n_chars - 1)  # subtract background
            mapper.append(n_chars)
        else:
            mapper.append(k)

        seg_map = np.where(labels == k, 255, 0).astype(np.uint8)
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)

        sx, ex = max(0, x - niter), min(img_w, x + w + niter + 1)
        sy, ey = max(0, y - niter), min(img_h, y + h + niter + 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        seg_map_crop = seg_map[sy:ey, sx:ex]
        seg_map[sy:ey, sx:ex] = cv2.dilate(seg_map_crop, kernel)

        # Make box
        contours, _ = cv2.findContours(
            seg_map.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        rectangle = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rectangle)

        # Align diamond-shape
        w_rect, h_rect = (
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[1] - box[2]),
        )
        box_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(contour[:, 0, 0]), max(contour[:, 0, 0])
            t, b = min(contour[:, 0, 1]), max(contour[:, 0, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # Sort points: top-left, top-right, bottom-right, bottom-left
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        boxes.append(box)
        polys.append(box if not poly else box)  # simplified: use box for polys too

    return boxes, polys, mapper if estimate_num_chars else None


def _adjust_result_coordinates(
    polys: list, ratio_w: float, ratio_h: float, ratio_net: float = 2
) -> list:
    """Scale detected coordinates back to original image size."""
    adjusted = []
    for poly in polys:
        poly = np.array(poly)
        poly *= (ratio_w * ratio_net, ratio_h * ratio_net)
        adjusted.append(poly)
    return adjusted


# ---------------------------------------------------------------------------
# Box grouping (from easyocr/utils.py)
# ---------------------------------------------------------------------------


def _group_text_box(
    polys: list,
    slope_ths: float = 0.1,
    ycenter_ths: float = 0.5,
    height_ths: float = 0.5,
    width_ths: float = 0.5,
    add_margin: float = 0.1,
    sort_output: bool = True,
) -> tuple[list, list]:
    horizontal_list = []
    free_list = []

    for poly in polys:
        slope_up = (poly[3] - poly[1]) / max(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / max(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max(poly[0], poly[2], poly[4], poly[6])
            x_min = min(poly[0], poly[2], poly[4], poly[6])
            y_max = max(poly[1], poly[3], poly[5], poly[7])
            y_min = min(poly[1], poly[3], poly[5], poly[7])
            horizontal_list.append(
                [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min]
            )
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])
            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(np.arctan((poly[1] - poly[5]) / max(10, (poly[0] - poly[4]))))
            theta24 = abs(np.arctan((poly[3] - poly[7]) / max(10, (poly[2] - poly[6]))))
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin
            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # Combine and merge boxes
    combined_list: list[list] = []
    new_box: list = []
    b_height: list[float] = []
    b_ycenter: list[float] = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    merged_list = []
    for boxes in combined_list:
        if len(boxes) == 1:
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append(
                [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]
            )
        else:
            boxes = sorted(boxes, key=lambda item: item[0])
            merged_box: list[list] = []
            new_box2: list = []
            b_height2: list[float] = []
            x_max = 0.0
            for box in boxes:
                if len(new_box2) == 0:
                    b_height2 = [box[5]]
                    x_max = box[1]
                    new_box2.append(box)
                else:
                    if (
                        abs(np.mean(b_height2) - box[5])
                        < height_ths * np.mean(b_height2)
                    ) and ((box[0] - x_max) < width_ths * (box[3] - box[2])):
                        b_height2.append(box[5])
                        x_max = box[1]
                        new_box2.append(box)
                    else:
                        b_height2 = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box2)
                        new_box2 = [box]
            if len(new_box2) > 0:
                merged_box.append(new_box2)

            for mbox in merged_box:
                if len(mbox) != 1:
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max_ = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]
                    box_width = x_max_ - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * min(box_width, box_height))
                    merged_list.append(
                        [
                            x_min - margin,
                            x_max_ + margin,
                            y_min - margin,
                            y_max + margin,
                        ]
                    )
                else:
                    box = mbox[0]
                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * min(box_width, box_height))
                    merged_list.append(
                        [
                            box[0] - margin,
                            box[1] + margin,
                            box[2] - margin,
                            box[3] + margin,
                        ]
                    )

    return merged_list, free_list


# ---------------------------------------------------------------------------
# Crop extraction helpers (from easyocr/utils.py)
# ---------------------------------------------------------------------------


def _four_point_transform(image: np.ndarray, rect: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _calculate_ratio(width: int, height: int) -> float:
    ratio = width / height
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return ratio


def _compute_ratio_and_resize(
    img: np.ndarray, width: int, height: int, model_height: int
) -> tuple[np.ndarray, float]:
    ratio = width / height
    if ratio < 1.0:
        ratio = _calculate_ratio(width, height)
        img = cv2.resize(
            img,
            (model_height, int(model_height * ratio)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    else:
        img = cv2.resize(
            img,
            (int(model_height * ratio), model_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return img, ratio


def _get_image_list(
    horizontal_list: list,
    free_list: list,
    img: np.ndarray,
    model_height: int = 64,
    sort_output: bool = True,
) -> tuple[list, int]:
    image_list = []
    maximum_y, maximum_x = img.shape

    max_ratio_hori, max_ratio_free = 1.0, 1.0
    for box in free_list:
        rect = np.array(box, dtype="float32")
        transformed_img = _four_point_transform(img, rect)
        ratio = _calculate_ratio(transformed_img.shape[1], transformed_img.shape[0])
        new_width = int(model_height * ratio)
        if new_width == 0:
            continue
        crop_img, ratio = _compute_ratio_and_resize(
            transformed_img,
            transformed_img.shape[1],
            transformed_img.shape[0],
            model_height,
        )
        image_list.append((box, crop_img))
        max_ratio_free = max(ratio, max_ratio_free)

    max_ratio_free = math.ceil(max_ratio_free)

    for box in horizontal_list:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        crop_img = img[y_min:y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        ratio = _calculate_ratio(width, height)
        new_width = int(model_height * ratio)
        if new_width == 0:
            continue
        crop_img, ratio = _compute_ratio_and_resize(
            crop_img, width, height, model_height
        )
        image_list.append(
            ([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img)
        )
        max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio) * model_height

    if sort_output:
        image_list = sorted(image_list, key=lambda item: item[0][0][1])
    return image_list, max_width


# ---------------------------------------------------------------------------
# Recognizer preprocessing (from easyocr/recognition.py — pure numpy/OpenCV)
# ---------------------------------------------------------------------------


def _contrast_grey(img: np.ndarray) -> tuple[float, float, float]:
    high = float(np.percentile(img, 90))
    low = float(np.percentile(img, 10))
    return (high - low) / max(10, high + low), high, low


def _adjust_contrast_grey(img: np.ndarray, target: float = 0.4) -> np.ndarray:
    contrast, high, low = _contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / max(10, high - low)
        img = (img - low + 25) * ratio
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _normalize_pad(
    img: np.ndarray,
    max_size: tuple[int, int, int],
) -> np.ndarray:
    """
    Convert grayscale numpy image (H, W) uint8 to normalised + right-padded array.
    Replaces the torch-based NormalizePAD.
    max_size: (channels, height, max_width)
    """
    c, h, max_w = max_size
    # Convert to float32 in [0, 1], then normalise to [-1, 1]
    img_arr = img.astype(np.float32) / 255.0
    img_arr = (img_arr - 0.5) / 0.5

    if img_arr.ndim == 2:
        img_arr = img_arr[np.newaxis, :, :]  # (1, H, W)

    _, ih, iw = img_arr.shape
    pad_img = np.zeros((c, h, max_w), dtype=np.float32)
    pad_img[:, :, :iw] = img_arr
    # Right-pad by repeating last column
    if max_w != iw:
        pad_img[:, :, iw:] = img_arr[:, :, iw - 1 : iw]

    return pad_img


def _prepare_recognizer_input(
    image_list: list[np.ndarray],
    imgH: int,
    imgW: int,
    adjust_contrast: float = 0.0,
) -> np.ndarray:
    """
    Prepare a batch of grayscale crops for the recognizer.
    Returns: (batch, 1, imgH, imgW) float32 array.
    Replaces AlignCollate + ListDataset + DataLoader.
    """
    resized_images = []
    for img in image_list:
        h, w = img.shape[:2]

        if adjust_contrast > 0:
            img = _adjust_contrast_grey(img, target=adjust_contrast)

        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = math.ceil(imgH * ratio)

        resized_image = cv2.resize(
            img, (resized_w, imgH), interpolation=cv2.INTER_CUBIC
        )
        padded = _normalize_pad(resized_image, (1, imgH, imgW))
        resized_images.append(padded)

    return np.stack(resized_images, axis=0)  # (B, 1, imgH, imgW)


# ---------------------------------------------------------------------------
# CTC Label Converter (from easyocr/utils.py — no torch dependency)
# ---------------------------------------------------------------------------


class CTCLabelConverter:
    """Convert between text-label and text-index (numpy only)."""

    def __init__(
        self,
        character: str | list[str],
        separator_list: dict | None = None,
        dict_pathlist: dict | None = None,
    ):
        if separator_list is None:
            separator_list = {}
        if dict_pathlist is None:
            dict_pathlist = {}

        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ["[blank]"] + dict_character

        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep
        self.ignore_idx = [0] + [i + 1 for i, _ in enumerate(separator_char)]

        # Load dictionaries
        if len(separator_list) == 0:
            dict_list: list | dict = []
            for lang, dict_path in dict_pathlist.items():
                try:
                    with open(dict_path, "r", encoding="utf-8-sig") as f:
                        dict_list += f.read().splitlines()  # type: ignore[union-attr]
                except Exception:
                    pass
        else:
            dict_list = {}
            for lang, dict_path in dict_pathlist.items():
                with open(dict_path, "r", encoding="utf-8-sig") as f:
                    dict_list[lang] = f.read().splitlines()

        self.dict_list = dict_list

    def decode_greedy(
        self, text_index: np.ndarray, length: np.ndarray | list
    ) -> list[str]:
        texts = []
        index = 0
        for l in length:
            t = text_index[index : index + l]
            a = np.insert(~(t[1:] == t[:-1]), 0, True)
            b = ~np.isin(t, np.array(self.ignore_idx))
            c = a & b
            text = "".join(np.array(self.character)[t[c.nonzero()]])
            texts.append(text)
            index += l
        return texts


# ---------------------------------------------------------------------------
# Recognizer post-processing (from easyocr/recognition.py — pure numpy)
# ---------------------------------------------------------------------------


def _custom_mean(x: np.ndarray) -> float:
    return float(x.prod() ** (2.0 / np.sqrt(len(x))))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _recognizer_predict(
    session: ort.InferenceSession,
    input_name: str,
    converter: CTCLabelConverter,
    images: np.ndarray,
    batch_max_length: int,
    ignore_idx: list[int],
    decoder: str = "greedy",
) -> list[tuple[str, float]]:
    """
    Run recognizer inference on a batch of preprocessed images.
    images: (B, 1, H, W) float32 numpy array
    Returns list of (text, confidence) tuples.
    """
    batch_size = images.shape[0]
    preds = session.run(None, {input_name: images})[0]  # (B, T, num_class)

    preds_size = [preds.shape[1]] * batch_size

    # Softmax + filter ignore chars (NOT the blank token at index 0 —
    # the blank is essential for CTC decoding to work correctly)
    preds_prob = _softmax(preds, axis=2)
    if len(ignore_idx) > 0:
        preds_prob[:, :, ignore_idx] = 0.0
        pred_norm = preds_prob.sum(axis=2)
        preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)

    result = []
    if decoder == "greedy":
        preds_index = preds_prob.argmax(axis=2)
        preds_index_flat = preds_index.reshape(-1)
        preds_str = converter.decode_greedy(preds_index_flat, preds_size)
    else:
        # Default to greedy for simplicity; beam search could be added later
        preds_index = preds_prob.argmax(axis=2)
        preds_index_flat = preds_index.reshape(-1)
        preds_str = converter.decode_greedy(preds_index_flat, preds_size)

    values = preds_prob.max(axis=2)
    indices = preds_prob.argmax(axis=2)
    for i, (pred, v, idx) in enumerate(zip(preds_str, values, indices)):
        max_probs = v[idx != 0]
        if len(max_probs) > 0:
            confidence = _custom_mean(max_probs)
        else:
            confidence = 0.0
        result.append((pred, confidence))

    return result


# ---------------------------------------------------------------------------
# Character set for English G2 model
# ---------------------------------------------------------------------------

# The English G2 model's character set — extracted from easyocr config.
# This must match what was used during training of the english_g2.pth model.
ENGLISH_G2_CHARACTERS = (
    "0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


# ---------------------------------------------------------------------------
# Main Reader class
# ---------------------------------------------------------------------------


class Reader:
    """
    ONNX-based text reader that replicates the EasyOCR pipeline without PyTorch.

    Usage:
        reader = Reader(
            detector_onnx_path="models/craft_mlt_25k.onnx",
            recognizer_onnx_path="models/english_g2.onnx",
        )
        results = reader.readtext("photo.jpg")
        for bbox, text, confidence in results:
            print(f"{text} ({confidence:.2f})")
    """

    def __init__(
        self,
        detector_onnx_path: str,
        recognizer_onnx_path: str,
        character: str | list[str] | None = None,
        imgH: int = 64,
        providers: list[str] | None = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if character is None:
            character = ENGLISH_G2_CHARACTERS

        # --- Load ONNX sessions ---
        self.detector_session = ort.InferenceSession(
            detector_onnx_path, providers=providers
        )
        self.recognizer_session = ort.InferenceSession(
            recognizer_onnx_path, providers=providers
        )

        # Discover input names
        self.detector_input_name = self.detector_session.get_inputs()[0].name
        self.recognizer_input_name = self.recognizer_session.get_inputs()[0].name

        # --- Set up recognizer converter ---
        self.character = character
        self.converter = CTCLabelConverter(character)
        self.imgH = imgH

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------

    def detect(
        self,
        img: np.ndarray,
        min_size: int = 20,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        add_margin: float = 0.1,
        optimal_num_chars: int | None = None,
    ) -> tuple[list, list]:
        """
        Run text detection on an RGB image.
        Returns (horizontal_list, free_list).
        """
        # Preprocess
        img_resized, target_ratio, size_heatmap = _resize_aspect_ratio(
            img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio

        x = _normalize_mean_variance(img_resized)
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, axis=0)  # add batch dim -> (1, 3, H, W)

        # Inference
        outputs = self.detector_session.run(
            None, {self.detector_input_name: x.astype(np.float32)}
        )
        y = outputs[0]  # (1, H/2, W/2, 2)

        # Post-process
        score_text = y[0, :, :, 0]
        score_link = y[0, :, :, 1]

        estimate_num_chars = optimal_num_chars is not None
        boxes, polys, mapper = _get_det_boxes(
            score_text,
            score_link,
            text_threshold,
            link_threshold,
            low_text,
            poly=False,
            estimate_num_chars=estimate_num_chars,
        )

        boxes = _adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = _adjust_result_coordinates(polys, ratio_w, ratio_h)

        if estimate_num_chars and mapper is not None:
            boxes = list(boxes)
            polys = list(polys)
            for k in range(len(polys)):
                if estimate_num_chars:
                    boxes[k] = (boxes[k], mapper[k])
                if polys[k] is None:
                    polys[k] = boxes[k]

        # Flatten polys into [x1,y1,...,x4,y4] format
        result_polys = []
        for poly in polys:
            p = np.array(poly)
            if p.ndim == 2:
                result_polys.append(p.astype(np.int32).reshape(-1))
            else:
                result_polys.append(p)

        # Group into horizontal + free lists
        horizontal_list, free_list = _group_text_box(
            result_polys,
            slope_ths,
            ycenter_ths,
            height_ths,
            width_ths,
            add_margin,
            (optimal_num_chars is None),
        )

        if min_size:
            horizontal_list = [
                i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > min_size
            ]
            free_list = [
                i
                for i in free_list
                if max(
                    max(c[0] for c in i) - min(c[0] for c in i),
                    max(c[1] for c in i) - min(c[1] for c in i),
                )
                > min_size
            ]

        return horizontal_list, free_list

    # -----------------------------------------------------------------------
    # Recognition
    # -----------------------------------------------------------------------

    def recognize(
        self,
        img_cv_grey: np.ndarray,
        horizontal_list: list | None = None,
        free_list: list | None = None,
        decoder: str = "greedy",
        batch_size: int = 1,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        filter_ths: float = 0.003,
        detail: int = 1,
        output_format: Literal["standard", "dict"] = "standard",
    ) -> list:
        """
        Run text recognition on detected regions.
        """
        if horizontal_list is None and free_list is None:
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []
        if horizontal_list is None:
            horizontal_list = []
        if free_list is None:
            free_list = []

        # ignore_idx is built from ignore_char — empty by default.
        # Important: do NOT include index 0 (the CTC blank token) here;
        # the blank is essential for correct CTC greedy decoding.
        ignore_idx: list[int] = []

        result = []
        for bbox in horizontal_list:
            h_list = [bbox]
            image_list, max_width = _get_image_list(
                h_list, [], img_cv_grey, model_height=self.imgH
            )
            if not image_list:
                continue
            batch_result = self._recognize_crops(
                image_list,
                max_width,
                ignore_idx,
                decoder,
                contrast_ths,
                adjust_contrast,
            )
            result += batch_result

        for bbox in free_list:
            image_list, max_width = _get_image_list(
                [], [bbox], img_cv_grey, model_height=self.imgH
            )
            if not image_list:
                continue
            batch_result = self._recognize_crops(
                image_list,
                max_width,
                ignore_idx,
                decoder,
                contrast_ths,
                adjust_contrast,
            )
            result += batch_result

        if filter_ths > 0:
            result = [r for r in result if r[2] > filter_ths]

        if detail == 0:
            return [item[1] for item in result]
        elif output_format == "dict":
            return [
                {"boxes": item[0], "text": item[1], "confident": item[2]}
                for item in result
            ]
        else:
            return result

    def _recognize_crops(
        self,
        image_list: list,
        max_width: int,
        ignore_idx: list[int],
        decoder: str,
        contrast_ths: float,
        adjust_contrast: float,
    ) -> list[tuple[list, str, float]]:
        """Run recognition on a list of (box, crop_image) tuples."""
        imgW = int(max_width)
        batch_max_length = int(imgW / 10)

        coords = [item[0] for item in image_list]
        img_list = [item[1] for item in image_list]

        # First pass
        batch_input = _prepare_recognizer_input(img_list, self.imgH, imgW)
        result1 = _recognizer_predict(
            self.recognizer_session,
            self.recognizer_input_name,
            self.converter,
            batch_input,
            batch_max_length,
            ignore_idx,
            decoder,
        )

        # Second pass for low confidence results with contrast adjustment
        low_confident_idx = [
            i for i, item in enumerate(result1) if item[1] < contrast_ths
        ]

        result2_map: dict[int, tuple[str, float]] = {}
        if len(low_confident_idx) > 0:
            img_list2 = [img_list[i] for i in low_confident_idx]
            batch_input2 = _prepare_recognizer_input(
                img_list2, self.imgH, imgW, adjust_contrast=adjust_contrast
            )
            result2 = _recognizer_predict(
                self.recognizer_session,
                self.recognizer_input_name,
                self.converter,
                batch_input2,
                batch_max_length,
                ignore_idx,
                decoder,
            )
            for j, orig_idx in enumerate(low_confident_idx):
                result2_map[orig_idx] = result2[j]

        # Merge results
        final_result = []
        for i, (coord, pred1) in enumerate(zip(coords, result1)):
            if i in result2_map:
                pred2 = result2_map[i]
                if pred1[1] > pred2[1]:
                    final_result.append((coord, pred1[0], pred1[1]))
                else:
                    final_result.append((coord, pred2[0], pred2[1]))
            else:
                final_result.append((coord, pred1[0], pred1[1]))

        return final_result

    # -----------------------------------------------------------------------
    # High-level API
    # -----------------------------------------------------------------------

    def readtext(
        self,
        image,
        decoder: str = "greedy",
        batch_size: int = 1,
        detail: int = 1,
        min_size: int = 20,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        add_margin: float = 0.1,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        filter_ths: float = 0.003,
        output_format: Literal["standard", "dict"] = "standard",
        optimal_num_chars: int | None = None,
    ) -> list:
        """
        Main entry point: detect + recognize text in an image.

        Args:
            image: file path (str), bytes, numpy array (RGB or BGR or grayscale),
                   or PIL Image (requires optional Pillow install).
            decoder: 'greedy' (default). Beam search not yet implemented.
            detail: 0 = text only, 1 = (bbox, text, confidence)
            output_format: 'standard' (list of tuples) or 'dict'
            ... and all detection/recognition thresholds.

        Returns:
            List of (bounding_box, text, confidence) tuples (when detail=1),
            or list of text strings (when detail=0).
        """
        img, img_cv_grey = reformat_input(image)

        # Detect
        horizontal_list, free_list = self.detect(
            img,
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths,
            height_ths=height_ths,
            width_ths=width_ths,
            add_margin=add_margin,
            optimal_num_chars=optimal_num_chars,
        )

        # Recognize
        result = self.recognize(
            img_cv_grey,
            horizontal_list=horizontal_list,
            free_list=free_list,
            decoder=decoder,
            batch_size=batch_size,
            contrast_ths=contrast_ths,
            adjust_contrast=adjust_contrast,
            filter_ths=filter_ths,
            detail=detail,
            output_format=output_format,
        )

        return result
