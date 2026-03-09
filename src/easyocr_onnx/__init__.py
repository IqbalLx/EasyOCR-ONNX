"""
EasyOCR-ONNX — Lightweight OCR engine using ONNX Runtime.

Drop-in replacement for easyocr that runs text detection and recognition
models via ONNX Runtime instead of PyTorch. Production dependencies are
only numpy, onnxruntime, and opencv-python-headless (no Pillow, no PyTorch).

Usage:
    from easyocr_onnx import Reader

    reader = Reader(
        detector_onnx_path="models/craft_mlt_25k.onnx",
        recognizer_onnx_path="models/english_g2.onnx",
    )
    results = reader.readtext("photo.jpg")
    for bbox, text, confidence in results:
        print(f"{text} ({confidence:.2f})")
"""

from __future__ import annotations

from easyocr_onnx.reader import (
    ENGLISH_G2_CHARACTERS,
    CTCLabelConverter,
    Reader,
    reformat_input,
)

__all__ = [
    "ENGLISH_G2_CHARACTERS",
    "CTCLabelConverter",
    "Reader",
    "reformat_input",
]
