"""
EasyOCR-ONNX Demo

Demonstrates text detection + recognition using ONNX Runtime
instead of PyTorch. No torch dependency required at runtime.

This script delegates to the installed easyocr-onnx CLI.
You can also run: easyocr-onnx <image_path>
"""

from easyocr_onnx.cli import main

if __name__ == "__main__":
    main()
