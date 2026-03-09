"""
EasyOCR-ONNX command-line interface.

Usage:
    easyocr-onnx path/to/image.jpg
    easyocr-onnx path/to/image.jpg --detector models/craft_mlt_25k.onnx --recognizer models/english_g2.onnx
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="easyocr-onnx",
        description="Lightweight OCR engine running EasyOCR models via ONNX Runtime",
    )
    parser.add_argument(
        "image",
        help="Path to the image file to read text from",
    )
    parser.add_argument(
        "--detector",
        default=None,
        help=(
            "Path to the CRAFT detector ONNX model "
            "(default: models/craft_mlt_25k.onnx relative to cwd)"
        ),
    )
    parser.add_argument(
        "--recognizer",
        default=None,
        help=(
            "Path to the English G2 recognizer ONNX model "
            "(default: models/english_g2.onnx relative to cwd)"
        ),
    )
    parser.add_argument(
        "--detail",
        type=int,
        choices=[0, 1],
        default=1,
        help="0 = text only, 1 = bounding box + text + confidence (default: 1)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=20,
        help="Minimum text region size in pixels (default: 20)",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=2560,
        help="Maximum image dimension for detection (default: 2560)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.7,
        help="Text confidence threshold (default: 0.7)",
    )
    parser.add_argument(
        "--low-text",
        type=float,
        default=0.4,
        help="Text low-bound score (default: 0.4)",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.4,
        help="Link confidence threshold (default: 0.4)",
    )

    args = parser.parse_args(argv)

    # Resolve model paths --------------------------------------------------
    models_dir = os.path.join(os.getcwd(), "models")

    detector_path = args.detector or os.path.join(models_dir, "craft_mlt_25k.onnx")
    recognizer_path = args.recognizer or os.path.join(models_dir, "english_g2.onnx")

    if not os.path.isfile(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    for path, name in [
        (detector_path, "Detector (CRAFT)"),
        (recognizer_path, "Recognizer (English G2)"),
    ]:
        if not os.path.isfile(path):
            print(f"Error: {name} ONNX model not found at: {path}", file=sys.stderr)
            print(
                "Run `notebooks/convert.py` first to export the ONNX models, "
                "or specify the path with --detector / --recognizer.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Lazy-import so --help is fast and doesn't load heavy deps -----------
    from easyocr_onnx import Reader  # noqa: E402

    print("Loading ONNX models...")
    reader = Reader(
        detector_onnx_path=detector_path,
        recognizer_onnx_path=recognizer_path,
    )

    print(f"Reading text from: {args.image}")
    print()

    results = reader.readtext(
        args.image,
        detail=args.detail,
        min_size=args.min_size,
        canvas_size=args.canvas_size,
        text_threshold=args.text_threshold,
        low_text=args.low_text,
        link_threshold=args.link_threshold,
    )

    if not results:
        print("No text detected.")
        return

    if args.detail == 0:
        for text in results:
            print(text)
    else:
        print(f"Found {len(results)} text region(s):")
        print("-" * 60)
        for i, (bbox, text, confidence) in enumerate(results, 1):
            print(f"  [{i}] {text}")
            print(f"      Confidence: {confidence:.4f}")
            print(f"      Bounding box: {bbox}")
            print()


if __name__ == "__main__":
    main()
