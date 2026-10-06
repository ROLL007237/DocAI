"""Train a YOLO detector for photo and MRZ regions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT_DIR / "yolo26n.pt")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="mps", help="mps for Apple Silicon, cpu for Intel Macs")
    parser.add_argument("--project", type=Path, default=PROJECT_DIR / "runs" / "detect")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    model = YOLO(args.weights.expanduser().resolve())
    model.train(data=str(args.data.expanduser().resolve()), epochs=args.epochs, imgsz=args.image_size,
                device=args.device, plots=True, project=str(args.project.expanduser().resolve()))


if __name__ == "__main__":
    main()
