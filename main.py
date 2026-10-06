"""Extract document photo and MRZ areas with a trained YOLO model."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import pytesseract
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ProcessingPaths:
    source: Path
    destination: Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / "incoming", help="folder containing document images")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "output", help="folder for extracted results")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "model" / "best.pt", help="path to trained YOLO weights")
    parser.add_argument("--tesseract", type=Path, help="path to tesseract executable; omit when it is in PATH")
    return parser.parse_args()


def find_images(folder: Path) -> list[Path]:
    return sorted(item for item in folder.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGES)


def crop_image(image, coordinates: tuple[int, int, int, int]):
    height, width = image.shape[:2]
    left, top, right, bottom = coordinates
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    return image[top:bottom, left:right]


def recognize_mrz(mrz_image) -> str:
    grayscale = cv2.cvtColor(mrz_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(binary, lang="eng", config="--psm 6").strip()


class DocumentProcessor:
    def __init__(self, model_path: Path, output_folder: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        self.detector = YOLO(model_path)
        self.output_folder = output_folder

    def process(self, source: Path) -> tuple[bool, bool]:
        paths = ProcessingPaths(source=source, destination=self.output_folder / source.stem)
        paths.destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.source, paths.destination / paths.source.name)
        image = cv2.imread(str(paths.source))
        if image is None:
            raise ValueError("OpenCV could not read the image")

        photo_saved = mrz_saved = False
        detections: list[str] = []
        result = self.detector(str(paths.source), verbose=False)[0]
        for box in (result.boxes if result.boxes is not None else ()):
            class_id = int(box.cls[0])
            label = self.detector.names[class_id]
            confidence = float(box.conf[0])
            coordinates = tuple(map(int, box.xyxy[0].tolist()))
            detections.append(f"{label} {confidence:.3f} {' '.join(map(str, coordinates))}")
            area = crop_image(image, coordinates)
            if area.size == 0:
                continue
            if label == "photo":
                cv2.imwrite(str(paths.destination / "face.jpg"), area)
                photo_saved = True
            elif label == "mrz":
                cv2.imwrite(str(paths.destination / "mrz.jpg"), area)
                (paths.destination / "mrz.txt").write_text(recognize_mrz(area), encoding="utf-8")
                mrz_saved = True

        (paths.destination / "boxes.txt").write_text("\n".join(detections), encoding="utf-8")
        return photo_saved, mrz_saved


def main() -> None:
    args = parse_arguments()
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = str(args.tesseract.expanduser())
    input_folder = args.input.expanduser().resolve()
    output_folder = args.output.expanduser().resolve()
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    images = find_images(input_folder)
    if not images:
        print(f"No supported images in {input_folder}. Add files and run the command again.")
        return

    print("Loading model…")
    processor = DocumentProcessor(args.model.expanduser().resolve(), output_folder)
    print(f"Processing {len(images)} image(s).")
    for number, image_path in enumerate(images, start=1):
        try:
            photo, mrz = processor.process(image_path)
            print(f"[{number}/{len(images)}] {image_path.name}: photo={'OK' if photo else 'missing'}, mrz={'OK' if mrz else 'missing'}")
        except (OSError, ValueError, pytesseract.TesseractError, pytesseract.TesseractNotFoundError) as error:
            print(f"[{number}/{len(images)}] {image_path.name}: failed — {error}")
    print(f"Results: {output_folder}")


if __name__ == "__main__":
    main()
