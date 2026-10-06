"""Copy JPEG files from a dataset tree into one numbered image folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="root directory to search recursively")
    parser.add_argument("--destination", type=Path, default=PROJECT_DIR / "dataset" / "images")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    pictures = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})
    for index, image_path in enumerate(pictures, start=1):
        target = destination / f"img_{index:03d}.jpg"
        shutil.copy2(image_path, target)
        print(f"{image_path} -> {target.name}")
    print(f"Copied {len(pictures)} image(s) to {destination}")


if __name__ == "__main__":
    main()
