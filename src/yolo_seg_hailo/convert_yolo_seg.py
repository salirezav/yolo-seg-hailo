"""Convert the Roboflow COCO Segmentation export to Ultralytics segment labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo_seg_hailo.coco_seg import convert_to_yolo_seg, resolve_dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Roboflow COCO export. Defaults to ./dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ultralytics dataset directory. Defaults to dataset-yolo next to the export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset)
    output_dir = args.output or dataset_dir.parent / "dataset-yolo"
    convert_to_yolo_seg(dataset_dir, output_dir)


if __name__ == "__main__":
    main()
