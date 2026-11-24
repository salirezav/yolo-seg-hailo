"""Train YOLO26n-seg for local / non-Hailo deployment.

Use this when the product does not need a Hailo HEF. YOLO26n-seg is the
current Ultralytics nano segmenter (end-to-end, NMS-free). The Hailo-8 path
stays on train_yolov8n_seg.py because that export targets YOLOv8n-seg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from yolo_seg_hailo.coco_seg import convert_to_yolo_seg, resolve_dataset_dir
from yolo_seg_hailo.recipe import HAILO_IMGSZ, YOLO26_WEIGHTS, YOLO_AUGMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Roboflow COCO export. Defaults to ./dataset")
    parser.add_argument("--output", type=Path, default=None, help="Converted YOLO dataset. Defaults to ./dataset-yolo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8, help="8 fits an 8 GB laptop GPU at 640")
    parser.add_argument("--imgsz", type=int, default=HAILO_IMGSZ)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="yolo26n-seg")
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset)
    yolo_dir = args.output or dataset_dir.parent / "dataset-yolo"
    data_yaml = convert_to_yolo_seg(dataset_dir, yolo_dir)
    # An absolute project path stays in this repo. Ultralytics otherwise joins a
    # relative project onto the global runs_dir from its user settings.
    project = Path(args.project)
    if not project.is_absolute():
        project = dataset_dir.parent / project

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU.")
        device = "cpu"

    model = YOLO(YOLO26_WEIGHTS)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=args.patience,
        amp=True,
        cache=True,
        rect=False,
        plots=True,
        **YOLO_AUGMENT,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print()
    print(f"Checkpoint: {best}")
    print("This run is for non-Hailo deployment. Use train-yolov8n-seg when you need a HEF.")


if __name__ == "__main__":
    main()
