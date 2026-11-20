"""Fine-tune RTMDet-Ins-tiny from the Apache-2.0 MMDetection implementation.

RTMDet-Ins-tiny is a real-time instance segmenter (about 5.6M parameters,
640 input, about 1.7 ms TensorRT FP16 on a 3090). Use this checkpoint when
the deployed product cannot take Ultralytics AGPL-3.0 or a paid enterprise
license. It does not compile through the Ultralytics Hailo exporter. The
Hailo-8 model is YOLOv8n-seg.

MMYOLO also ships RTMDet, and that repository is GPL-3.0. This trainer loads
the config and COCO-pretrained weights from MMDetection only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mmdet
from mmdet.utils import register_all_modules, setup_cache_size_limit_of_dynamo
from mmengine.config import Config
from mmengine.runner import Runner

from yolo_seg_hailo.coco_seg import class_names, resolve_dataset_dir, write_mmdet_annotations
from yolo_seg_hailo.recipe import HAILO_IMGSZ, RTMDET_HSV

COCO_PRETRAIN = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet-ins_tiny_8xb32-300e_coco/"
    "rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
)
TINY_CONFIG = "rtmdet-ins_tiny_8xb32-300e_coco.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Roboflow COCO export. Defaults to ./dataset")
    parser.add_argument("--annotations", type=Path, default=None, help="Remapped COCO json directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4, help="4 is safe on an 8 GB GPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--work-dir", default="runs/rtmdet-ins-tiny")
    parser.add_argument("--imgsz", type=int, default=HAILO_IMGSZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.imgsz != HAILO_IMGSZ:
        raise SystemExit(f"imgsz must stay {HAILO_IMGSZ} so this model matches the YOLO input size.")

    setup_cache_size_limit_of_dynamo()
    register_all_modules(init_default_scope=True)

    dataset_dir = resolve_dataset_dir(args.dataset)
    annotation_dir = args.annotations or dataset_dir.parent / "dataset-coco"
    annotation_paths = write_mmdet_annotations(dataset_dir, annotation_dir)
    names = class_names(dataset_dir)

    cfg = Config.fromfile(str(_tiny_config_path()))
    _point_at_dataset(cfg, dataset_dir, annotation_paths, names)
    _fit_schedule(cfg, args.epochs, args.batch, args.workers, args.work_dir)
    _apply_color_and_flips(cfg)
    _use_batchnorm_on_one_gpu(cfg)
    cfg.load_from = COCO_PRETRAIN
    cfg.resume = False

    runner = Runner.from_cfg(cfg)
    runner.train()
    print(f"Checkpoints: {Path(args.work_dir).resolve()}")
    print("Deploy the best checkpoint with MMDetection or MMDeploy. License: Apache-2.0.")


def _tiny_config_path() -> Path:
    package_root = Path(mmdet.__file__).resolve().parent
    path = package_root / ".mim" / "configs" / "rtmdet" / TINY_CONFIG
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Install the RTMDet extra with: uv sync --extra rtmdet"
        )
    return path


def _point_at_dataset(
    cfg: Config,
    dataset_dir: Path,
    annotation_paths: dict[str, Path],
    class_names: list[str],
) -> None:
    cfg.model.bbox_head.num_classes = len(class_names)
    metainfo = dict(classes=tuple(class_names))
    split_to_loader = {
        "train": cfg.train_dataloader,
        "valid": cfg.val_dataloader,
    }
    for split, loader in split_to_loader.items():
        dataset = loader.dataset
        dataset.data_root = ""
        dataset.ann_file = str(annotation_paths[split])
        dataset.data_prefix = dict(img=f"{(dataset_dir / split).as_posix()}/")
        dataset.metainfo = metainfo
        dataset.filter_cfg = dict(filter_empty_gt=False, min_size=32)
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator.ann_file = str(annotation_paths["valid"])
    cfg.val_evaluator.classwise = True
    cfg.test_evaluator = cfg.val_evaluator


def _fit_schedule(cfg: Config, epochs: int, batch: int, workers: int, work_dir: str) -> None:
    cfg.work_dir = work_dir
    cfg.train_cfg.max_epochs = epochs
    cfg.train_cfg.val_interval = 10
    cfg.train_cfg.dynamic_intervals = [(max(epochs - 10, 1), 1)]
    cfg.train_dataloader.batch_size = batch
    cfg.train_dataloader.num_workers = workers
    cfg.train_dataloader.persistent_workers = workers > 0
    cfg.val_dataloader.num_workers = workers
    cfg.val_dataloader.persistent_workers = workers > 0
    if cfg.get("auto_scale_lr"):
        cfg.auto_scale_lr.enable = True
        cfg.auto_scale_lr.base_batch_size = 256

    for scheduler in cfg.param_scheduler:
        if scheduler.get("type") == "LinearLR":
            scheduler["end"] = 50
        elif scheduler.get("type") == "CosineAnnealingLR":
            scheduler["begin"] = epochs // 2
            scheduler["end"] = epochs
            scheduler["T_max"] = max(epochs // 2, 1)
            base_lr = cfg.optim_wrapper.optimizer.lr
            scheduler["eta_min"] = base_lr * 0.05

    for hook in cfg.custom_hooks:
        if hook.get("type") == "PipelineSwitchHook":
            hook["switch_epoch"] = max(epochs - 10, 1)

    checkpoint = cfg.default_hooks.checkpoint
    checkpoint.interval = 10
    checkpoint.save_best = "coco/segm_mAP"
    checkpoint.rule = "greater"
    checkpoint.max_keep_ckpts = 3


def _apply_color_and_flips(cfg: Config) -> None:
    pipelines = [cfg.train_dataloader.dataset.pipeline]
    for hook in cfg.custom_hooks:
        if hook.get("type") == "PipelineSwitchHook":
            pipelines.append(hook.switch_pipeline)
    for pipeline in pipelines:
        _tune_pipeline(pipeline)


def _tune_pipeline(pipeline: list) -> None:
    insert_at: int | None = None
    for index, step in enumerate(pipeline):
        if step.get("type") == "YOLOXHSVRandomAug":
            step.update(RTMDET_HSV)
        elif step.get("type") == "RandomFlip" and step.get("direction", "horizontal") == "horizontal":
            step["prob"] = 0.5
            insert_at = index + 1
        elif step.get("type") == "CachedMixUp":
            step["prob"] = 0.0
    has_vertical = any(
        step.get("type") == "RandomFlip" and step.get("direction") == "vertical" for step in pipeline
    )
    if insert_at is not None and not has_vertical:
        pipeline.insert(insert_at, dict(type="RandomFlip", prob=0.5, direction="vertical"))


def _use_batchnorm_on_one_gpu(cfg: Config) -> None:
    """SyncBN expects a process group. One GPU on Windows trains with BN."""

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "SyncBN":
                node["type"] = "BN"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(cfg.model)


if __name__ == "__main__":
    main()
