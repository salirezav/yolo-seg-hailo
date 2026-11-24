"""Read the Roboflow COCO Segmentation export and write trainer-specific copies."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

SPLITS = ("train", "valid", "test")
IGNORED_CLASSES = frozenset({"pecan-surface"})


def resolve_dataset_dir(path: Path | None) -> Path:
    """Find the Roboflow export that contains train/valid/test COCO json files."""
    if path is not None:
        dataset_dir = path.expanduser().resolve()
    else:
        cwd_dataset = Path.cwd() / "dataset"
        dataset_dir = cwd_dataset if _is_roboflow_coco(cwd_dataset) else _search_parents()
    if not _is_roboflow_coco(dataset_dir):
        raise SystemExit(
            f"COCO segmentation export not found at {dataset_dir}. "
            "Expected train/valid/test folders with _annotations.coco.json."
        )
    return dataset_dir


def convert_to_yolo_seg(dataset_dir: Path, output_dir: Path) -> Path:
    """Write Ultralytics segmentation labels and return the data.yaml path.

    Polygon annotations are copied through. Compressed RLE masks, which this
    export mixes in, are traced to contours so every instance becomes a polygon.
    """
    output_dir = output_dir.resolve()
    class_names = _require_consistent_classes(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {"polygons": 0, "rle": 0, "dropped": 0}

    for split in SPLITS:
        coco = _load(dataset_dir / split / "_annotations.coco.json")
        names = _class_names(coco)
        if names != class_names:
            raise SystemExit(f"{split} classes {names} do not match train classes {class_names}")
        class_index = {category["id"]: index for index, category in enumerate(_sorted_categories(coco))}
        images_by_id = {image["id"]: image for image in coco["images"]}
        grouped: dict[int, list[dict]] = {image_id: [] for image_id in images_by_id}
        for annotation in coco["annotations"]:
            if annotation["category_id"] not in class_index:
                totals["ignored"] = totals.get("ignored", 0) + 1
                continue
            grouped.setdefault(annotation["image_id"], []).append(annotation)

        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image in coco["images"]:
            source = dataset_dir / split / image["file_name"]
            if not source.is_file():
                raise SystemExit(f"Missing image {source}")
            _link_or_copy(source, image_dir / source.name)
            lines: list[str] = []
            for annotation in grouped.get(image["id"], []):
                class_id = class_index[annotation["category_id"]]
                for polygon in _polygons_for_annotation(annotation, image, totals):
                    coords = _normalize_polygon(polygon, image["width"], image["height"])
                    if coords is None:
                        totals["dropped"] += 1
                        continue
                    lines.append(f"{class_id} {coords}")
            (label_dir / f"{source.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

    yaml_path = output_dir / "data.yaml"
    _write_data_yaml(yaml_path, class_names)
    print(
        f"Wrote {yaml_path} ({len(class_names)} classes, "
        f"{totals['polygons']} polygons, {totals['rle']} RLE masks converted, "
        f"{totals['dropped']} unusable contours dropped, "
        f"{totals.get('ignored', 0)} ignored-class annotations skipped)"
    )
    return yaml_path


def class_names(dataset_dir: Path) -> list[str]:
    return _require_consistent_classes(dataset_dir)


def write_mmdet_annotations(dataset_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Rewrite category ids to 1..N. MMDetection treats id 0 as a missing class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = _require_consistent_classes(dataset_dir)
    written: dict[str, Path] = {}
    for split in SPLITS:
        coco = _load(dataset_dir / split / "_annotations.coco.json")
        categories = _sorted_categories(coco)
        id_map = {category["id"]: index + 1 for index, category in enumerate(categories)}
        remapped = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": coco["images"],
            "annotations": [
                {**annotation, "category_id": id_map[annotation["category_id"]]}
                for annotation in coco["annotations"]
                if annotation["category_id"] in id_map
            ],
            "categories": [
                {
                    "id": id_map[category["id"]],
                    "name": category["name"],
                    "supercategory": category.get("supercategory", "object"),
                }
                for category in categories
            ],
        }
        destination = output_dir / f"{split}.json"
        destination.write_text(json.dumps(remapped), encoding="utf-8")
        written[split] = destination
    print(f"Wrote MMDetection COCO json for {len(class_names)} classes under {output_dir}")
    return written


def _require_consistent_classes(dataset_dir: Path) -> list[str]:
    names = _class_names(_load(dataset_dir / "train" / "_annotations.coco.json"))
    if not names:
        raise SystemExit(f"No categories in {dataset_dir / 'train' / '_annotations.coco.json'}")
    return names


def _is_roboflow_coco(path: Path) -> bool:
    return all((path / split / "_annotations.coco.json").is_file() for split in SPLITS)


def _search_parents() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "dataset"
        if _is_roboflow_coco(candidate):
            return candidate
    return Path.cwd() / "dataset"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sorted_categories(coco: dict) -> list[dict]:
    kept = [category for category in coco["categories"] if category["name"] not in IGNORED_CLASSES]
    return sorted(kept, key=lambda category: category["id"])


def _class_names(coco: dict) -> list[str]:
    return [category["name"] for category in _sorted_categories(coco)]


def _polygons_for_annotation(annotation: dict, image: dict, totals: dict[str, int]) -> list[list[float]]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        totals["polygons"] += 1
        rings: list[list[float]] = []
        for ring in segmentation:
            if isinstance(ring, list) and len(ring) >= 6:
                rings.append([float(value) for value in ring])
        return rings
    if isinstance(segmentation, dict) and "counts" in segmentation:
        totals["rle"] += 1
        return _rle_to_polygons(segmentation, int(image["height"]), int(image["width"]))
    totals["dropped"] += 1
    return []


def _rle_to_polygons(segmentation: dict, height: int, width: int) -> list[list[float]]:
    counts = segmentation["counts"]
    rle = {
        "size": segmentation.get("size", [height, width]),
        "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
    }
    decoded = mask_util.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    mask = np.ascontiguousarray(decoded, dtype=np.uint8)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        contour = _limit_points(contour)
        flat = contour.reshape(-1, 2)
        if len(flat) < 3:
            continue
        polygons.append([float(value) for point in flat for value in point])
    return polygons


def _limit_points(contour: np.ndarray) -> np.ndarray:
    if len(contour) <= 500:
        return contour
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approximated = cv2.approxPolyDP(contour, epsilon, True)
    return approximated if len(approximated) >= 3 else contour


def _normalize_polygon(polygon: list[float], width: int, height: int) -> str | None:
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return None
    coords: list[str] = []
    for index in range(0, len(polygon), 2):
        x = min(1.0, max(0.0, polygon[index] / width))
        y = min(1.0, max(0.0, polygon[index + 1] / height))
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    if len(coords) < 6:
        return None
    return " ".join(coords)


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_data_yaml(path: Path, class_names: list[str]) -> None:
    # Omit path. Ultralytics then uses this file's directory as the dataset root.
    # A path of "." is the process cwd, which is the repo root, not dataset-yolo.
    lines = [
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
