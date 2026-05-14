#!/usr/bin/env python3
"""Prepare a Kaggle-ready Ultralytics YOLO-seg dataset from semantic PNG masks.

The source dataset is expected to look like:

    yolo26_seg/
      train/
        image_001.jpg
        image_001_mask.png
        _classes.csv
      valid/
        image_101.jpg
        image_101_mask.png
        _classes.csv

This script converts each semantic mask into one or more YOLO segmentation
polygons per connected component, writes a standard Ultralytics directory tree,
and saves `data.yaml` plus a semantic/YOLO class-id mapping file.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_SUFFIX = "_mask.png"
BACKGROUND_MASK_VALUE = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("yolo26_seg"),
        help="Source dataset root containing train/ and valid/ folders.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("yolo26_seg_ultralytics"),
        help="Output dataset root to upload to Kaggle.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=8.0,
        help="Minimum contour area in pixels to keep as a YOLO polygon.",
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.002,
        help="Polygon simplification ratio passed to approxPolyDP.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating hardlinks when possible.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the destination directory before rebuilding it.",
    )
    return parser.parse_args()


def find_classes_csv(src_root: Path) -> Path:
    candidates = sorted(src_root.rglob("_classes.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find _classes.csv under {src_root}")
    return candidates[0]


def load_class_rows(csv_path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pixel_value = int(row["Pixel Value"].strip())
            class_name = row[" Class"].strip() if " Class" in row else row["Class"].strip()
            rows.append((pixel_value, class_name))
    if not rows:
        raise ValueError(f"No class rows found in {csv_path}")
    return rows


def build_foreground_mapping(class_rows: list[tuple[int, str]]) -> tuple[list[str], dict[int, int]]:
    foreground = [(mask_value, name) for mask_value, name in class_rows if mask_value != BACKGROUND_MASK_VALUE]
    if not foreground:
        raise ValueError("No foreground classes found after removing background class 0.")
    foreground.sort(key=lambda item: item[0])
    names = [name for _, name in foreground]
    yolo_id_from_mask = {mask_value: idx for idx, (mask_value, _) in enumerate(foreground)}
    return names, yolo_id_from_mask


def ensure_clean_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_files_for_split(split_dir: Path) -> list[Path]:
    files = []
    for path in sorted(split_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "_classes.csv":
            continue
        if path.name.endswith(MASK_SUFFIX):
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


def mask_path_for_image(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}{MASK_SUFFIX}")


def hardlink_or_copy(src: Path, dst: Path, *, copy_images: bool) -> None:
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        # Roboflow may occasionally save a palette-expanded mask; prefer one channel.
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(mask[:, :, 1], mask[:, :, 2]):
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Expected a single-channel semantic mask, got shape {mask.shape} at {mask_path}")
    return mask.astype(np.uint8, copy=False)


def contour_to_polygon(contour: np.ndarray, width: int, height: int) -> list[float] | None:
    contour = contour.reshape(-1, 2)
    if contour.shape[0] < 3:
        return None

    polygon: list[float] = []
    for x, y in contour:
        x_norm = min(max((float(x) + 0.5) / float(width), 0.0), 1.0)
        y_norm = min(max((float(y) + 0.5) / float(height), 0.0), 1.0)
        polygon.extend((x_norm, y_norm))
    return polygon if len(polygon) >= 6 else None


def mask_to_yolo_lines(
    mask: np.ndarray,
    yolo_id_from_mask: dict[int, int],
    *,
    min_area: float,
    epsilon_ratio: float,
) -> tuple[list[str], dict[int, int]]:
    height, width = mask.shape
    lines: list[str] = []
    instance_counts: dict[int, int] = {yolo_id: 0 for yolo_id in yolo_id_from_mask.values()}

    for mask_value, yolo_id in sorted(yolo_id_from_mask.items()):
        binary = (mask == mask_value).astype(np.uint8)
        if not np.any(binary):
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue

            if epsilon_ratio > 0.0:
                perimeter = float(cv2.arcLength(contour, True))
                epsilon = epsilon_ratio * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if approx.shape[0] >= 3:
                    contour = approx

            polygon = contour_to_polygon(contour, width, height)
            if polygon is None:
                continue

            coord_text = " ".join(f"{value:.6f}" for value in polygon)
            lines.append(f"{yolo_id} {coord_text}")
            instance_counts[yolo_id] += 1

    return lines, instance_counts


def write_text_file(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_data_yaml(dst_root: Path, names: list[str]) -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/valid",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    write_text_file(dst_root / "data.yaml", lines)


def write_mapping_yaml(dst_root: Path, class_rows: list[tuple[int, str]], yolo_id_from_mask: dict[int, int]) -> None:
    lines = [
        "background_mask_value: 0",
        "mask_value_to_yolo_class:",
    ]
    for mask_value, yolo_id in sorted(yolo_id_from_mask.items()):
        lines.append(f"  {mask_value}: {yolo_id}")
    lines.append("yolo_class_to_mask_value:")
    for mask_value, yolo_id in sorted(yolo_id_from_mask.items(), key=lambda item: item[1]):
        lines.append(f"  {yolo_id}: {mask_value}")
    lines.append("mask_classes:")
    for mask_value, class_name in sorted(class_rows, key=lambda item: item[0]):
        lines.append(f"  {mask_value}: {class_name}")
    write_text_file(dst_root / "semantic_map.yaml", lines)


def write_readme(dst_root: Path, names: list[str]) -> None:
    lines = [
        "Ultralytics YOLO-seg dataset prepared from semantic PNG masks.",
        "",
        "What is inside:",
        "- data.yaml: train/val config for Ultralytics.",
        "- semantic_map.yaml: mapping between original mask pixel values and YOLO class ids.",
        "- images/: training and validation images.",
        "- labels/: YOLO segmentation polygons generated from connected components in the masks.",
        "",
        "Important:",
        "- Background is not a YOLO class. Only foreground lane classes are trained.",
        "- YOLO class ids 0..4 correspond to original semantic mask values 1..5.",
        "- When integrating predictions back into this repo, map predicted class_id back to mask_value = class_id + 1.",
        "",
        "Foreground class order:",
    ]
    lines.extend(f"- {idx}: {name}" for idx, name in enumerate(names))
    lines.extend([
        "",
        "Example Kaggle training snippet:",
        "from ultralytics import YOLO",
        "model = YOLO('/kaggle/input/<your-weights>/yolo26n-seg.pt')",
        "model.train(data='/kaggle/input/<your-dataset>/data.yaml', imgsz=640, epochs=100, batch=16)",
        "",
    ])
    write_text_file(dst_root / "README.kaggle.txt", lines)


def prepare_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    *,
    yolo_id_from_mask: dict[int, int],
    min_area: float,
    epsilon_ratio: float,
    copy_images: bool,
) -> tuple[int, dict[int, int]]:
    src_split = src_root / split
    if not src_split.is_dir():
        raise FileNotFoundError(f"Split directory not found: {src_split}")

    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    instance_totals: dict[int, int] = {yolo_id: 0 for yolo_id in yolo_id_from_mask.values()}
    images = image_files_for_split(src_split)
    if not images:
        raise ValueError(f"No images found in {src_split}")

    for image_path in images:
        mask_path = mask_path_for_image(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image {image_path.name}: expected {mask_path.name}")

        hardlink_or_copy(image_path, dst_images / image_path.name, copy_images=copy_images)
        mask = load_mask(mask_path)
        lines, counts = mask_to_yolo_lines(
            mask,
            yolo_id_from_mask,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
        )
        write_text_file(dst_labels / f"{image_path.stem}.txt", lines)
        for class_id, count in counts.items():
            instance_totals[class_id] += count

    return len(images), instance_totals


def write_stats_yaml(dst_root: Path, stats: dict[str, dict[int, int] | int], names: list[str]) -> None:
    lines = [
        "prepared_dataset_stats:",
        f"  train_images: {stats['train_images']}",
        f"  valid_images: {stats['valid_images']}",
        "  train_instances_by_class:",
    ]
    for class_id, name in enumerate(names):
        lines.append(f"    {class_id}: {stats['train_instances_by_class'][class_id]}  # {name}")
    lines.append("  valid_instances_by_class:")
    for class_id, name in enumerate(names):
        lines.append(f"    {class_id}: {stats['valid_instances_by_class'][class_id]}  # {name}")
    write_text_file(dst_root / "dataset_stats.yaml", lines)


def main() -> None:
    args = parse_args()
    class_rows = load_class_rows(find_classes_csv(args.src))
    names, yolo_id_from_mask = build_foreground_mapping(class_rows)

    ensure_clean_dir(args.dst, args.clean)
    train_images, train_counts = prepare_split(
        args.src,
        args.dst,
        "train",
        yolo_id_from_mask=yolo_id_from_mask,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
        copy_images=args.copy_images,
    )
    valid_images, valid_counts = prepare_split(
        args.src,
        args.dst,
        "valid",
        yolo_id_from_mask=yolo_id_from_mask,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
        copy_images=args.copy_images,
    )

    write_data_yaml(args.dst, names)
    write_mapping_yaml(args.dst, class_rows, yolo_id_from_mask)
    write_readme(args.dst, names)
    write_stats_yaml(
        args.dst,
        {
            "train_images": train_images,
            "valid_images": valid_images,
            "train_instances_by_class": train_counts,
            "valid_instances_by_class": valid_counts,
        },
        names,
    )

    print(f"Prepared YOLO-seg dataset: {args.dst}")
    print(f"  train images: {train_images}")
    print(f"  valid images: {valid_images}")
    print(f"  classes     : {len(names)} foreground classes")
    for class_id, name in enumerate(names):
        print(
            f"    {class_id}: {name} | train_instances={train_counts[class_id]} | valid_instances={valid_counts[class_id]}"
        )


if __name__ == "__main__":
    main()
