# -*- coding: utf-8 -*-
"""Build a tiny local dataset package from existing test images.

This is a structural smoke-test dataset, not a replacement for CCPD training.
YOLO labels are generated from the current best.pt model as pseudo labels.
Recognition crops are created only when true labels are supplied with CSV.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plate_recognizer import CHAR_TO_INDEX, PLATE_LENGTH
from project_utils import BEST_PT_PATH, TEST_IMAGES_DIR, read_image
from yolo_runtime import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="基于本机测试图生成轻量训练目录样例。")
    parser.add_argument("--source-dir", default=str(TEST_IMAGES_DIR), help="输入图片目录。")
    parser.add_argument("--output-root", default="datasets/local_light", help="输出根目录。")
    parser.add_argument("--model", default=str(BEST_PT_PATH), help="用于生成检测伪标签的YOLO权重。")
    parser.add_argument("--conf", type=float, default=0.25, help="检测伪标签置信度阈值。")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="YOLO验证集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--labels-csv",
        default="",
        help="可选CSV，列为 filename,plate。提供后会生成识别裁剪图。",
    )
    return parser.parse_args()


def load_plate_labels(csv_path: str) -> dict[str, str]:
    if not csv_path:
        return {}
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            plate = (row.get("plate") or "").replace("·", "").strip()
            if filename and is_supported_plate(plate):
                labels[filename] = plate
    return labels


def is_supported_plate(plate: str) -> bool:
    return len(plate) == PLATE_LENGTH and all(char in CHAR_TO_INDEX for char in plate)


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def write_yolo_label(path: Path, box, width: int, height: int) -> None:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"0 {cx:.6f} {cy:.6f} {box_width:.6f} {box_height:.6f}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)
    yolo_root = output_root / "plate_yolo_dataset"
    recognition_root = output_root / "plate_recognition_dataset"
    labels = load_plate_labels(args.labels_csv)

    if output_root.exists():
        shutil.rmtree(output_root)
    (yolo_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    recognition_root.mkdir(parents=True, exist_ok=True)

    image_paths = [path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise RuntimeError(f"未找到测试图片: {source_dir}")
    random.Random(args.seed).shuffle(image_paths)

    val_count = max(1, int(len(image_paths) * args.val_ratio))
    val_set = set(image_paths[:val_count])
    model = YOLO(model=args.model)

    detection_count = 0
    recognition_count = 0
    skipped = 0

    for index, path in enumerate(image_paths):
        image = read_image(path)
        height, width = image.shape[:2]
        result = model.predict(source=image, show=False, save=False, verbose=False, conf=args.conf)[0]
        if len(result.boxes) == 0:
            skipped += 1
            print(f"跳过无检测图片: {path}")
            continue

        box = result.boxes.xyxy[0].cpu().numpy().tolist()
        split = "val" if path in val_set else "train"
        image_name = f"{index:04d}_{path.name}"
        shutil.copy2(path, yolo_root / "images" / split / image_name)
        write_yolo_label(yolo_root / "labels" / split / f"{Path(image_name).stem}.txt", box, width, height)
        detection_count += 1

        plate = labels.get(path.name)
        if plate:
            x1, y1, x2, y2 = [int(value) for value in box]
            crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if crop.size:
                write_image(recognition_root / f"{plate}_{index:04d}.jpg", crop)
                recognition_count += 1

    (yolo_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\n\nnames:\n  0: plate\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        "# Local Light Dataset\n\n"
        "This directory is a tiny structural sample generated from local test images.\n"
        "YOLO labels are pseudo labels from the current model and should not be used as final training truth.\n"
        "Recognition crops are generated only when `--labels-csv` supplies true plate numbers.\n",
        encoding="utf-8",
    )
    print(f"轻量样例生成完成: detections={detection_count}, recognition={recognition_count}, skipped={skipped}")
    print(f"输出目录: {output_root}")


if __name__ == "__main__":
    main()
