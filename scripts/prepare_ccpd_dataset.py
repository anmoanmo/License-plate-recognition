# -*- coding: utf-8 -*-
"""Convert CCPD images into YOLO detection and plate recognition datasets."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plate_recognizer import CHAR_TO_INDEX, PLATE_LENGTH
from project_utils import read_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
    "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
ADS = ALPHABETS + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def parse_args():
    parser = argparse.ArgumentParser(description="将CCPD原始图片转换为本项目训练目录。")
    parser.add_argument("--raw-dir", required=True, help="CCPD解压后的图片目录，可递归搜索。")
    parser.add_argument("--output-root", default="datasets/ccpd_prepared", help="输出根目录。")
    parser.add_argument("--max-images", type=int, default=0, help="最多转换多少张，0表示全部。")
    # CCPD 当前下载数据省份分布极不均衡。下面两个参数用于标注和复现实验中的数据漂移，
    # 不是项目主流程的强制步骤；最终演示应优先使用已经验证过的当前模型权重。
    parser.add_argument("--max-per-province", type=int, default=0, help="每个省份最多转换多少张，0表示不限。")
    parser.add_argument("--min-per-province", type=int, default=0, help="省份总样本少于该值时排除，0表示不排除。")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="YOLO验证集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--rec-width", type=int, default=240, help="识别图透视矫正后的宽度。")
    parser.add_argument("--rec-height", type=int, default=80, help="识别图透视矫正后的高度。")
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help="YOLO图片写入方式；AutoDL同盘建议hardlink节省空间。",
    )
    return parser.parse_args()


def print_distribution(title: str, counts: Counter, limit: int = 12) -> None:
    if not counts:
        print(f"{title}: 空")
        return
    total = sum(counts.values())
    most_common = ", ".join(f"{char}:{count}" for char, count in counts.most_common(limit))
    rare = ", ".join(f"{char}:{count}" for char, count in sorted(counts.items(), key=lambda item: item[1])[:limit])
    print(f"{title}: total={total}, classes={len(counts)}, 最多 {most_common}, 最少 {rare}")


def parse_point(text: str) -> tuple[int, int]:
    x_text, y_text = text.split("&")
    return int(x_text), int(y_text)


def parse_ccpd_name(path: Path):
    parts = path.stem.split("-")
    if len(parts) < 5:
        raise ValueError("filename does not match CCPD pattern")

    left_top, right_bottom = [parse_point(item) for item in parts[2].split("_")]
    corners = [parse_point(item) for item in parts[3].split("_")]
    if len(corners) != 4:
        raise ValueError("filename does not contain 4 plate corners")

    plate_ids = [int(item) for item in parts[4].split("_")]
    if len(plate_ids) != PLATE_LENGTH:
        raise ValueError("plate code is not 7 characters")

    chars = [PROVINCES[plate_ids[0]], ALPHABETS[plate_ids[1]]]
    chars.extend(ADS[index] for index in plate_ids[2:])
    plate_text = "".join(chars)
    if any(char not in CHAR_TO_INDEX for char in plate_text):
        raise ValueError(f"unsupported plate character: {plate_text}")

    x1, y1 = left_top
    x2, y2 = right_bottom
    return plate_text, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), corners


def select_image_paths(
    paths: list[Path],
    max_images: int,
    max_per_province: int,
    min_per_province: int,
    seed: int,
) -> list[Path]:
    # 先按文件名解析省份，再做上限/下限过滤。这样可以明确看到 CCPD 原始分布带来的
    # 长尾问题：有些省份只有个位数样本，强行纳入会导致首位省份识别明显漂移。
    rng = random.Random(seed)
    groups = defaultdict(list)
    invalid = 0
    for path in paths:
        try:
            plate_text, _, _ = parse_ccpd_name(path)
        except Exception:
            invalid += 1
            continue
        groups[plate_text[0]].append(path)

    raw_counts = Counter({province: len(province_paths) for province, province_paths in groups.items()})
    print_distribution("原始可解析省份分布", raw_counts)

    excluded_counts = Counter()
    capped = 0
    candidates_by_province = {}
    for province, province_paths in sorted(groups.items()):
        if min_per_province > 0 and len(province_paths) < min_per_province:
            excluded_counts[province] = len(province_paths)
            continue

        shuffled = province_paths[:]
        rng.shuffle(shuffled)
        if max_per_province > 0 and len(shuffled) > max_per_province:
            capped += len(shuffled) - max_per_province
            shuffled = shuffled[:max_per_province]
        candidates_by_province[province] = shuffled

    selected = []
    if max_images <= 0:
        for province_paths in candidates_by_province.values():
            selected.extend(province_paths)
    else:
        province_order = list(candidates_by_province)
        while len(selected) < max_images and province_order:
            rng.shuffle(province_order)
            next_order = []
            for province in province_order:
                province_paths = candidates_by_province[province]
                if not province_paths:
                    continue
                selected.append(province_paths.pop())
                if len(selected) >= max_images:
                    break
                if province_paths:
                    next_order.append(province)
            province_order = next_order

    rng.shuffle(selected)
    province_counts = Counter()
    for path in selected:
        plate_text, _, _ = parse_ccpd_name(path)
        province_counts[plate_text[0]] += 1

    if invalid:
        print(f"抽样阶段跳过无效文件名: {invalid}")
    if excluded_counts:
        print_distribution("按最小省份样本数排除", excluded_counts)
    if capped:
        print(f"按省份限额跳过: {capped}")
    print_distribution("抽样省份分布", province_counts)
    return selected


def split_validation_set(paths: list[Path], val_ratio: float, seed: int) -> set[Path]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for path in paths:
        plate_text, _, _ = parse_ccpd_name(path)
        groups[plate_text[0]].append(path)

    val_set = set()
    train_counts = Counter()
    val_counts = Counter()
    for province, province_paths in sorted(groups.items()):
        shuffled = province_paths[:]
        rng.shuffle(shuffled)
        if val_ratio <= 0 or len(shuffled) == 1:
            val_count = 0
        else:
            val_count = max(1, int(len(shuffled) * val_ratio))

        province_val = shuffled[:val_count]
        val_set.update(province_val)
        val_counts[province] += len(province_val)
        train_counts[province] += len(shuffled) - len(province_val)

    if not val_set and len(paths) > 1:
        val_set.add(paths[0])
        plate_text, _, _ = parse_ccpd_name(paths[0])
        val_counts[plate_text[0]] += 1
        train_counts[plate_text[0]] -= 1

    print_distribution("YOLO训练集省份分布", train_counts)
    print_distribution("YOLO验证集省份分布", val_counts)
    return val_set


def copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        try:
            os.link(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_yolo_label(label_path: Path, bbox: tuple[int, int, int, int], image_width: int, image_height: int) -> None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image_width - 1, x1))
    x2 = max(0, min(image_width - 1, x2))
    y1 = max(0, min(image_height - 1, y1))
    y2 = max(0, min(image_height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid bbox")

    cx = ((x1 + x2) / 2) / image_width
    cy = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n", encoding="utf-8")


def order_points(points: list[tuple[int, int]]) -> np.ndarray:
    pts = np.asarray(points, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]   # top-left
    rect[2] = pts[np.argmax(sums)]   # bottom-right
    rect[1] = pts[np.argmin(diffs)]  # top-right
    rect[3] = pts[np.argmax(diffs)]  # bottom-left
    return rect


def save_recognition_warp(
    image,
    corners: list[tuple[int, int]],
    output_path: Path,
    target_width: int,
    target_height: int,
) -> None:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("invalid recognition target size")

    src = order_points(corners)
    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if warped.size == 0:
        raise ValueError("empty warped plate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", warped)
    if not ok:
        raise ValueError("failed to encode crop")
    encoded.tofile(str(output_path))


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)
    yolo_root = output_root / "plate_yolo_dataset"
    recognition_root = output_root / "plate_recognition_dataset"

    image_paths = [path for path in raw_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise RuntimeError(f"未找到图片: {raw_dir}")

    image_paths = select_image_paths(
        image_paths,
        args.max_images,
        args.max_per_province,
        args.min_per_province,
        args.seed,
    )
    if not image_paths:
        raise RuntimeError("抽样后没有可转换图片")

    val_set = split_validation_set(image_paths, args.val_ratio, args.seed)
    converted = 0
    skipped = 0

    for index, path in enumerate(image_paths):
        try:
            plate_text, bbox, corners = parse_ccpd_name(path)
            image = read_image(path)
            height, width = image.shape[:2]
            split = "val" if path in val_set else "train"
            image_name = f"{index:07d}_{path.name}"
            yolo_image_path = yolo_root / "images" / split / image_name
            yolo_label_path = yolo_root / "labels" / split / f"{Path(image_name).stem}.txt"

            copy_image(path, yolo_image_path, args.copy_mode)
            write_yolo_label(yolo_label_path, bbox, width, height)
            save_recognition_warp(
                image,
                corners,
                recognition_root / f"{plate_text}_{index:07d}.jpg",
                args.rec_width,
                args.rec_height,
            )
            converted += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"跳过 {path}: {exc}")

    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        "train: images/train\nval: images/val\n\nnames:\n  0: plate\n",
        encoding="utf-8",
    )
    print(f"转换完成: {converted} 张，跳过: {skipped}")
    print(f"YOLO数据集: {yolo_root}")
    print(f"识别数据集: {recognition_root}")


if __name__ == "__main__":
    main()
