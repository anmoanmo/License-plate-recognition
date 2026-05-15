# -*- coding: utf-8 -*-
"""Evaluate trained plate recognizer samples with confidence diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plate_recognizer import PLATE_LENGTH, TorchPlateRecognizer
from project_utils import RECOGNIZER_MODEL_PATH, read_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="抽样评估PyTorch车牌字符识别模型。")
    parser.add_argument("--weights", default=str(RECOGNIZER_MODEL_PATH), help="PyTorch识别权重路径。")
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "datasets" / "ccpd_prepared" / "plate_recognition_dataset"),
        help="识别数据集目录，文件名前7位必须是车牌号。",
    )
    parser.add_argument("--limit", type=int, default=50, help="最多评估多少张；0表示全量。")
    parser.add_argument("--batch-size", type=int, default=256, help="推理批大小。")
    parser.add_argument("--device", default="", help="推理设备，例如 cuda、cuda:0、mps 或 cpu。留空自动选择。")
    parser.add_argument("--topk", type=int, default=2, help="每个字符输出Top-K候选。")
    parser.add_argument("--min-conf", type=float, default=0.45, help="低置信度字符阈值。")
    parser.add_argument("--print-limit", type=int, default=50, help="最多打印多少行样本明细。")
    parser.add_argument("--confusion-limit", type=int, default=20, help="最多打印多少个省份混淆项。")
    parser.add_argument("--char-confusion-limit", type=int, default=30, help="最多打印多少个字符混淆项。")
    parser.add_argument("--position-confusion-limit", type=int, default=8, help="每个位置最多打印多少个字符混淆项。")
    parser.add_argument("--show-errors-only", action="store_true", help="只打印识别错误样本。")
    parser.add_argument("--output-csv", default="", help="可选CSV输出路径。")
    return parser.parse_args()


def collect_paths(data_dir: Path, limit: int) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    paths = [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS and len(path.stem[:PLATE_LENGTH]) == PLATE_LENGTH
    ]
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"未在 {data_dir} 找到可评估图片")
    return paths


def batched(items: list[Path], batch_size: int):
    batch_size = max(1, batch_size)
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def topk_to_text(topk) -> str:
    positions = []
    for index, char_options in enumerate(topk, start=1):
        options = "/".join(f"{char}:{confidence:.2f}" for char, confidence in char_options)
        positions.append(f"{index}[{options}]")
    return " ".join(positions)


def evaluate(args) -> None:
    data_dir = Path(args.data_dir)
    weights = Path(args.weights)
    paths = collect_paths(data_dir, args.limit)

    recognizer = TorchPlateRecognizer(weights, device=args.device or None)
    recognizer.warmup()

    rows = []
    total_chars = 0
    correct_chars = 0
    correct_plates = 0
    low_conf_wrong = 0
    print_count = 0
    position_correct = [0] * PLATE_LENGTH
    position_total = [0] * PLATE_LENGTH
    position_errors = Counter()
    province_confusions = Counter()
    char_confusions = Counter()
    position_char_confusions = [Counter() for _ in range(PLATE_LENGTH)]

    # 该脚本用于标注误差来源。当前项目已确认 CCPD 会带来省份长尾和形近字符混淆，
    # 因此这里保留详细混淆统计，方便在报告中解释“数据集漂移”，而不是盲目继续调模型。
    for path_batch in batched(paths, args.batch_size):
        images = [read_image(path) for path in path_batch]
        predictions = recognizer.predict_with_confidence(images, topk=args.topk)

        for path, prediction in zip(path_batch, predictions):
            expected = path.stem[:PLATE_LENGTH]
            predicted = prediction.raw_text
            char_matches = [left == right for left, right in zip(expected, predicted)]
            plate_ok = all(char_matches)
            low_conf = prediction.min_confidence < args.min_conf
            wrong_positions = [
                str(index + 1)
                for index, matched in enumerate(char_matches)
                if not matched
            ]
            wrong_pairs = [
                f"{index + 1}:{expected_char}->{predicted_char}"
                for index, (expected_char, predicted_char, matched) in enumerate(
                    zip(expected, predicted, char_matches)
                )
                if not matched
            ]

            total_chars += PLATE_LENGTH
            correct_chars += sum(char_matches)
            correct_plates += int(plate_ok)
            low_conf_wrong += int((not plate_ok) and low_conf)
            for index, matched in enumerate(char_matches):
                position_total[index] += 1
                position_correct[index] += int(matched)
                if not matched:
                    position_errors[index + 1] += 1
                    expected_char = expected[index]
                    predicted_char = predicted[index]
                    char_confusions[(expected_char, predicted_char)] += 1
                    position_char_confusions[index][(expected_char, predicted_char)] += 1
            if expected[0] != predicted[0]:
                province_confusions[(expected[0], predicted[0])] += 1

            row = {
                "path": str(path),
                "expected": expected,
                "predicted": predicted,
                "formatted": prediction.text,
                "plate_ok": plate_ok,
                "wrong_positions": " ".join(wrong_positions),
                "wrong_pairs": " ".join(wrong_pairs),
                "first_wrong_position": wrong_positions[0] if wrong_positions else "",
                "avg_confidence": f"{prediction.avg_confidence:.6f}",
                "min_confidence": f"{prediction.min_confidence:.6f}",
                "char_confidences": " ".join(f"{value:.3f}" for value in prediction.char_confidences),
                "topk": topk_to_text(prediction.topk),
            }
            rows.append(row)

            should_print = not args.show_errors_only or not plate_ok
            if should_print and print_count < args.print_limit:
                print(
                    f"true={expected} pred={prediction.text} ok={plate_ok} "
                    f"avg={prediction.avg_confidence:.3f} min={prediction.min_confidence:.3f} "
                    f"chars={row['char_confidences']} topk={row['topk']}"
                )
                print_count += 1

    total = len(rows)
    plate_acc = correct_plates / max(total, 1)
    char_acc = correct_chars / max(total_chars, 1)
    avg_conf = mean(float(row["avg_confidence"]) for row in rows)
    avg_min_conf = mean(float(row["min_confidence"]) for row in rows)

    print(
        f"汇总: samples={total} char_acc={char_acc:.4f} plate_acc={plate_acc:.4f} "
        f"avg_conf={avg_conf:.4f} avg_min_conf={avg_min_conf:.4f} "
        f"wrong_low_conf={low_conf_wrong}/{total - correct_plates}"
    )
    position_text = " ".join(
        f"pos{index + 1}_acc={position_correct[index] / max(position_total[index], 1):.4f}"
        for index in range(PLATE_LENGTH)
    )
    print(f"位置准确率: {position_text}")

    if position_errors:
        error_text = ", ".join(f"pos{position}:{count}" for position, count in position_errors.most_common())
        print(f"错误位置统计: {error_text}")

    if province_confusions:
        print("省份混淆Top:")
        for (expected, predicted), count in province_confusions.most_common(args.confusion_limit):
            print(f"  {expected}->{predicted}: {count}")

    if char_confusions:
        print("字符混淆Top:")
        for (expected, predicted), count in char_confusions.most_common(args.char_confusion_limit):
            print(f"  {expected}->{predicted}: {count}")

    if any(position_char_confusions):
        print("按位置字符混淆Top:")
        for index, counter in enumerate(position_char_confusions, start=1):
            if not counter:
                continue
            items = ", ".join(
                f"{expected}->{predicted}:{count}"
                for (expected, predicted), count in counter.most_common(args.position_confusion_limit)
            )
            print(f"  pos{index}: {items}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"已写出CSV: {output_path}")


if __name__ == "__main__":
    evaluate(parse_args())
