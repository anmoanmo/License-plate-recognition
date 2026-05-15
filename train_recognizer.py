# -*- coding: utf-8 -*-
"""Train the PyTorch plate character recognizer."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from plate_recognizer import (
    CHAR_TO_INDEX,
    CHARACTERS,
    INPUT_SIZE,
    NUM_CLASSES,
    PLATE_LENGTH,
    PlateRecognizerNet,
    load_torch_checkpoint,
    select_torch_device,
)
from project_utils import CCPD_NEW_DIR, RECOGNIZER_MODEL_PATH, TRAINING_RESULTS_DIR, read_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="训练PyTorch车牌字符识别模型。")
    parser.add_argument("--data-dir", default=str(CCPD_NEW_DIR), help="车牌图像目录，文件名前7位必须是车牌号。")
    parser.add_argument("--output-dir", default=str(TRAINING_RESULTS_DIR), help="训练输出目录。")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小。")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW权重衰减。")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例。")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker数量。")
    parser.add_argument("--device", default="", help="训练设备，例如 cuda、cuda:0、mps 或 cpu。留空自动选择。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--weights", default="", help="可选的已有PyTorch识别权重，用于继续训练。")
    parser.add_argument(
        "--augment-profile",
        choices=["standard", "light", "none"],
        default="standard",
        help="训练增强强度；light适合透视矫正后的字符精修。",
    )
    parser.add_argument("--balance-province", action="store_true", help="按第1位省份做均衡采样。")
    parser.add_argument("--province-loss-weight", type=float, default=1.0, help="第1位省份loss权重倍数。")
    parser.add_argument(
        "--province-weight-power",
        type=float,
        default=0.5,
        help="省份类别权重指数，0.5表示按样本数倒数开方。",
    )
    return parser.parse_args()


def collect_image_paths(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    paths = []
    skipped = 0
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = path.stem[:PLATE_LENGTH]
        if len(label) != PLATE_LENGTH or any(char not in CHAR_TO_INDEX for char in label):
            skipped += 1
            continue
        paths.append(path)

    if not paths:
        raise RuntimeError(f"未从 {data_dir} 找到可训练图片。文件名前7位需要是车牌字符。")
    if skipped:
        print(f"跳过 {skipped} 个文件名不符合标签规则的样本")
    return paths


def split_paths(paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    # 按省份分层拆分，避免少数省份只进入训练集或验证集。CCPD 的省份长尾会影响
    # 第 1 位字符评估，该逻辑用于减少拆分噪声，而不是解决数据集覆盖不足。
    rng = random.Random(seed)
    groups = defaultdict(list)
    for path in paths:
        groups[path.stem[0]].append(path)

    train_paths = []
    val_paths = []
    for _, province_paths in sorted(groups.items()):
        shuffled = province_paths[:]
        rng.shuffle(shuffled)
        if val_ratio <= 0 or len(shuffled) == 1:
            val_count = 0
        else:
            val_count = max(1, int(len(shuffled) * val_ratio))
        val_paths.extend(shuffled[:val_count])
        train_paths.extend(shuffled[val_count:])

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    if not val_paths and len(train_paths) > 1:
        val_paths.append(train_paths.pop())
    return train_paths, val_paths


def count_provinces(paths: list[Path]) -> Counter:
    return Counter(path.stem[0] for path in paths)


def print_province_distribution(name: str, paths: list[Path]) -> None:
    counts = count_provinces(paths)
    if not counts:
        return
    most_common = ", ".join(f"{char}:{count}" for char, count in counts.most_common(6))
    rare = ", ".join(f"{char}:{count}" for char, count in sorted(counts.items(), key=lambda item: item[1])[:6])
    print(f"{name}省份分布: {len(counts)}类，最多 {most_common}，最少 {rare}")


def build_province_sampler(paths: list[Path]) -> WeightedRandomSampler:
    # 省份均衡采样只用于实验诊断。当前数据中部分省份样本过少，过强均衡会损伤整体识别。
    counts = count_provinces(paths)
    sample_weights = [1.0 / counts[path.stem[0]] for path in paths]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_province_class_weights(paths: list[Path], power: float, device: torch.device) -> torch.Tensor:
    counts = count_provinces(paths)
    weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    if not counts:
        return weights

    max_count = max(counts.values())
    observed_indices = []
    for char, count in counts.items():
        index = CHAR_TO_INDEX[char]
        weights[index] = (max_count / max(count, 1)) ** power
        observed_indices.append(index)

    observed = torch.tensor(observed_indices, dtype=torch.long, device=device)
    weights[observed] = weights[observed] / torch.clamp(weights[observed].mean(), min=1e-6)

    province_weights = sorted(
        ((CHARACTERS[index], float(weights[index].item())) for index in observed_indices),
        key=lambda item: item[1],
        reverse=True,
    )
    preview = ", ".join(f"{char}:{weight:.2f}" for char, weight in province_weights[:8])
    print(f"省份loss类别权重预览: {preview}")
    return weights


class PlateDataset(Dataset):
    def __init__(self, paths: list[Path], augment_profile: str):
        self.paths = paths
        self.augment_profile = augment_profile

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = read_image(path)
        image = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))
        image = augment_plate_image(image, self.augment_profile)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        label = torch.tensor([CHAR_TO_INDEX[char] for char in path.stem[:PLATE_LENGTH]], dtype=torch.long)
        return tensor, label


def augment_plate_image(image: np.ndarray, profile: str) -> np.ndarray:
    if profile == "none":
        return image

    h, w = image.shape[:2]
    if profile == "light":
        # 透视矫正后的识别图不宜做强几何扰动，否则会放大 D/0、S/5、B/8 等形近字符混淆。
        if random.random() < 0.25:
            angle = random.uniform(-1.0, 1.0)
            scale = random.uniform(0.98, 1.02)
            tx = random.uniform(-0.01, 0.01) * w
            ty = random.uniform(-0.02, 0.02) * h
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            matrix[:, 2] += [tx, ty]
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        if random.random() < 0.5:
            alpha = random.uniform(0.90, 1.10)
            beta = random.uniform(-12, 12)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if random.random() < 0.08:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        if random.random() < 0.12:
            noise = np.random.normal(0, random.uniform(2, 5), image.shape).astype("float32")
            image = np.clip(image.astype("float32") + noise, 0, 255).astype("uint8")

        return image

    if random.random() < 0.8:
        angle = random.uniform(-3.0, 3.0)
        scale = random.uniform(0.94, 1.06)
        tx = random.uniform(-0.03, 0.03) * w
        ty = random.uniform(-0.06, 0.06) * h
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        matrix[:, 2] += [tx, ty]
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.7:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-25, 25)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.2:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if random.random() < 0.25:
        noise = np.random.normal(0, random.uniform(3, 10), image.shape).astype("float32")
        image = np.clip(image.astype("float32") + noise, 0, 255).astype("uint8")

    return image


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criteria: list[nn.Module],
    position_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = torch.stack([criteria[i](logits[:, i, :], labels[:, i]) for i in range(PLATE_LENGTH)])
    if position_weights is None:
        return losses.mean()
    return (losses * position_weights).sum() / torch.clamp(position_weights.sum(), min=1e-6)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criteria: list[nn.Module],
    position_weights: torch.Tensor | None = None,
):
    model.eval()
    total_loss = 0.0
    total_chars = 0
    correct_chars = 0
    total_plates = 0
    correct_plates = 0
    position_correct = torch.zeros(PLATE_LENGTH, dtype=torch.long)
    position_total = torch.zeros(PLATE_LENGTH, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = compute_loss(logits, labels, criteria, position_weights)
        total_loss += float(loss.item()) * images.shape[0]

        predicted = torch.argmax(logits, dim=-1)
        correct = predicted.eq(labels)
        correct_chars += int(correct.sum().item())
        total_chars += labels.numel()
        correct_plates += int(correct.all(dim=1).sum().item())
        total_plates += labels.shape[0]
        position_correct += correct.sum(dim=0).cpu()
        position_total += torch.full((PLATE_LENGTH,), labels.shape[0], dtype=torch.long)

    metrics = {
        "loss": total_loss / max(total_plates, 1),
        "char_acc": correct_chars / max(total_chars, 1),
        "plate_acc": correct_plates / max(total_plates, 1),
    }
    for index in range(PLATE_LENGTH):
        metrics[f"pos{index + 1}_acc"] = int(position_correct[index]) / max(int(position_total[index]), 1)
    return metrics


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "recognizer_training_log.csv"
    best_path = output_dir / RECOGNIZER_MODEL_PATH.name

    paths = collect_image_paths(data_dir)
    train_paths, val_paths = split_paths(paths, args.val_ratio, args.seed)
    print(f"训练样本: {len(train_paths)}，验证样本: {len(val_paths)}")
    print_province_distribution("训练集", train_paths)
    print_province_distribution("验证集", val_paths)

    device = select_torch_device(args.device or None)
    print(f"使用设备: {device}")
    print(f"训练增强策略: {args.augment_profile}")

    train_sampler = build_province_sampler(train_paths) if args.balance_province else None
    if train_sampler is not None:
        print("已启用省份均衡采样: --balance-province")

    train_loader = DataLoader(
        PlateDataset(train_paths, augment_profile=args.augment_profile),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PlateDataset(val_paths, augment_profile="none"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = PlateRecognizerNet().to(device)
    if args.weights:
        checkpoint = load_torch_checkpoint(args.weights, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
        print(f"已加载已有权重: {args.weights}")

    use_province_class_weights = args.balance_province or args.province_loss_weight != 1.0
    province_weights = (
        build_province_class_weights(train_paths, args.province_weight_power, device)
        if use_province_class_weights
        else None
    )
    criteria = [nn.CrossEntropyLoss(weight=province_weights)] + [
        nn.CrossEntropyLoss() for _ in range(PLATE_LENGTH - 1)
    ]
    position_weights = torch.ones(PLATE_LENGTH, dtype=torch.float32, device=device)
    position_weights[0] = max(args.province_loss_weight, 0.0)
    if args.province_loss_weight != 1.0:
        print(f"已启用第1位省份loss权重: {args.province_loss_weight}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "val_loss", "val_char_acc", "val_plate_acc",
                "val_pos1_acc", "val_pos2_acc", "val_pos3_acc", "val_pos4_acc",
                "val_pos5_acc", "val_pos6_acc", "val_pos7_acc", "lr", "time",
            ],
        )
        writer.writeheader()

    best_plate_acc = -1.0
    best_char_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = compute_loss(logits, labels, criteria, position_weights)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.shape[0]
            seen += images.shape[0]

        train_loss = running_loss / max(seen, 1)
        metrics = evaluate(model, val_loader, device, criteria, position_weights)
        scheduler.step(metrics["char_acc"])
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{metrics['loss']:.6f}",
            "val_char_acc": f"{metrics['char_acc']:.6f}",
            "val_plate_acc": f"{metrics['plate_acc']:.6f}",
            "val_pos1_acc": f"{metrics['pos1_acc']:.6f}",
            "val_pos2_acc": f"{metrics['pos2_acc']:.6f}",
            "val_pos3_acc": f"{metrics['pos3_acc']:.6f}",
            "val_pos4_acc": f"{metrics['pos4_acc']:.6f}",
            "val_pos5_acc": f"{metrics['pos5_acc']:.6f}",
            "val_pos6_acc": f"{metrics['pos6_acc']:.6f}",
            "val_pos7_acc": f"{metrics['pos7_acc']:.6f}",
            "lr": f"{lr:.8f}",
            "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"char_acc={metrics['char_acc']:.4f} plate_acc={metrics['plate_acc']:.4f} "
            f"pos1_acc={metrics['pos1_acc']:.4f} lr={lr:.6f}"
        )

        improved = (
            metrics["plate_acc"] > best_plate_acc
            or metrics["plate_acc"] == best_plate_acc
            and metrics["char_acc"] > best_char_acc
        )
        if improved:
            best_plate_acc = metrics["plate_acc"]
            best_char_acc = metrics["char_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_char_acc": best_char_acc,
                    "val_plate_acc": best_plate_acc,
                    "input_size": INPUT_SIZE,
                    "plate_length": PLATE_LENGTH,
                    "augment_profile": args.augment_profile,
                    "balance_province": args.balance_province,
                    "province_loss_weight": args.province_loss_weight,
                    "province_weight_power": args.province_weight_power,
                },
                best_path,
            )
            print(f"已保存最佳模型: {best_path}")

    final_path = output_dir / "final_recognizer.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": INPUT_SIZE,
            "plate_length": PLATE_LENGTH,
            "augment_profile": args.augment_profile,
            "balance_province": args.balance_province,
            "province_loss_weight": args.province_loss_weight,
            "province_weight_power": args.province_weight_power,
        },
        final_path,
    )
    print(f"训练完成，最终模型: {final_path}")
    print(f"最佳模型: {best_path}")


if __name__ == "__main__":
    train(parse_args())
