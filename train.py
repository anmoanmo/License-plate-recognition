# -*- coding: utf-8 -*-
"""YOLO 车牌检测模型训练入口。"""

import argparse
import warnings

warnings.filterwarnings('ignore')

from project_utils import DATA_YAML_PATH, DATA_YAML_TEMPLATE_PATH
from yolo_runtime import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="训练 YOLO 车牌检测模型。")
    parser.add_argument("--model", default="yolo11n.pt", help="模型配置或权重路径，例如 yolo11n.pt。")
    parser.add_argument("--data", default=str(DATA_YAML_PATH), help="数据集配置文件路径。")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数。")
    parser.add_argument("--batch", type=int, default=4, help="批大小。")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸。")
    parser.add_argument("--device", default="", help="训练设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--project", default="runs/train", help="训练输出目录。")
    parser.add_argument("--name", default="exp", help="实验名称。")
    parser.add_argument("--weights", default="", help="可选的预训练权重路径，例如 yolo11n.pt。")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker数量，Windows建议0，AutoDL可设为8。")
    parser.add_argument("--optimizer", default="SGD", help="优化器，例如 SGD、AdamW 或 auto。")
    parser.add_argument("--close-mosaic", type=int, default=10, help="最后多少轮关闭mosaic增强。")
    parser.add_argument("--cache", action="store_true", help="缓存数据集以加速训练。")
    parser.add_argument("--resume", action="store_true", help="从上次训练中断处继续。")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATA_YAML_PATH.exists() and args.data == str(DATA_YAML_PATH):
        raise FileNotFoundError(
            f"data.yaml not found: {DATA_YAML_PATH}\n"
            f"Create it from the template first: {DATA_YAML_TEMPLATE_PATH}\n"
            "Or pass --data to specify your own dataset config."
        )

    model = YOLO(model=args.weights or args.model)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        project=args.project,
        name=args.name,
        single_cls=False,
        cache=args.cache,
    )


if __name__ == '__main__':
    main()
