# -*- coding: utf-8 -*-
"""YOLO 车牌检测模型训练入口。"""

import argparse
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

from project_utils import DATA_YAML_PATH, DATA_YAML_TEMPLATE_PATH, YOLO_MODEL_CFG_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="训练 YOLO 车牌检测模型。")
    parser.add_argument("--model", default=str(YOLO_MODEL_CFG_PATH), help="模型配置文件路径。")
    parser.add_argument("--data", default=str(DATA_YAML_PATH), help="数据集配置文件路径。")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数。")
    parser.add_argument("--batch", type=int, default=4, help="批大小。")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸。")
    parser.add_argument("--device", default="", help="训练设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--project", default="runs/train", help="训练输出目录。")
    parser.add_argument("--name", default="exp", help="实验名称。")
    parser.add_argument("--weights", default="", help="可选的预训练权重路径。")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATA_YAML_PATH.exists() and args.data == str(DATA_YAML_PATH):
        raise FileNotFoundError(
            f"data.yaml not found: {DATA_YAML_PATH}\n"
            f"Create it from the template first: {DATA_YAML_TEMPLATE_PATH}\n"
            "Or pass --data to specify your own dataset config."
        )

    model = YOLO(model=args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=0,
        device=args.device,
        optimizer='SGD',
        close_mosaic=10,
        resume=False,
        project=args.project,
        name=args.name,
        single_cls=False,
        cache=False,
    )


if __name__ == '__main__':
    main()
