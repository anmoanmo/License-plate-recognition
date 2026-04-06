# -*- coding: utf-8 -*-
"""车牌检测命令行入口。"""

import argparse

from ultralytics import YOLO

from project_utils import BEST_PT_PATH, get_default_test_image


def parse_args():
    parser = argparse.ArgumentParser(description="使用本地权重进行车牌检测。")
    parser.add_argument(
        "--source",
        default=str(get_default_test_image()),
        help="待检测图片路径，默认使用仓库内真实存在的测试图片。",
    )
    parser.add_argument("--save", action="store_true", help="保存预测结果。")
    parser.add_argument("--save-crop", action="store_true", help="保存裁剪结果。")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(model=str(BEST_PT_PATH))
    results = model.predict(
        source=args.source,
        save=args.save,
        show=False,
        save_crop=args.save_crop,
        verbose=False,
    )
    result = next(iter(results))
    print(f"Detections: {len(result.boxes)}")
