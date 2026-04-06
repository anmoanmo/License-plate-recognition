"""项目公共路径与图像读取工具。"""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
UI_DIR = PROJECT_ROOT / "ui"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
TRAINING_RESULTS_DIR = PROJECT_ROOT / "training_results"
CCPD_DIR = PROJECT_ROOT / "ccpd"
CCPD_NEW_DIR = CCPD_DIR / "new_ccpd"
BEST_PT_PATH = PROJECT_ROOT / "best.pt"
CNN_MODEL_PATH = TRAINING_RESULTS_DIR / "best_model.h5"
YOLO_MODEL_CFG_PATH = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml"
DATA_YAML_PATH = PROJECT_ROOT / "data.yaml"
DATA_YAML_TEMPLATE_PATH = PROJECT_ROOT / "data.yaml.example"
DEFAULT_TEST_IMAGE_CANDIDATES = [
    TEST_IMAGES_DIR / "3.jpg",
    TEST_IMAGES_DIR / "2.jpg",
    TEST_IMAGES_DIR / "4.jpg",
    TEST_IMAGES_DIR / "5.jpg",
    TEST_IMAGES_DIR / "7.jpg",
]


def resolve_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_path_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description}不存在: {path}")
    return path


def read_image(image_path) -> np.ndarray:
    path = Path(image_path)
    ensure_path_exists(path, "图像文件")
    image_data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def get_default_test_image() -> Path:
    for candidate in DEFAULT_TEST_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate

    for candidate in sorted(TEST_IMAGES_DIR.glob("*")):
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return candidate

    raise FileNotFoundError(f"未在测试目录中找到可用图片: {TEST_IMAGES_DIR}")
