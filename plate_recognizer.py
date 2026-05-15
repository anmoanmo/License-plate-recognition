# -*- coding: utf-8 -*-
"""PyTorch plate character recognizer runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn

from project_utils import CNN_MODEL_PATH, RECOGNIZER_MODEL_PATH


CHARACTERS = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
    "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
    "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
    "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARACTERS)}
# 当前项目按普通 7 位蓝牌整理，新能源 8 位车牌不纳入本次交付范围。
PLATE_LENGTH = 7
NUM_CLASSES = len(CHARACTERS)
INPUT_SIZE = (80, 240)


@dataclass(frozen=True)
class PlatePrediction:
    """Structured recognizer output for confidence diagnostics."""

    image: np.ndarray
    text: str
    raw_text: str
    avg_confidence: float
    min_confidence: float
    char_confidences: list[float]
    topk: list[list[tuple[str, float]]]


def select_torch_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PlateRecognizerNet(nn.Module):
    """A compact 7-head recognizer represented as [batch, 7, 65] logits.

    该结构适合作为课程/项目展示的轻量识别器。CCPD 长尾省份和形近字符漂移主要是
    数据覆盖问题，不在运行时通过硬编码规则修正，避免把数据集偏置写死到产品逻辑里。
    """

    def __init__(self, num_classes: int = NUM_CLASSES, plate_length: int = PLATE_LENGTH):
        super().__init__()
        self.num_classes = num_classes
        self.plate_length = plate_length
        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, 128),
            self._block(128, 128),
            nn.MaxPool2d(2),
            self._block(128, 256),
            self._block(256, 256),
            nn.AdaptiveAvgPool2d((1, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, plate_length * num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.shape[0], self.plate_length, self.num_classes)


def preprocess_plate_image(image: np.ndarray) -> torch.Tensor:
    if image is None or image.size == 0:
        raise ValueError("empty plate image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype("float32") / 255.0
    return torch.from_numpy(normalized.transpose(2, 0, 1))


def format_plate_text(indices: Iterable[int]) -> str:
    chars = format_plate_raw(indices)
    return chars[:2] + "·" + chars[2:]


def format_plate_raw(indices: Iterable[int]) -> str:
    return "".join(CHARACTERS[idx] for idx in indices)


def load_torch_checkpoint(path: str | Path, map_location):
    """Load checkpoints on both new PyTorch and AutoDL's common PyTorch 2.0 image."""

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _softmax_numpy(values: np.ndarray) -> np.ndarray:
    values = values.astype("float32")
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def _ensure_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype="float32").reshape(PLATE_LENGTH, NUM_CLASSES)
    row_sums = values.sum(axis=-1)
    if np.all(values >= 0) and np.allclose(row_sums, 1.0, atol=1e-3):
        return values
    return _softmax_numpy(values)


def _prediction_from_probabilities(
    image: np.ndarray,
    probabilities: np.ndarray,
    topk: int = 3,
) -> PlatePrediction:
    probabilities = _ensure_probabilities(probabilities)
    topk = max(1, min(int(topk), NUM_CLASSES))
    indices = np.argmax(probabilities, axis=-1)
    confidences = probabilities[np.arange(PLATE_LENGTH), indices]

    topk_indices = np.argsort(probabilities, axis=-1)[:, -topk:][:, ::-1]
    topk_values = np.take_along_axis(probabilities, topk_indices, axis=-1)
    topk_chars = [
        [
            (CHARACTERS[int(char_index)], float(confidence))
            for char_index, confidence in zip(char_indices, char_confidences)
        ]
        for char_indices, char_confidences in zip(topk_indices, topk_values)
    ]

    raw_text = format_plate_raw(indices.tolist())
    return PlatePrediction(
        image=image,
        text=raw_text[:2] + "·" + raw_text[2:],
        raw_text=raw_text,
        avg_confidence=float(np.mean(confidences)),
        min_confidence=float(np.min(confidences)),
        char_confidences=[float(value) for value in confidences],
        topk=topk_chars,
    )


class TorchPlateRecognizer:
    def __init__(self, weights_path: str | Path = RECOGNIZER_MODEL_PATH, device: str | None = None):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"PyTorch识别权重不存在: {self.weights_path}")

        self.device = select_torch_device(device)
        checkpoint = load_torch_checkpoint(self.weights_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model = PlateRecognizerNet().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def warmup(self) -> None:
        dummy = torch.zeros((1, 3, INPUT_SIZE[0], INPUT_SIZE[1]), dtype=torch.float32, device=self.device)
        self.model(dummy)

    @torch.inference_mode()
    def predict(
        self,
        plate_images: Iterable[np.ndarray],
        min_position_confidence: float = 0.8,
        min_confident_positions: int = 4,
    ) -> list[tuple[np.ndarray, str]]:
        predictions = []
        for result in self.predict_with_confidence(plate_images, topk=1):
            if int((np.array(result.char_confidences) >= min_position_confidence).sum()) < min_confident_positions:
                continue
            predictions.append((result.image, result.text))
        return predictions

    @torch.inference_mode()
    def predict_with_confidence(self, plate_images: Iterable[np.ndarray], topk: int = 3) -> list[PlatePrediction]:
        images = [image for image in plate_images if image is not None and image.size > 0]
        if not images:
            return []

        batch = torch.stack([preprocess_plate_image(image) for image in images]).to(self.device)
        probabilities = torch.softmax(self.model(batch), dim=-1).cpu().numpy()
        return [
            _prediction_from_probabilities(image, probability_row, topk=topk)
            for image, probability_row in zip(images, probabilities)
        ]


class LegacyKerasPlateRecognizer:
    """Compatibility wrapper for the existing training_results/best_model.h5 model."""

    def __init__(self, weights_path: str | Path = CNN_MODEL_PATH):
        from tensorflow import keras

        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Keras识别权重不存在: {self.weights_path}")
        self.model = keras.models.load_model(str(self.weights_path))

    def warmup(self) -> None:
        self.model.predict(np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32), verbose=0)

    def predict(
        self,
        plate_images: Iterable[np.ndarray],
        min_position_confidence: float = 0.8,
        min_confident_positions: int = 4,
    ) -> list[tuple[np.ndarray, str]]:
        predictions = []
        for result in self.predict_with_confidence(plate_images, topk=1):
            if int((np.array(result.char_confidences) >= min_position_confidence).sum()) < min_confident_positions:
                continue
            predictions.append((result.image, result.text))
        return predictions

    def predict_with_confidence(self, plate_images: Iterable[np.ndarray], topk: int = 3) -> list[PlatePrediction]:
        predictions = []
        for image in plate_images:
            if image is None or image.size == 0:
                continue
            resized = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0])).astype("float32") / 255.0
            output = self.model.predict(resized.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 3), verbose=0)
            predictions.append(_prediction_from_probabilities(image, np.array(output), topk=topk))
        return predictions


def load_plate_recognizer(
    weights_path: str | Path = RECOGNIZER_MODEL_PATH,
    legacy_weights_path: str | Path = CNN_MODEL_PATH,
    device: str | None = None,
):
    weights_path = Path(weights_path)
    if weights_path.exists():
        print(f"使用PyTorch识别模型: {weights_path}")
        recognizer = TorchPlateRecognizer(weights_path, device=device)
    else:
        print(f"使用Keras识别模型: {legacy_weights_path}")
        recognizer = LegacyKerasPlateRecognizer(legacy_weights_path)

    recognizer.warmup()
    return recognizer
