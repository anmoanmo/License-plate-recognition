# -*- coding: utf-8 -*-
"""车牌识别图形界面主入口。"""

import sys

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow

from plate_align import (
    PLATE_TARGET_HEIGHT,
    PLATE_TARGET_WIDTH,
    build_plate_candidates,
    is_plate_like_image,
)
from plate_recognizer import load_plate_recognizer
from project_utils import BEST_PT_PATH, TEST_IMAGES_DIR, read_image
from ui.mainui import Ui_MainWindow
from yolo_runtime import YOLO


class LPRWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.source = None
        self.roi_img = None
        self.warped_img = None
        self.model = YOLO(model=str(BEST_PT_PATH))
        self.recognizer = load_plate_recognizer()
        self.model.predict(np.zeros((48, 48, 3), dtype=np.uint8), verbose=False)
        self.weight = 0.975
        self.yolo_init()

    def yolo_init(self):
        self.pushButton_UploadImg.clicked.connect(self.open_src_file)
        self.pushButton_DetectPlate.clicked.connect(self.img_predict)
        self.pushButton_RecognizePlate.clicked.connect(self.carNumber_predict)
        self.pushButton_Clear.clicked.connect(self.stop)

    def open_src_file(self):
        name, _ = QFileDialog.getOpenFileName(
            self,
            '选择图片',
            str(TEST_IMAGES_DIR),
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )

        if name:
            self.source = name
            print(f"Loaded file: {name}")
            self.stop(clear_source=False)
            self.show_image(name, self.label_input, 'path')

    @staticmethod
    def show_image(img, label, flag):
        try:
            if flag == "path":
                img_src = read_image(img)
            else:
                img_src = img

            if img_src is None:
                raise ValueError("无法显示空图像")

            if img_src.ndim == 2:
                img_src = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)

            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            label.setFixedSize(w, h)

            if iw / w > ih / h:
                scale = w / iw
                nw = w
                nh = int(scale * ih)
            else:
                scale = h / ih
                nw = int(scale * iw)
                nh = h

            img_src_ = cv2.resize(img_src, (nw, nh))
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            q_img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[2] * frame.shape[1],
                QImage.Format_RGB888
            )
            label.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print(repr(e))

    def img_predict(self):
        try:
            if not self.source:
                self.label_carNumber.setText("请先上传图片")
                return None, None, None, None

            image = read_image(self.source)
            whole_image_plate = is_plate_like_image(image)
            results = self.model.predict(source=image, show=False, save=False, verbose=False)
            result = next(iter(results))

            if whole_image_plate:
                if len(result.boxes) > 0:
                    self.show_image(result.plot(), self.label_input, 'img')
                    print("输入图像接近车牌比例，优先按整张车牌识别")
                else:
                    self.show_image(image, self.label_input, 'img')
                    print("YOLO未检测到车牌，输入图像接近车牌比例，按整张车牌识别")

                self.roi_img = cv2.resize(image, (PLATE_TARGET_WIDTH, PLATE_TARGET_HEIGHT))
                self.warped_img = [self.roi_img]
                self.show_image(self.roi_img, self.label_chepai, 'img')
                self.label_carNumber.setText("已按整图车牌处理，可开始识别")
                return None, None, image, result

            if len(result.boxes) == 0:
                print("未检测到车牌")
                self.warped_img = None
                self.label_chepai.clear()
                self.label_carNumber.setText("未检测到车牌")
                return None, None, None, None

            bbox = result.boxes.xyxy[0].tolist()
            print(f"检测到的边界框: {bbox}")

            x_min, y_min, x_max, y_max = bbox
            initial_corners = np.array(
                [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ],
                dtype="float32",
            )

            image_box = result.plot()
            self.show_image(image_box, self.label_input, 'img')

            self.roi_img, self.warped_img, _ = build_plate_candidates(
                image,
                initial_corners,
                result,
                self.weight,
            )
            self.show_image(self.roi_img, self.label_chepai, 'img')
            self.label_carNumber.setText("已完成定位，可开始识别")

            return bbox, initial_corners, image, result

        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            self.label_carNumber.setText("定位失败")
            return None, None, None, None

    def carNumber_predict(self):
        if not self.warped_img:
            self.label_carNumber.setText("请先完成车牌定位")
            return

        if hasattr(self.recognizer, "predict_with_confidence"):
            predictions = self.recognizer.predict_with_confidence(self.warped_img, topk=2)
            if not predictions:
                self.label_carNumber.setText("未识别到车牌")
                return

            result = max(predictions, key=lambda item: (item.avg_confidence, item.min_confidence))
            topk_summary = [
                "/".join(f"{char}:{confidence:.2f}" for char, confidence in char_options)
                for char_options in result.topk
            ]
            print(
                f"识别结果: {result.text}, avg={result.avg_confidence:.3f}, "
                f"min={result.min_confidence:.3f}, topk={topk_summary}"
            )

            self.roi_img = result.image
            self.show_image(self.roi_img, self.label_chepai, 'img')

            # 置信度只作为用户提示，不作为硬性纠错规则。当前实验表明，CCPD 数据集的省份
            # 分布偏置和 D/0、S/5、B/8 等形近字符会造成高置信误判，因此 GUI 只标注风险。
            if result.avg_confidence < 0.90 or result.min_confidence < 0.75:
                self.label_carNumber.setText(f"{result.text}（低置信度 {result.min_confidence:.2f}）")
            else:
                self.label_carNumber.setText(f"{result.text}（{result.avg_confidence:.2f}）")
            return

        lpr_results = self.recognizer.predict(self.warped_img)
        print(lpr_results)
        self.label_carNumber.setText(lpr_results[0][1] if lpr_results else "未识别到车牌")

    def stop(self, clear_source=True):
        self.label_input.clear()
        self.label_chepai.clear()
        self.label_carNumber.clear()
        self.roi_img = None
        self.warped_img = None
        if clear_source:
            self.source = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = LPRWindow()
    Home.show()
    sys.exit(app.exec())
