# -*- coding: utf-8 -*-
"""车牌识别图形界面主入口。"""

import sys

import cv2
import numpy as np
import tensorflow as tf
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow
from tensorflow import keras
from ultralytics import YOLO

from ALPR_predict import cnn_predict
from plate_align import four_point_transform, refine_corners_with_contours
from project_utils import BEST_PT_PATH, CNN_MODEL_PATH, TEST_IMAGES_DIR, read_image
from ui.mainui import Ui_MainWindow


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class LPRWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.source = None
        self.roi_img = None
        self.warped_img = None
        self.model = YOLO(model=str(BEST_PT_PATH))
        self.cnn = keras.models.load_model(str(CNN_MODEL_PATH))
        self.model.predict(np.zeros((48, 48, 3), dtype=np.uint8), verbose=False)
        self.cnn.predict(np.zeros((1, 80, 240, 3), dtype=np.float32), verbose=0)
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
            results = self.model.predict(source=image, show=False, save=False, verbose=False)
            result = next(iter(results))

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

            refined_corners = refine_corners_with_contours(image, initial_corners, result, self.weight)
            self.roi_img, self.warped_img = four_point_transform(image, refined_corners)
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

        lpr_results = cnn_predict(self.cnn, self.warped_img)
        print(lpr_results)

        if not lpr_results:
            self.label_carNumber.setText("未识别到车牌")
        else:
            plate_text = lpr_results[0][1]
            self.label_carNumber.setText(plate_text)

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
