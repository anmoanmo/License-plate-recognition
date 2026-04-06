# -*- coding: utf-8 -*-
"""车牌字符识别 CNN 训练入口。"""

import argparse
import datetime
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

from project_utils import CCPD_NEW_DIR, TRAINING_RESULTS_DIR, read_image


plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="训练车牌字符识别 CNN。")
    parser.add_argument("--epochs", type=int, default=35, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小。")
    parser.add_argument("--data-dir", default=str(CCPD_NEW_DIR), help="车牌字符数据集目录。")
    parser.add_argument(
        "--output-dir",
        default=str(TRAINING_RESULTS_DIR),
        help="训练输出目录，默认与 GUI 读取目录保持一致。",
    )
    return parser.parse_args()


def train(epochs, batch_size, data_dir, output_dir):
    char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                 "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                 "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
                 "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
                 "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
                 "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
                 "W": 61, "X": 62, "Y": 63, "Z": 64}

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    log_file = output_dir / "training_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"训练开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"数据集目录: {data_dir}\n")

    pic_names = sorted([p.name for p in data_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    n = len(pic_names)

    X = []
    y = []

    print(f"开始读取 {n} 张图片...")
    for i, name in enumerate(pic_names):
        if i % 500 == 0 and i > 0:
            print(f"已读取 {i}/{n} 张图片")

        try:
            img_path = data_dir / name
            img = read_image(img_path)

            if img.shape != (80, 240, 3):
                img = cv2.resize(img, (240, 80))
            img = img.astype("float32") / 255.0

            plate_name = Path(name).stem
            label = [char_dict[char] for char in plate_name[:7]]

            if len(label) == 7:
                X.append(img)
                y.append(label)
        except Exception as e:
            print(f"处理图片 {name} 时出错: {str(e)}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"处理图片 {name} 时出错: {str(e)}\n")

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise RuntimeError(f"未从 {data_dir} 读取到可训练的样本。")

    print(f"成功读取 {len(X)} 张图片")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"成功读取 {len(X)} 张图片\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_list = [y_train[:, i] for i in range(7)]
    y_val_list = [y_val[:, i] for i in range(7)]

    model = build_model()
    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.h5'),
            monitor='val_c1_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.CSVLogger(str(output_dir / 'training_log.csv')),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    print("开始训练模型...")
    history = model.fit(
        X_train, y_train_list,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val_list),
        callbacks=callbacks_list,
        verbose=1
    )

    model.save(output_dir / 'final_model.h5')
    print(f"最终模型已保存到 {output_dir / 'final_model.h5'}")

    plot_training_history(history, output_dir)
    analyze_training_results(history, output_dir, X_val, y_val, char_dict)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"训练结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("训练完成!\n")


def build_model():
    model_input = layers.Input((80, 240, 3))
    x = model_input
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
    for i in range(3):
        x = layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
        x = layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
        x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    outputs = [layers.Dense(65, activation='softmax', name='c%d' % (i + 1))(x) for i in range(7)]
    model = models.Model(inputs=model_input, outputs=outputs)
    model.summary()
    return model


def plot_training_history(history, output_dir):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('周期')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history.history['c1_accuracy'], label='训练准确率')
    plt.plot(history.history['val_c1_accuracy'], label='验证准确率')
    plt.title('第一个字符准确率')
    plt.ylabel('准确率')
    plt.xlabel('周期')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history.history['c7_accuracy'], label='训练准确率')
    plt.plot(history.history['val_c7_accuracy'], label='验证准确率')
    plt.title('第七个字符准确率')
    plt.ylabel('准确率')
    plt.xlabel('周期')
    plt.legend()
    plt.grid(True)

    train_acc = np.mean([history.history[f'c{i}_accuracy'] for i in range(1, 8)], axis=0)
    val_acc = np.mean([history.history[f'val_c{i}_accuracy'] for i in range(1, 8)], axis=0)

    plt.subplot(2, 2, 4)
    plt.plot(train_acc, label='训练准确率')
    plt.plot(val_acc, label='验证准确率')
    plt.title('平均字符准确率')
    plt.ylabel('准确率')
    plt.xlabel('周期')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()


def analyze_training_results(history, output_dir, X_val, y_val, char_dict):
    reverse_char_dict = {v: k for k, v in char_dict.items()}

    metrics_file = output_dir / "training_metrics.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        best_val_acc = max(history.history['val_c1_accuracy'])
        best_epoch = np.argmax(history.history['val_c1_accuracy']) + 1

        f.write(f"最佳验证准确率 (第一个字符): {best_val_acc:.4f} (周期 {best_epoch})\n")
        f.write("\n每个字符的最终验证准确率:\n")
        for i in range(1, 8):
            acc = history.history[f'val_c{i}_accuracy'][-1]
            f.write(f"字符 {i}: {acc:.4f}\n")

        avg_acc = np.mean([history.history[f'val_c{i}_accuracy'][-1] for i in range(1, 8)])
        f.write(f"\n平均验证准确率: {avg_acc:.4f}\n")

        final_loss = history.history['val_loss'][-1]
        f.write(f"最终验证损失: {final_loss:.4f}\n")

    plt.figure(figsize=(15, 10))
    sample_indices = np.random.choice(len(X_val), min(10, len(X_val)), replace=False)

    for i, idx in enumerate(sample_indices):
        img = X_val[idx]
        true_labels = y_val[idx]

        predictions = history.model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_labels = [np.argmax(pred[0]) for pred in predictions]
        true_chars = ''.join([reverse_char_dict[label] for label in true_labels])
        pred_chars = ''.join([reverse_char_dict[label] for label in pred_labels])

        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"真实: {true_chars}\n预测: {pred_chars}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_examples.png', dpi=150)
    plt.close()
    try:
        log_df = pd.read_csv(output_dir / 'training_log.csv')
        log_df.to_excel(output_dir / 'training_log.xlsx', index=False)
    except Exception as e:
        print(f"保存Excel日志时出错: {str(e)}")


if __name__ == '__main__':
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
