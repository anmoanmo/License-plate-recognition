# License Plate Recognition

基于 YOLO 的车牌检测与识别示例项目。项目流程包括车辆图片输入、车牌区域检测、车牌透视矫正、字符识别和 PySide6 图形界面展示。

## 功能

- 使用 YOLO 检测图片中的车牌区域。
- 对车牌区域做裁剪、角点修正和透视矫正。
- 使用字符识别模型输出 7 位普通车牌号。
- 提供命令行检测入口和 PySide6 GUI。
- 内置可运行 demo 权重：`best.pt` 和 `training_results/best_model.h5`。

## 目录结构

```text
.
├── MainUI.py                         GUI 主程序
├── detect.py                         命令行检测入口
├── best.pt                           YOLO 车牌检测权重
├── training_results/best_model.h5    字符识别模型权重
├── plate_align.py                    车牌区域矫正与透视变换
├── plate_recognizer.py               字符识别运行时
├── project_utils.py                  公共路径和图片读取工具
├── yolo_runtime.py                   Ultralytics YOLO 导入兼容层
├── ui/                               Qt Designer 生成的界面文件
├── test_images/                      示例图片
├── scripts/                          数据转换和评估工具
├── requirements.txt                  Python 依赖
└── data.yaml.example                 YOLO 数据集配置模板
```

## 环境安装

建议使用 Python 3.10 或 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 Conda：

```bash
conda create -n plate-lpr python=3.11 -y
conda activate plate-lpr
pip install -r requirements.txt
```

说明：

- `ultralytics` 用于 YOLO 检测。
- `tensorflow` 用于加载内置 `.h5` 字符识别模型。
- `torch` 用于可选 PyTorch 字符识别模型。
- `PySide6` 用于图形界面。

## 运行

命令行检测：

```bash
python detect.py --source test_images/3.jpg
```

启动 GUI：

```bash
python MainUI.py
```

GUI 操作流程：

1. 点击“上传图片”选择图片。
2. 点击“检测车牌”定位车牌区域。
3. 点击“识别车牌”读取车牌号。
4. 点击“清空”重置界面。

## 训练

复制数据集模板：

```bash
cp data.yaml.example data.yaml
```

按你的数据集路径修改 `data.yaml` 后训练 YOLO 检测模型：

```bash
python train.py --data data.yaml --model yolo11n.pt --epochs 50 --batch 4
```

CCPD 数据转换脚本：

```bash
python scripts/prepare_ccpd_dataset.py \
  --raw-dir /path/to/ccpd/raw \
  --output-root datasets/ccpd_prepared \
  --max-images 5000 \
  --copy-mode hardlink
```

训练 PyTorch 字符识别模型：

```bash
python train_recognizer.py \
  --data-dir datasets/ccpd_prepared/plate_recognition_dataset \
  --output-dir training_results \
  --epochs 60 \
  --batch-size 1024
```

如果存在 `training_results/best_recognizer.pt`，程序会优先加载 PyTorch 识别模型；否则回退到内置的 `training_results/best_model.h5`。

## 数据集局限

本项目实验中使用过 CCPD 数据集。CCPD 文件名包含车牌框、四角点和车牌号标签，适合用于训练检测与识别模型。

需要注意：

- CCPD 中不同省份样本分布不均衡。
- 稀有省份和高频省份会造成省份简称识别偏置。
- `D/0/Q`、`S/5`、`B/8`、`Z/2/7` 等形近字符容易混淆。
- 提升实际准确率通常需要补充更均衡、更高质量的真实车牌数据。

## License

本项目代码以 MIT License 开源。第三方依赖和模型框架遵循各自许可证。
