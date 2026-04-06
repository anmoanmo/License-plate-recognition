import cv2
import numpy as np
from ultralytics import YOLO
import os


def img_predict(image_path):
    """改进的预测函数，返回边界框和角点"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        print(f"原始图像尺寸: {image.shape}")

        # 加载模型
        model = YOLO(model='best.pt')
        results = model.predict(source=image, show=False, save=False, conf=0.5)
        result = next(iter(results))

        # 检查是否检测到车牌
        if len(result.boxes) == 0:
            print("未检测到车牌")
            return None, None, None, None

        # 获取边界框坐标
        bbox = result.boxes.xyxy[0].tolist()
        print(f"检测到的边界框: {bbox}")

        # 从边界框生成初始角点
        x_min, y_min, x_max, y_max = bbox
        initial_corners = np.array([
            [x_min, y_min],  # 左上
            [x_max, y_min],  # 右上
            [x_max, y_max],  # 右下
            [x_min, y_max]  # 左下
        ], dtype="float32")

        return bbox, initial_corners, image, result

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return None, None, None, None


def adaptive_thresholding(gray_image):
    """应用多种自适应阈值方法并选择最佳结果"""
    best_binary = None
    max_white_pixels = 0

    # 尝试不同的k值
    for k in np.linspace(-50, 0, 15):
        binary = cv2.adaptiveThreshold(
            gray_image, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            17, k
        )

        # 计算白色像素数量
        white_pixels = np.sum(binary == 255)

        # 选择白色像素最多的二值图像
        if white_pixels > max_white_pixels:
            max_white_pixels = white_pixels
            best_binary = binary

    return best_binary


def fit_line_ransac(points, offset):
    """使用RANSAC算法拟合直线"""
    if len(points) < 2:
        return 0, 0

    # 转换为适合fitLine的格式
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    # 使用RANSAC拟合直线
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    # 计算直线的斜率
    k = vy / vx

    # 计算左右两端的y坐标
    lefty = int((-x * k) + y + offset)
    righty = int(((gray_image.shape[1] - x) * k) + y + offset)

    return lefty, righty


def find_character_contours(binary):
    """在二值图像中查找字符轮廓"""
    line_upper = []
    line_lower = []

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 筛选可能是字符的轮廓
        aspect_ratio = h / float(w) if w > 0 else 0
        area = w * h

        # 字符通常具有特定的宽高比和面积范围
        if (aspect_ratio > 0.7 and 100 < area < 1200) or (aspect_ratio > 3 and area < 100):
            line_upper.append([x, y])
            line_lower.append([x + w, y + h])

    return line_upper, line_lower


def perspective_correction(image, bbox):
    """基于字符轮廓的车牌矫正方法"""
    # 提取车牌区域
    x_min, y_min, x_max, y_max = map(int, bbox)
    plate_roi = image[y_min:y_max, x_min:x_max]

    # 转换为灰度图
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

    # 应用自适应阈值
    binary = adaptive_thresholding(gray)

    # 如果无法获取有效的二值图像，返回原始区域
    if binary is None:
        return plate_roi

    # 查找字符轮廓
    line_upper, line_lower = find_character_contours(binary)

    # 如果没有找到足够的点，返回原始区域
    if len(line_upper) < 2 or len(line_lower) < 2:
        return plate_roi

    # 拟合上下边界线
    leftyA, rightyA = fit_line_ransac(line_lower, 0)
    leftyB, rightyB = fit_line_ransac(line_upper, 0)

    # 创建带边界的图像（防止边界问题）
    bordered = cv2.copyMakeBorder(plate_roi, 30, 30, 0, 0, cv2.BORDER_REPLICATE)

    # 定义源点和目标点
    h, w = bordered.shape[:2]
    pts_map1 = np.float32([
        [w - 1, rightyA + 30],  # 右下角
        [0, leftyA + 30],  # 左下角
        [w - 1, rightyB + 30],  # 右上角
        [0, leftyB + 30]  # 左上角
    ])

    # 目标点（标准车牌尺寸）
    target_w, target_h = 440, 140
    pts_map2 = np.float32([
        [target_w - 1, target_h - 1],  # 右下角
        [0, target_h - 1],  # 左下角
        [target_w - 1, 0],  # 右上角
        [0, 0]  # 左上角
    ])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts_map1, pts_map2)

    # 应用透视变换
    corrected = cv2.warpPerspective(bordered, M, (target_w, target_h))

    return corrected


def save_and_show_image(image, window_name, file_name, output_dir="output"):
    """保存并显示图像，确保目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, file_name)
    cv2.imwrite(file_path, image)
    print(f"已保存: {file_path}")

    # 显示图像
    cv2.imshow(window_name, image)
    cv2.waitKey(100)  # 短暂等待确保窗口创建


if __name__ == '__main__':
    # 获取车牌检测结果
    image_path = 'test_images/3.jpg'
    bbox, initial_corners, image, result = img_predict(image_path)

    # 检查是否检测到车牌
    if bbox is None:
        print("未检测到车牌")
        exit()

    # 显示原始图像
    save_and_show_image(image, "Original Image", "original.jpg")

    # 绘制检测框
    debug_image = image.copy()
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    save_and_show_image(debug_image, "Detection Result", "detection.jpg")

    # 应用基于字符轮廓的矫正方法
    try:
        corrected_plate = perspective_correction(image, bbox)

        if corrected_plate is None or corrected_plate.size == 0:
            raise ValueError("矫正失败，返回空图像")

        save_and_show_image(corrected_plate, "Corrected Plate", "corrected_plate.jpg")
    except Exception as e:
        print(f"矫正过程中发生错误: {e}")
        # 备选方案：使用边界框裁剪
        plate_roi = image[y_min:y_max, x_min:x_max]
        if plate_roi.size > 0:
            save_and_show_image(plate_roi, "Fallback Plate", "fallback_plate.jpg")
        else:
            print("备选方案也失败，无法获取车牌图像")

    # 等待按键关闭窗口
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()