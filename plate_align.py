import cv2
import numpy as np
from ultralytics import YOLO

from project_utils import BEST_PT_PATH, get_default_test_image, read_image


def point_to_line_distance(point, line_start, line_end):
    """计算点到直线的距离"""
    if line_end[0] - line_start[0] == 0:  # 垂直线
        return abs(point[0] - line_start[0])

    # 计算直线斜率
    k = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])

    # 计算点到直线的距离公式
    numerator = abs(k * point[0] - point[1] + line_start[1] - k * line_start[0])
    denominator = np.sqrt(k ** 2 + 1)
    return numerator / denominator


def img_predict(image_path):
    try:
        image = read_image(image_path)

        # 加载模型
        model = YOLO(model=str(BEST_PT_PATH))
        results = model.predict(source=image, show=False, save=False, verbose=False)
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
            [x_min, y_max]   # 左下
        ], dtype="float32")

        return bbox, initial_corners, image, result

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return None, None, None, None


def refine_corners_with_contours(image, initial_corners, result, weight=0.975):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 自适应阈值创建二值图像
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 在边界框内查找轮廓
    x_min, y_min = int(initial_corners[0][0]), int(initial_corners[0][1])
    x_max, y_max = int(initial_corners[2][0]), int(initial_corners[2][1])

    # 确保坐标在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1] - 1, x_max)
    y_max = min(image.shape[0] - 1, y_max)

    roi = binary[y_min:y_max, x_min:x_max]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未在边界框内找到轮廓")
        return initial_corners

    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = largest_contour + np.array([[x_min, y_min]])

    # 加权距离优化角点
    refined_corners = []
    top_line = (initial_corners[0], initial_corners[1])  # 上边
    right_line = (initial_corners[1], initial_corners[2])  # 右边
    bottom_line = (initial_corners[2], initial_corners[3])  # 下边
    left_line = (initial_corners[3], initial_corners[0])  # 左边

    lines = [top_line, right_line, bottom_line, left_line]

    # 对每个角点进行优化
    for i, corner in enumerate(initial_corners):
        min_distance = float('inf')
        best_point = corner

        line1 = lines[i]
        line2 = lines[(i - 1) % 4]

        # 在轮廓点中寻找最佳匹配点
        for point in largest_contour[:, 0, :]:
            point = point.astype(float)

            # 计算点到两条边的距离
            dist_to_line1 = point_to_line_distance(point, line1[0], line1[1])
            dist_to_line2 = point_to_line_distance(point, line2[0], line2[1])
            dist_to_corner = np.linalg.norm(point - corner)
            weighted_distance = (weight * (dist_to_line1 + dist_to_line2) +
                                 (1 - weight) * dist_to_corner)

            if weighted_distance < min_distance:
                min_distance = weighted_distance
                best_point = point

        refined_corners.append(best_point)

    return np.array(refined_corners, dtype="float32")


def draw_contours(image, pts, initial_pts=None, color=(0, 255, 0), thickness=2):
    # 绘制四边形轮廓
    cv2.polylines(image, [pts.astype(np.int32)], True, color, thickness)

    # 绘制角点
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for i, point in enumerate(pts):
        x, y = point.astype(int)
        cv2.circle(image, (x, y), 5, colors[i], -1)
        cv2.putText(image, str(i + 1), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    if initial_pts is not None:
        for i, point in enumerate(initial_pts):
            x, y = point.astype(int)
            cv2.circle(image, (x, y), 5, (255, 0, 0), 2)  # 蓝色表示初始角点

    return image


def order_points(pts):
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    top_left_idx = np.argmin(sorted_pts[:, 1])
    ordered_pts = np.roll(sorted_pts, -top_left_idx, axis=0)

    return ordered_pts


def four_point_transform(image, pts):
    rect = order_points(pts)
    target_height = 145
    target_width = 450
    warped_img = []
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    warped_img.append(warped)

    return warped, warped_img


if __name__ == '__main__':
    bbox, initial_corners, image, result = img_predict(get_default_test_image())
    if bbox is None:
        print("未检测到车牌")
        exit()

    weight = 0.9  # 点线距离的权重
    refined_corners = refine_corners_with_contours(image, initial_corners, result, weight)

    debug_image = image.copy()
    debug_image = draw_contours(debug_image, refined_corners, initial_corners)
    roi_img, warped_img = four_point_transform(image, refined_corners)

    cv2.imshow("Original Image with Corners", debug_image)
    cv2.imshow("Aligned License Plate", roi_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
