import cv2
import numpy as np

from project_utils import BEST_PT_PATH, get_default_test_image, read_image
from yolo_runtime import YOLO

PLATE_TARGET_WIDTH = 240
PLATE_TARGET_HEIGHT = 80


def is_plate_like_image(image, min_aspect=2.0, max_aspect=6.0, min_side=24):
    """Return True when the whole input already looks like a plate crop."""
    if image is None or image.size == 0:
        return False
    h, w = image.shape[:2]
    if h < min_side or w < min_side:
        return False
    aspect = w / h
    return min_aspect <= aspect <= max_aspect


def _clip_box(box, image_shape):
    h, w = image_shape[:2]
    x_min, y_min, x_max, y_max = box
    x_min = int(max(0, min(w - 1, round(x_min))))
    y_min = int(max(0, min(h - 1, round(y_min))))
    x_max = int(max(0, min(w, round(x_max))))
    y_max = int(max(0, min(h, round(y_max))))
    if x_max <= x_min:
        x_max = min(w, x_min + 1)
    if y_max <= y_min:
        y_max = min(h, y_min + 1)
    return x_min, y_min, x_max, y_max


def _box_from_corners(corners):
    corners = np.asarray(corners, dtype="float32")
    x_min = float(np.min(corners[:, 0]))
    y_min = float(np.min(corners[:, 1]))
    x_max = float(np.max(corners[:, 0]))
    y_max = float(np.max(corners[:, 1]))
    return x_min, y_min, x_max, y_max


def _corners_from_box(box):
    x_min, y_min, x_max, y_max = box
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype="float32",
    )


def _expanded_box(corners, image_shape, pad_ratio=0.20, min_pad=4):
    x_min, y_min, x_max, y_max = _box_from_corners(corners)
    width = x_max - x_min
    height = y_max - y_min
    pad_x = max(min_pad, width * pad_ratio)
    pad_y = max(min_pad, height * pad_ratio)
    return _clip_box(
        (x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y),
        image_shape,
    )


def _quad_area(pts):
    pts = np.asarray(pts, dtype="float32")
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def _is_reasonable_plate_quad(pts, image_shape):
    pts = np.asarray(pts, dtype="float32")
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return False

    h, w = image_shape[:2]
    if np.any(pts[:, 0] < -2) or np.any(pts[:, 0] > w + 2):
        return False
    if np.any(pts[:, 1] < -2) or np.any(pts[:, 1] > h + 2):
        return False

    rect = order_points(pts)
    top_width = np.linalg.norm(rect[1] - rect[0])
    bottom_width = np.linalg.norm(rect[2] - rect[3])
    left_height = np.linalg.norm(rect[3] - rect[0])
    right_height = np.linalg.norm(rect[2] - rect[1])
    avg_width = (top_width + bottom_width) / 2.0
    avg_height = (left_height + right_height) / 2.0
    if avg_width < 8 or avg_height < 4:
        return False

    aspect = avg_width / max(avg_height, 1e-6)
    area = _quad_area(rect)
    return 1.6 <= aspect <= 7.5 and area >= 40


def _rect_quad_from_contour(contour, offset):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), _ = rect
    if rw <= 1 or rh <= 1:
        return None

    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
    if aspect < 1.6:
        return None

    points = cv2.boxPoints(rect).astype("float32")
    points[:, 0] += offset[0]
    points[:, 1] += offset[1]
    return order_points(points)


def _find_blue_plate_quads(image, box):
    x_min, y_min, x_max, y_max = box
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 45, 35]), np.array([140, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
    mask = cv2.bitwise_or(blue_mask, green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(80, roi.shape[0] * roi.shape[1] * 0.05)
    quads = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        quad = _rect_quad_from_contour(contour, (x_min, y_min))
        if quad is not None and _is_reasonable_plate_quad(quad, image.shape):
            quads.append(quad)
    return quads


def _find_edge_plate_quads(image, box):
    x_min, y_min, x_max, y_max = box
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(80, roi.shape[0] * roi.shape[1] * 0.04)
    quads = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        quad = _rect_quad_from_contour(contour, (x_min, y_min))
        if quad is not None and _is_reasonable_plate_quad(quad, image.shape):
            quads.append(quad)
    return quads


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
    search_box = _expanded_box(initial_corners, image.shape)
    for quad in _find_blue_plate_quads(image, search_box) + _find_edge_plate_quads(image, search_box):
        return order_points(quad).astype("float32")

    print("未找到稳定车牌角点，使用YOLO矩形框兜底")
    return order_points(initial_corners).astype("float32")


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
    pts = np.asarray(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def four_point_transform(image, pts, target_width=PLATE_TARGET_WIDTH, target_height=PLATE_TARGET_HEIGHT):
    rect = order_points(pts)
    warped_img = []
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        M,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_img.append(warped)

    return warped, warped_img


def _crop_candidate(image, box, target_width=PLATE_TARGET_WIDTH, target_height=PLATE_TARGET_HEIGHT):
    x_min, y_min, x_max, y_max = _clip_box(box, image.shape)
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _add_unique_candidate(candidates, candidate):
    if candidate is None or candidate.size == 0:
        return
    if candidate.shape[:2] != (PLATE_TARGET_HEIGHT, PLATE_TARGET_WIDTH):
        candidate = cv2.resize(
            candidate,
            (PLATE_TARGET_WIDTH, PLATE_TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

    for existing in candidates:
        if existing.shape == candidate.shape:
            diff = cv2.absdiff(existing, candidate)
            if float(np.mean(diff)) < 1.0:
                return
    candidates.append(candidate)


def build_plate_candidates(image, initial_corners, result=None, weight=0.975):
    refined_corners = refine_corners_with_contours(image, initial_corners, result, weight)
    candidates = []

    for corners in (refined_corners, order_points(initial_corners)):
        if _is_reasonable_plate_quad(corners, image.shape):
            try:
                warped, _ = four_point_transform(image, corners)
                _add_unique_candidate(candidates, warped)
            except cv2.error:
                pass

    initial_box = _clip_box(_box_from_corners(initial_corners), image.shape)
    expanded = _expanded_box(initial_corners, image.shape)
    _add_unique_candidate(candidates, _crop_candidate(image, expanded))
    _add_unique_candidate(candidates, _crop_candidate(image, initial_box))

    if not candidates:
        _add_unique_candidate(candidates, cv2.resize(image, (PLATE_TARGET_WIDTH, PLATE_TARGET_HEIGHT)))

    return candidates[0], candidates, refined_corners


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
