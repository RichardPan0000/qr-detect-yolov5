import numpy as np
import math

# 计算向量的点积
def dot_product(v1, v2):
    return np.dot(v1, v2)


# 计算向量的模长
def vector_magnitude(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2)

def compute_center(box):
    x1,y1,x2,y2=box[0],box[1],box[2],box[3]

    return ((x1+x2)/2.0,(y1+y2)/2.0)



# 计算向量之间的角度
def angle_between(v1, v2):
    dot_prod = dot_product(v1, v2)
    mag_v1 = vector_magnitude(v1)
    mag_v2 = vector_magnitude(v2)

    cos_theta = dot_prod / (mag_v1 * mag_v2)

    # 防止数值误差导致cos_theta超过-1到1的范围
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# 找出直角顶点
def find_right_angle_vertex(box1, box2, box3):
    # 计算向量
    v1 = np.array([box2[0] - box1[0], box2[1] - box1[1]])  # box1 -> box2
    v2 = np.array([box3[0] - box1[0], box3[1] - box1[1]])  # box1 -> box3
    v3 = np.array([box3[0] - box2[0], box3[1] - box2[1]])  # box2 -> box3

    # 计算角度
    angle1 = angle_between(v1, v2)  # box1 -> box2 和 box1 -> box3 的夹角
    angle2 = angle_between(-v1, v3)  # box1 -> box2 和 box2 -> box3 的夹角
    angle3 = angle_between(v2, v3)  # box1 -> box3 和 box2 -> box3 的夹角
    # 设置一个容忍度阈值
    threshold = 5  # 比如容忍角度偏差±5度
    # 找到最大角度，并判断它是否接近 90 度
    angles = [angle1, angle2, angle3]
    max_angle = max(angles)
    # if 90 - threshold <= max_angle <= 90 + threshold:
    #     # 找到最大角度接近 90 度的那个
    #     if max_angle == angle1:
    #         return "box1"
    #     elif max_angle == angle2:
    #         return "box2"
    #     elif max_angle == angle3:
    #         return "box3"
    # else:
    #     return "No right angle detected"

    # 将三个box的中心返回

    center1=compute_center(box1)
    center2=compute_center(box2)
    center3=compute_center(box3)
    angle_box_dict={angle1:center1,angle2:center2,angle3:center3}
    angle_box_dict=dict(sorted(angle_box_dict.items())) # 按值排序
    return angle_box_dict




def sort_boxes_by_corners(box1, box2, box3, box4):
    """
    根据左上角、左下角、右下角、右上角的顺序排列四个框
    :param box1, box2, box3, box4: 四个 box，每个格式为 [x1, y1, x2, y2]
    :return: 按顺序排列的四个框
    """
    # 将 box 统一到一个列表中
    boxes = [box1, box2, box3, box4]

    # 计算每个 box 的中心点
    centers = [(box, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)) for box in boxes]

    # 按中心点的 y 坐标排序
    centers = sorted(centers, key=lambda b: b[1][1])  # 先按 y 值排序
    top_boxes = sorted(centers[:2], key=lambda b: b[1][0])  # 上方的两个按 x 值排序
    bottom_boxes = sorted(centers[2:], key=lambda b: b[1][0])  # 下方的两个按 x 值排序

    # 左上、左下、右下、右上
    sorted_boxes = [top_boxes[0][0], bottom_boxes[0][0], bottom_boxes[1][0], top_boxes[1][0]]
    return sorted_boxes


import numpy as np


def sort_boxes_to_rectangle(box1, box2, box3, box4):
    """
    将四个 box 按照矩形的顺序排列为左上、左下、右下、右上。
    """
    # 提取点坐标
    points = np.array([box1, box2, box3, box4], dtype=np.float32)

    # 计算所有两两点之间的欧几里得距离
    distances = []
    for i in range(4):
        for j in range(i + 1, 4):
            distances.append((np.linalg.norm(points[i] - points[j]), i, j))

    # 找出两条最短边
    distances = sorted(distances, key=lambda x: x[0])
    edge1 = distances[0]  # 最短边
    edge2 = distances[1]  # 第二短边

    # 短边的四个点
    short_edge_points = {edge1[1], edge1[2], edge2[1], edge2[2]}
    if len(short_edge_points) != 4:
        raise ValueError("输入点不符合矩形逻辑，无法排序")

    # 将点从短边中提取出来
    short_edge_points = list(short_edge_points)
    p1, p2 = points[edge1[1]], points[edge1[2]]  # 最短边的两个点
    p3, p4 = points[edge2[1]], points[edge2[2]]  # 第二短边的两个点

    # 确定短边的上下关系（左上和左下 or 右上和右下）
    if p1[1] < p2[1]:  # p1 在上方
        left_top, left_bottom = p1, p2
    else:  # p2 在上方
        left_top, left_bottom = p2, p1

    if p3[1] < p4[1]:  # p3 在上方
        right_top, right_bottom = p3, p4
    else:  # p4 在上方
        right_top, right_bottom = p4, p3

    # 按矩形顺序返回点
    return [compute_center(left_top),compute_center(left_bottom),compute_center(right_bottom),compute_center(right_top)]
    # return [left_top.tolist(), left_bottom.tolist(), right_bottom.tolist(), right_top.tolist()]

def sort_boxes_to_fixed_order(box1, box2, box3, box4):
    """
    将四个点按照固定顺序排列为右上角、左上角、左下角、右下角。
    """
    # 提取点坐标
    points = np.array([box1, box2, box3, box4], dtype=np.float32)

    # 按照 y 坐标升序排列（先区分上下两组点）
    points_sorted_y = sorted(points, key=lambda p: p[1])  # 按 y 排序

    # 上下两组点
    top_points = points_sorted_y[:2]  # y 坐标较小的两个点（上方点）
    bottom_points = points_sorted_y[2:]  # y 坐标较大的两个点（下方点）

    # 分别按 x 坐标对上下点排序
    top_points = sorted(top_points, key=lambda p: p[0])  # 左 -> 右
    bottom_points = sorted(bottom_points, key=lambda p: p[0])  # 左 -> 右

    # 固定顺序：右上角、左上角、左下角、右下角
    right_top = top_points[1]
    left_top = top_points[0]
    left_bottom = bottom_points[0]
    right_bottom = bottom_points[1]

    # return [right_top, left_top, left_bottom, right_bottom]
    return [compute_center(right_top),compute_center(left_top),compute_center(left_bottom),compute_center(right_bottom)]
def sort_boxes_by_center_angle(boxes_list,clazz_list):
    """
       找到 class=2 的框，并以其为起点，按相对于中心点的逆时针顺序排列其他框。
       输入框格式为 xyxy (x_min, y_min, x_max, y_max)。
       """
    # 计算所有框的中心点
    centers = np.array(
        [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes_list],
        dtype=np.float32
    )
    overall_center = np.mean(centers, axis=0)  # 整体中心点

    # 计算每个框相对于整体中心的角度
    angles = [
        math.atan2(center[1] - overall_center[1], center[0] - overall_center[0])
        for center in centers
    ]

    # 找到 class=2 的框的索引
    idx_class_2 = clazz_list.index(2)
    angle_class_2 = angles[idx_class_2]

    # 调整角度，使以 class=2 框为基准，逆时针排序; 因为图像识别中的坐标的y轴是相反的，所以这里要变成顺时针。
    adjusted_angles = [
        # (angle - angle_class_2) % (2 * math.pi) for angle in angles
        (angle_class_2 - angle) % (2 * math.pi) for angle in angles
    ]

    print('adjusted_angles',adjusted_angles)
    # 按调整后的角度排序
    sorted_indices = np.argsort(adjusted_angles)
    print('sorted_indices',sorted_indices)
    sorted_boxes = [boxes_list[i] for i in sorted_indices]
    return [compute_center(sorted_boxes[0]),compute_center(sorted_boxes[1]),compute_center(sorted_boxes[2]),compute_center(sorted_boxes[3])]
    # return sorted_boxes


if __name__ == '__main__':
    # 示例框和类别
    boxes_list = [
        [100, 50, 120, 70],  # Box 1
        [50, 50, 70, 70],  # Box 2
        [50, 100, 70, 120],  # Box 3 (class=2)
        [100, 100, 120, 120]  # Box 4
    ]

    # clazz_list = [0, 0, 2, 0]
    clazz_list = [0, 2, 0, 0]

    # 排序框
    sorted_boxes = sort_boxes_by_center_angle(boxes_list, clazz_list)
    print(sorted_boxes)
