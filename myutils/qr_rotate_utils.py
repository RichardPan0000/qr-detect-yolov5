import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_rotation_angle(quad):
    """计算四边形的倾斜角度"""
    p1, p2 = quad[0], quad[1]
    direction = p2 - p1
    angle_rad = np.arctan2(direction[1], direction[0])
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180
    return angle_deg


def calculate_rectangle_size(quad):
    """计算近似矩形的宽度和高度"""
    # 计算所有边的长度
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        edge = np.linalg.norm(quad[next_i] - quad[i])
        edges.append(edge)

    # 取对边的平均值作为宽和高
    width = (edges[0] + edges[2]) / 2
    height = (edges[1] + edges[3]) / 2

    return width, height


def create_rotated_rectangle(center, width, height, angle):
    """创建旋转的矩形"""
    # 创建未旋转的矩形顶点
    points = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ], dtype=np.float32)

    # 创建旋转矩阵
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # 应用旋转
    rotated_points = np.dot(points, rotation_matrix.T)

    # 平移到中心位置
    rotated_points += center

    return rotated_points,angle_rad



def visualize_quads(original_quad, rotated_rect):
    """可视化原始四边形和旋转矩形"""
    plt.figure(figsize=(10, 10))

    # 绘制原始四边形
    plt.plot(np.append(original_quad[:, 0], original_quad[0, 0]),
             np.append(original_quad[:, 1], original_quad[0, 1]),
             'r-', label='Original Quad', linewidth=2)

    # 标注原始四边形的顶点
    for i, (x, y) in enumerate(original_quad):
        plt.plot(x, y, 'ro')
        plt.text(x + 5, y + 5, f'P{i}', fontsize=12, color='red')

    # 绘制旋转矩形
    plt.plot(np.append(rotated_rect[:, 0], rotated_rect[0, 0]),
             np.append(rotated_rect[:, 1], rotated_rect[0, 1]),
             'b--', label='Rotated Rectangle', linewidth=2)

    # 标注旋转矩形的顶点
    for i, (x, y) in enumerate(rotated_rect):
        plt.plot(x, y, 'bo')
        plt.text(x - 20, y - 20, f'R{i}', fontsize=12, color='blue')

    plt.title('Original Quadrilateral vs Rotated Rectangle')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def process_quad(quad_xy):
    """处理四边形：计算角度和生成旋转矩形"""
    # 1. 计算倾斜角度
    angle = calculate_rotation_angle(quad_xy)

    # 2. 计算中心点
    center = np.mean(quad_xy, axis=0)

    # 3. 计算近似矩形的宽度和高度
    width, height = calculate_rectangle_size(quad_xy)

    # 4. 创建旋转矩形
    rotated_rect,angle_rad = create_rotated_rectangle(center, width, height, angle)

    # print(f"Rotation Angle: {angle:.2f} degrees")
    # print(f"Center: ({center[0]:.2f}, {center[1]:.2f})")
    # print(f"Width: {width:.2f}, Height: {height:.2f}")

    return rotated_rect,angle_rad


# def process_quad(quad_xy):
#     """处理四边形：计算最小外接矩形"""
#     # 确保输入是numpy数组
#     points = np.array(quad_xy, dtype=np.float32)
#
#     # 使用OpenCV找到最小外接矩形
#     rect = cv2.minAreaRect(points)
#     box = cv2.boxPoints(rect)
#     box = np.array(box)
#
#     # 获取角度
#     angle = rect[2]
#     if angle < -45:
#         angle += 90
#
#     return box, np.radians(angle)
# 使用示例
if __name__ == "__main__":
    # 原始四边形数据
    quad_xy = np.array([
        [31.257, 103.38],
        [175.75, 36.258],
        [253.12, 196.8],
        [107.31, 263.08]
    ], dtype=np.float32)

    # 处理四边形并生成旋转矩形
    rotated_rect,_ = process_quad(quad_xy)
    print(rotated_rect)
    # 可视化结果
    visualize_quads(quad_xy, rotated_rect)
