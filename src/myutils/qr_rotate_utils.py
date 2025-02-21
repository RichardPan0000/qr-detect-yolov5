import cv2
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def calculate_rotation_angle(quad):
    """计算四边形的倾斜角度
    因为np.arctan2 只返回[-pi,pi] 之间
    加上后面限制，所以角度只会在[0,180]之间。

    """
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

def process_quad_v2(quad_xy):
    """处理四边形：计算角度和生成旋转矩形"""
    # 1. 计算倾斜角度
    angle = calculate_rotation_angle(quad_xy)

    # 2. 计算中心点
    center = np.mean(quad_xy, axis=0)

    # 3. 计算近似矩形的宽度和高度
    width, height = calculate_rectangle_size(quad_xy)

    # 4. 创建旋转矩形
    rotated_rect,angle_rad = create_rotated_rectangle(center, width, height, angle)

    # print(f"Center: ({center[0]:.2f}, {center[1]:.2f})")
    # print(f"Width: {width:.2f}, Height: {height:.2f}")

    return rotated_rect,angle_rad,(width,height)

def correct_angles(transformed_num_dets, rec_centers, angles):
    # 对transformed_num_dets 进行for循环，
    for i,det in enumerate(transformed_num_dets):
        rec_center=rec_centers[i]
        angle_rad=angles[i]
        # 看是角度还是弧度
        # angles[i]=angle_correction(det, rec_center, angle_rad)
        angles[i] = angle_correction_v4(det, rec_center, angle_rad)
    return angles


# import math
# import numpy as np

def angle_correction_v4(transformed_num_det, rec_centers, angle_rad):
    """
    根据数字和二维码中心点向量的角度 (angle_A_degrees)，基于水平/垂直阈值和象限判断校正二维码角度.

    Args:
        transformed_num_det: 数字的边界框 (x1, y1, x2, y2).
        rec_centers: 二维码中心点坐标 (x, y).
        angle_rad: 从二维码外接四边形计算得到的角度 (弧度).

    Returns:
        corrected_angle_rad: 校正后的角度 (弧度).
    """
    # 1. 计算数字中心点坐标
    angle=math.degrees(angle_rad) # 仍然转换为角度，仅为了后续打印方便查看，核心逻辑使用弧度
    num_x1, num_y1, num_x2, num_y2 = transformed_num_det
    num_center_x = (num_x1 + num_x2) / 2
    num_center_y = (num_y1 + num_y2) / 2
    num_center = np.array([num_center_x, num_center_y])

    # 2. 获取二维码中心点坐标
    rec_center = np.array(rec_centers)

    # 3. 计算向量 vec_A (从二维码中心指向数字中心)
    vec_A = num_center - rec_center
    dx = vec_A[0]
    dy = vec_A[1]

    # 4. 计算角度A (angle_A) - 使用 atan2 得到有符号角度，范围 [-pi, pi] 弧度
    angle_A_radians = math.atan2(dy, dx)
    angle_A_degrees = math.degrees(angle_A_radians)

    # 5. 角度校正逻辑 (优先水平/垂直阈值判断，后备象限判断)
    corrected_angle_rad = angle_rad  # 默认不校正
    threshold_angle = 10 # 角度阈值， degrees
    angle_90_radians = math.pi / 2
    angle_180_degrees = 90


    is_horizontal = (abs(angle_A_degrees - 0) <= threshold_angle) or (abs(angle_A_degrees - 180) <= threshold_angle)
    is_vertical = (abs(angle_A_degrees - 90) <= threshold_angle) or (abs(angle_A_degrees - 270) <= threshold_angle)

    correction_reason = "Default (No Correction)" # 记录校正原因，默认为不校正

    if angle is not None:  # 确保 angle 不是 None
        if is_horizontal:
            corrected_angle_rad = 0.0 # 水平方向，校正角度为 0 弧度 (水平)
            correction_reason = "Horizontal Correction"
        elif is_vertical:
            corrected_angle_rad = math.pi / 2 # 垂直方向，校正角度为 90 度 (垂直)
            correction_reason = "Vertical Correction"
        else: # 既不水平也不垂直，应用象限判断逻辑
            if 0 < angle_A_radians < angle_90_radians or -math.pi < angle_A_radians < -angle_90_radians:
                # 角度A 在 第一象限 (0 < angle_A < pi/2) 或 第三象限 (-pi < angle_A < -pi/2)
                if angle > angle_180_degrees: # 原始角度 angle  > 90 度 (角度制)
                    corrected_angle_rad = angle_rad - angle_90_radians
                    correction_reason = "Quadrant Correction (Q1/Q3)"
            elif angle_90_radians < angle_A_radians <= math.pi or -angle_90_radians < angle_A_radians <= 0:
                # 角度A 在 第二象限 (pi/2 < angle_A <= pi) 或 第四象限 (-pi/2 < angle_A <= 0)
                if angle < angle_180_degrees: # 原始角度 angle < 90 度 (角度制)
                    corrected_angle_rad = angle_90_radians - angle_rad
                    correction_reason = "Quadrant Correction (Q2/Q4)"
            else:
                correction_reason = "No Horizontal/Vertical/Quadrant Correction" # 明确标记没有应用任何校正

    corrected_angle=np.degrees(corrected_angle_rad) # 转换回角度，仅为了打印输出，核心返回值是弧度
    print(f"Original Angle: {angle:.1f}°  Corrected Angle: {corrected_angle:.1f}°,  Angle A: {angle_A_degrees:.1f}°  Horizontal: {is_horizontal}, Vertical: {is_vertical}, Reason: {correction_reason}")

    return corrected_angle_rad # 返回校正后的角度 (弧度)



def angle_correction(transformed_num_det, rec_centers, angle_rad):
    """
    根据数字和二维码中心点向量的角度，校正二维码角度.

    Args:
        transformed_num_det: 数字的边界框 (x1, y1, x2, y2).
        rec_centers: 二维码中心点坐标 (x, y).
        angle: 从二维码外接四边形计算得到的角度 (0-180度).

    Returns:
        corrected_angle: 校正后的角度 (0-180度).
    """
    # 转换为角度制
    # 1. 计算数字中心点坐标
    angle=math.degrees(angle_rad)

    num_x1, num_y1, num_x2, num_y2 = transformed_num_det
    num_center_x = (num_x1 + num_x2) / 2
    num_center_y = (num_y1 + num_y2) / 2
    num_center = np.array([num_center_x, num_center_y])

    # 2. 获取二维码中心点坐标
    rec_center = np.array(rec_centers)

    # 3. 计算向量 vec_A (从二维码中心指向数字中心)
    vec_A = num_center - rec_center
    dx = vec_A[0]
    dy = vec_A[1]

    # 4. 计算角度A (angle_A) - 使用 atan2 得到有符号角度，范围 [-pi, pi] 弧度
    angle_A_radians = math.atan2(dy, dx)
    angle_A_degrees = math.degrees(angle_A_radians)



    # 5. 角度校正逻辑
    corrected_angle = angle  # 默认不校正，先赋值为原始角度
    corrected_angle_rad=np.radians(corrected_angle)
    if angle is not None: # 确保 angle 不是 None

        if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
            # 角度A 在 第一象限 (dx>0, dy>0) 或 第三象限 (dx<0, dy<0) (图像坐标系)
            if angle > 90:
                # corrected_angle = 180 - angle
                corrected_angle_rad = angle_rad-math.pi/2
        elif (dx < 0 and dy > 0) or (dx > 0 and dy < 0):
            # 角度A 在 第二象限 (dx<0, dy>0) 或 第四象限 (dx>0, dy<0) (图像坐标系)
            if angle < 90:
                # corrected_angle = 180 - angle
                corrected_angle_rad = math.pi/2 - angle_rad
    corrected_angle=np.degrees(corrected_angle_rad)
    # print( f"Original Angle_rad: {angle:.1f}°  Corrected Angle: {corrected_angle:.1f}°, corrected_angle_rad:{ corrected_angle_rad}   Angle A: {angle_A_degrees:.1f}°")

    return  corrected_angle_rad# 返回校正后的角度和角度A

def generate_test_data():
    """生成多组测试数据，覆盖不同象限和角度情况."""
    test_data = []
    # Case 1: 角度A在第一象限，angle > 90
    test_data.append({
        "transformed_num_dets": [100, 150, 200, 200],
        "rec_centers": [150, 100],
        "angle": 135.0,
        "case_name": "Case 1: Quadrant 1, angle > 90"
    })
    # Case 2: 角度A在第三象限，angle > 90
    test_data.append({
        "transformed_num_dets": [50, 50, 150, 100],
        "rec_centers": [200, 150],
        "angle": 120.0,
        "case_name": "Case 2: Quadrant 3, angle > 90"
    })
    # Case 3: 角度A在第二象限，angle < 90
    test_data.append({
        "transformed_num_dets": [50, 150, 100, 200],
        "rec_centers": [150, 100],
        "angle": 45.0,
        "case_name": "Case 3: Quadrant 2, angle < 90"
    })
    # Case 4: 角度A在第四象限，angle < 90
    test_data.append({
        "transformed_num_dets": [200, 50, 250, 100],
        "rec_centers": [150, 150],
        "angle": 60.0,
        "case_name": "Case 4: Quadrant 4, angle < 90"
    })
    # Case 5: 角度A在第一象限，angle < 90，无需校正
    test_data.append({
        "transformed_num_dets": [160, 110, 210, 140],
        "rec_centers": [150, 100],
        "angle": 30.0,
        "case_name": "Case 5: Quadrant 1, angle < 90, No Correction"
    })
     # Case 6: 角度A在第二象限，angle > 90，无需校正
    test_data.append({
        "transformed_num_dets": [40, 110, 90, 140],
        "rec_centers": [100, 100],
        "angle": 120.0,
        "case_name": "Case 6: Quadrant 2, angle > 90, No Correction"
    })
    return test_data

def visualize_correction(test_case, corrected_angle, angle_A_degrees):
    """绘制图表展示角度校正前后的效果."""
    transformed_num_dets = test_case["transformed_num_dets"]
    rec_centers = test_case["rec_centers"]
    angle = test_case["angle"]
    case_name = test_case["case_name"]

    num_x1, num_y1, num_x2, num_y2 = transformed_num_dets
    num_center_x = (num_x1 + num_x2) / 2
    num_center_y = (num_y1 + num_y2) / 2
    num_center = np.array([num_center_x, num_center_y])
    rec_center = np.array(rec_centers)

    fig, ax = plt.subplots()
    ax.set_aspect('equal') # 保证坐标轴比例一致
    ax.invert_yaxis() # 反转Y轴，使其符合图像坐标系

    # 绘制数字边框
    rect = plt.Rectangle((num_x1, num_y1), num_x2 - num_x1, num_y2 - num_y1, linewidth=1, edgecolor='r', facecolor='none', label='Number Box')
    ax.add_patch(rect)
    # 绘制二维码中心点
    ax.plot(rec_center[0], rec_center[1], 'bo', markersize=8, label='QR Code Center')
    # 绘制数字中心点
    ax.plot(num_center[0], num_center[1], 'go', markersize=8, label='Number Center')
    # 绘制连接线
    ax.plot([rec_center[0], num_center[0]], [rec_center[1], num_center[1]], 'k--', linewidth=1, label='Vector A')

    # 设置图表标题和标签
    ax.set_title(f"{case_name}\nOriginal Angle: {angle:.1f}°  Corrected Angle: {corrected_angle:.1f}°  Angle A: {angle_A_degrees:.1f}°")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='upper right')

    # 自动调整坐标轴范围，使所有元素可见
    min_x = min(num_x1, rec_center[0], num_center[0]) - 20
    min_y = min(num_y1, rec_center[1], num_center[1]) - 20
    max_x = max(num_x2, rec_center[0], num_center[0]) + 20
    max_y = max(num_y2, rec_center[1], num_center[1]) + 20
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


    plt.show()

# if __name__ == '__main__':
#     test_data_list = generate_test_data()
#
#     for test_case in test_data_list:
#         transformed_num_dets = test_case["transformed_num_dets"]
#         rec_centers = test_case["rec_centers"]
#         angle = test_case["angle"]
#         case_name = test_case["case_name"]
#
#         corrected_angle_result, angle_A_degrees_result = angle_correction(transformed_num_dets, rec_centers, angle)
#         # print(f"\n----- {case_name} -----")
#         # print(f"原始角度 (angle): {angle:.1f}°")
#         # print(f"角度A (angle_A_degrees): {angle_A_degrees_result:.1f}°")
#         # print(f"校正后角度 (corrected_angle): {corrected_angle_result:.1f}°")
#
#         visualize_correction(test_case, corrected_angle_result, angle_A_degrees_result)

# # 使用示例
# if __name__ == "__main__":
#     # 原始四边形数据
#     quad_xy = np.array([
#         [31.257, 103.38],
#         [175.75, 36.258],
#         [253.12, 196.8],
#         [107.31, 263.08]
#     ], dtype=np.float32)
#
#     # 处理四边形并生成旋转矩形
#     rotated_rect,_ = process_quad(quad_xy)
#     print(rotated_rect)
#     # 可视化结果
#     visualize_quads(quad_xy, rotated_rect)
def my_test_correct_v4():

    # 示例使用 (假设 transformed_num_det, rec_centers, angle_rad 已经有值)
    # 为了测试方便，这里假设一些输入值
    transformed_num_det_example = [10, 10, 20, 20] # 示例数字边界框
    rec_centers_example = [15, 15] # 示例二维码中心点


    # 测试 水平方向
    angle_rad_example_horizontal = math.radians(30)
    vec_A_horizontal = np.array([5, 0])
    angle_A_radians_horizontal = math.atan2(vec_A_horizontal[1], vec_A_horizontal[0])
    angle_A_degrees_horizontal = math.degrees(angle_A_radians_horizontal)
    corrected_angle_rad_result_h = angle_correction_v4(transformed_num_det_example, rec_centers_example, angle_rad_example_horizontal)
    print(f"\n--- 水平方向 ---")
    print(f"Angle A (degrees): {angle_A_degrees_horizontal:.1f}°")
    print(f"校正后角度 (角度): {math.degrees(corrected_angle_rad_result_h):.1f}°")


    # 测试 垂直方向
    angle_rad_example_vertical = math.radians(60)
    vec_A_vertical = np.array([0, 5])
    angle_A_radians_vertical = math.atan2(vec_A_vertical[1], vec_A_vertical[0]) # 故意使用 vec_A_vertical_90 错误变量名，已修正为 vec_A_vertical
    angle_A_degrees_vertical = math.degrees(angle_A_radians_vertical)
    corrected_angle_rad_result_v = angle_correction_v4(transformed_num_det_example, rec_centers_example, angle_rad_example_vertical)
    print(f"\n--- 垂直方向 ---")
    print(f"Angle A (degrees): {angle_A_degrees_vertical:.1f}°")
    print(f"校正后角度 (角度): {math.degrees(corrected_angle_rad_result_v):.1f}°")


    # 测试  第一象限，应用象限校正
    angle_rad_example_q1 = math.radians(120) # 原始角度 > 90
    vec_A_q1 = np.array([3, 2]) # 第一象限
    angle_A_radians_q1 = math.atan2(vec_A_q1[1], vec_A_q1[0])
    angle_A_degrees_q1 = math.degrees(angle_A_radians_q1)
    corrected_angle_rad_result_q1 = angle_correction_v4(transformed_num_det_example, rec_centers_example, angle_rad_example_q1)
    print(f"\n--- 第一象限 (象限校正) ---")
    print(f"Angle A (degrees): {angle_A_degrees_q1:.1f}°")
    print(f"校正后角度 (角度): {math.degrees(corrected_angle_rad_result_q1):.1f}°")


    # 测试  第二象限，应用象限校正
    angle_rad_example_q2 = math.radians(45) # 原始角度 < 90
    vec_A_q2 = np.array([-2, 3]) # 第二象限
    angle_A_radians_q2 = math.atan2(vec_A_q2[1], vec_A_q2[0])
    angle_A_degrees_q2 = math.degrees(angle_A_radians_q2)
    corrected_angle_rad_result_q2 = angle_correction_v4(transformed_num_det_example, rec_centers_example, angle_rad_example_q2)
    print(f"\n--- 第二象限 (象限校正) ---")
    print(f"Angle A (degrees): {angle_A_degrees_q2:.1f}°")
    print(f"校正后角度 (角度): {math.degrees(corrected_angle_rad_result_q2):.1f}°")


    # 测试  非水平非垂直，且不满足象限校正条件 (默认不校正)
    angle_rad_example_default = math.radians(60) # 原始角度 < 90
    vec_A_default = np.array([3, 30]) #  Angle A 角度很小，不满足象限校正条件
    angle_A_radians_default = math.atan2(vec_A_default[1], vec_A_default[0])
    angle_A_degrees_default = math.degrees(angle_A_radians_default)
    corrected_angle_rad_result_default = angle_correction_v4(transformed_num_det_example, rec_centers_example, angle_rad_example_default)
    print(f"\n--- 默认不校正 (非水平/垂直/象限) ---")
    print(f"Angle A (degrees): {angle_A_degrees_default:.1f}°")
    print(f"校正后角度 (角度): {math.degrees(corrected_angle_rad_result_default):.1f}°")
