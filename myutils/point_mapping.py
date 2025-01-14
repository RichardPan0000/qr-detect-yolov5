import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
# 原始三角形的三个顶点 (原始坐标系)
src_points = np.array([[100, 200], [400, 200], [200, 500]], dtype=np.float32)

# 实际拍照后三角形的三个顶点 (拍照后坐标系)
dst_points = np.array([[120, 220], [420, 180], [220, 520]], dtype=np.float32)

# 计算仿射变换矩阵
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
print("Affine Transformation Matrix:")
print(affine_matrix)

# 创建一个空白图像来绘制三角形
img = np.ones((600, 600, 3), dtype=np.uint8) * 255  # 白色背景

# 绘制原始三角形
cv2.polylines(img, [np.int32([src_points])], isClosed=True, color=(0, 0, 255), thickness=2)

# 绘制二维码框的原始四个顶点
qr_box = np.array([[150, 250], [350, 250], [350, 450], [150, 450]], dtype=np.float32)
cv2.polylines(img, [np.int32([qr_box])], isClosed=True, color=(0, 255, 0), thickness=2)

# 应用仿射变换
transformed_img = cv2.warpAffine(img, affine_matrix, (img.shape[1], img.shape[0]))

# 绘制变换后的三角形
cv2.polylines(transformed_img, [np.int32([dst_points])], isClosed=True, color=(0, 255, 0), thickness=2)

# 绘制二维码框的变换后的四个顶点
# 应用仿射变换到框的四个顶点

transformed_qr_box = cv2.transform(np.array([qr_box]), affine_matrix)[0]
print('transformed_qr_box',transformed_qr_box)
cv2.polylines(transformed_img, [np.int32([transformed_qr_box])], isClosed=True, color=(255, 0, 0), thickness=2)

# 使用matplotlib展示
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 显示原始图像
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")
ax[0].axis('off')

# 显示经过仿射变换后的图像
ax[1].imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
ax[1].set_title("Transformed Image")
ax[1].axis('off')

# 在原始图像中标出实际拍照后的三个顶点
for point in dst_points:
    ax[0].plot(point[0], point[1], 'bo')  # 蓝色圆点表示实际拍照后的顶点

# 在变换后的图像中标出实际拍照后的三个顶点
for point in dst_points:
    ax[1].plot(point[0], point[1], 'ro')  # 红色圆点表示实际拍照后的顶点

# 绘制框的四个顶点（原始和变换后的）
for point in qr_box:
    ax[0].plot(point[0], point[1], 'go')  # 绿色圆点表示二维码框的四个顶点（原始）

for point in transformed_qr_box:
    ax[1].plot(point[0], point[1], 'mo')  # 粉色圆点表示二维码框的四个顶点（变换后的）

# 在图像上添加注释
for i, point in enumerate(dst_points):
    ax[0].text(point[0] + 10, point[1] - 10, f'P{i+1}', color='blue', fontsize=12)
    ax[1].text(point[0] + 10, point[1] - 10, f'P{i+1}', color='red', fontsize=12)

# 绘制框的注释
for i, point in enumerate(qr_box):
    ax[0].text(point[0] + 10, point[1] - 10, f'Q{i+1}', color='green', fontsize=12)
for i, point in enumerate(transformed_qr_box):
    ax[1].text(point[0] + 10, point[1] - 10, f'Q{i+1}', color='magenta', fontsize=12)

plt.tight_layout()
plt.show()
'''

def get_affine_matrix(src_points = np.array([[100, 200], [400, 200], [200, 500]], dtype=np.float32),dst_points = np.array([[120, 220], [420, 180], [220, 520]], dtype=np.float32)):
    # 原始三角形的三个顶点 (原始坐标系)
    # 实际拍照后三角形的三个顶点 (拍照后坐标系
    # 计算仿射变换矩阵
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    return affine_matrix

# 返回变换后的坐标
def get_transformed_box(affine_matrix,qr_box:list):
    transformed_qr_box = cv2.transform(np.array([qr_box]), affine_matrix)[0]
    return transformed_qr_box

def get_transformed_box(qr_box:list,src_points = np.array([[100, 200], [400, 200], [200, 500]], dtype=np.float32),dst_points = np.array([[120, 220], [420, 180], [220, 520]], dtype=np.float32)):
    affine_matrix=get_affine_matrix(src_points,dst_points)
    qr_cpu_list=[[ele.cpu() for ele in sublist] for sublist in qr_box]
    qr_cpu_list=np.array(qr_cpu_list,dtype=np.float32)
    qr_cpu_list=qr_cpu_list.reshape(-1,2)
    transformed_qr_box = cv2.transform(np.array([qr_cpu_list]), affine_matrix)[0]
    transformed_qr_box=transformed_qr_box.reshape(-1,4) #再将其变回原来的形式
    return transformed_qr_box
def get_transformed_quad_xyes(qr_box:list,src_points = np.array([[100, 200], [400, 200], [200, 500]], dtype=np.float32),dst_points = np.array([[120, 220], [420, 180], [220, 520]], dtype=np.float32)):
    """
    使用透视变换将框进行变换
    :param qr_box: 输入框的坐标列表（二维数组）
    :param src_points: 源点的四个顶点坐标
    :param dst_points: 目标点的四个顶点坐标
    :return: 透视变换后的框
    """
    # 计算透视变换矩阵
    affine_matrix=get_affine_matrix(src_points,dst_points)

    # 将框的坐标转换为 NumPy 数组并展平为二维点集
    quadxy_cpu_list = [[ele.cpu() if hasattr(ele, 'cpu') else ele for ele in sublist] for sublist in qr_box]
    quadxy_cpu_list = np.array(quadxy_cpu_list, dtype=np.float32).reshape(-1, 2)  # 变为 [N, 2] 形式

    # 应用透视变换
    transformed_quadxy_box = cv2.transform(np.array([quadxy_cpu_list]), affine_matrix)[0]

    # 将变换后的点重新组织为原始形式
    transformed_quadxy_box = transformed_quadxy_box.reshape( -1,4, 2)
    # transformed_quadxy_box = transformed_quadxy_box.reshape( -1,4)

    return transformed_quadxy_box

def get_transformed_box_four(qr_box: list,
                        src_points=np.array([[100, 200], [400, 200], [400, 500], [100, 500]], dtype=np.float32),
                        dst_points=np.array([[120, 220], [420, 180], [420, 520], [120, 520]], dtype=np.float32))->np.ndarray:
    """
    使用透视变换将框进行变换
    :param qr_box: 输入框的坐标列表（二维数组）
    :param src_points: 源点的四个顶点坐标
    :param dst_points: 目标点的四个顶点坐标
    :return: 透视变换后的框
    """
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 将框的坐标转换为 NumPy 数组并展平为二维点集
    qr_cpu_list = [[ele.cpu() if hasattr(ele, 'cpu') else ele for ele in sublist] for sublist in qr_box]
    qr_cpu_list = np.array(qr_cpu_list, dtype=np.float32).reshape(-1, 2)  # 变为 [N, 2] 形式

    # 应用透视变换
    transformed_qr_box = cv2.perspectiveTransform(np.array([qr_cpu_list]), perspective_matrix)[0]

    # 将变换后的点重新组织为原始形式
    # transformed_qr_box = transformed_qr_box.reshape(len(qr_box), -1, 2)
    transformed_qr_box = transformed_qr_box.reshape( -1, 4)

    return transformed_qr_box.tolist()
def get_transformed_quad_xyes_four(qr_box: list,
                             src_points=np.array([[100, 200], [400, 200], [400, 500], [100, 500]], dtype=np.float32),
                             dst_points=np.array([[120, 220], [420, 180], [420, 520], [120, 520]], dtype=np.float32)):
    """
    使用透视变换将框进行变换
    :param qr_box: 输入框的坐标列表（二维数组）
    :param src_points: 源点的四个顶点坐标
    :param dst_points: 目标点的四个顶点坐标
    :return: 透视变换后的框
    """
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 将框的坐标转换为 NumPy 数组并展平为二维点集
    quadxy_cpu_list = [[ele.cpu() if hasattr(ele, 'cpu') else ele for ele in sublist] for sublist in qr_box]
    quadxy_cpu_list = np.array(quadxy_cpu_list, dtype=np.float32).reshape(-1, 2)  # 变为 [N, 2] 形式

    # 应用透视变换
    transformed_quadxy_box = cv2.perspectiveTransform(np.array([quadxy_cpu_list]), perspective_matrix)[0]

    # 将变换后的点重新组织为原始形式
    # transformed_quadxy_box = transformed_quadxy_box.reshape(len(qr_box), -1, 2)
    transformed_quadxy_box = transformed_quadxy_box.reshape( -1,4, 2)

    return transformed_quadxy_box
