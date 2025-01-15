import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def make_topology(pos_dict: dict, fig_save_dir, figsize=(8, 6), main_node=None):
    """
    根据目标检测的坐标生成拓扑结构，并根据指定的 figsize 绘制图。

    :param pos_dict: 字典，包含节点的坐标和对应的解码字符串
    :param fig_save_dir: 图像保存的目录
    :param figsize: 图像大小
    :param main_node: 作为中心节点的索引。如果为 None，则自动选择最中心的节点
    """
    # 创建一个简单的图
    G = nx.Graph()

    # 创建节点位置字典，pos_dict 应包含坐标和解码后的字符串
    # pos_dict 结构示例：{index: (x, y, label)}
    pos = {}  # 存储每个节点的绘制坐标
    labels = {}  # 存储每个节点对应的标签

    for idx, (coord, label) in pos_dict.items():

        x, y = coord

        if isinstance(x,torch.Tensor):
            # print('x is tensor')
            x = x.cpu()

        if isinstance(y,torch.Tensor):
            # print('y is tensor')
            y = y.cpu()
        # print('x ,y :',x,y)
        y=-y # 与cv2的读法方向一致
        pos[idx] = (x, y)
        labels[idx] = label

    # 如果没有传递 main_node，则自动选择最中心的节点
    if main_node is None:
        # 计算所有节点的几何中心（质心）
        center_x = np.mean([x for x, y in pos.values()])
        center_y = np.mean([y for x, y in pos.values()])


        # 计算每个节点到中心的距离
        distances = {node: np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for node, (x, y) in pos.items()}
        print('distances :',distances)

        # 找到距离最小的节点，作为 main_node
        main_node = min(distances, key=distances.get)

    # 将主节点与其他所有节点连接
    for node in pos.keys():
        if node != main_node:
            G.add_edge(main_node, node)

    # 绘制图形
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12, font_weight='bold',
            edge_color='gray')

    # 添加自定义的节点名字（根据解码后的字符串）
    for node, (x, y) in pos.items():
        plt.text(x, y, labels[node], color='red', fontsize=12, ha='center', va='center')

    # 设置图的标题
    plt.title("Custom 2D Topology with Node Names")

    # 显示图形
    plt.show()

    # 如果需要保存图像
    if fig_save_dir:
        plt.savefig(fig_save_dir)



if __name__ == '__main__':
    # 示例数据：假设目标检测给出了每个节点的坐标（相对于原图的坐标）以及解码后的标签
    pos_dict = {
        0: ((56, 96), 'MainNode'),  # 中心节点
        1: ((1, 2), 'Node1'),
        2: ((85, 14), 'Node2'),
        3: ((35, 0), 'Node3'),
        4: ((4, 4), 'Node4')
    }

    # 调用函数生成拓扑图
    make_topology(pos_dict, "topology_output.png", figsize=(8, 6))
