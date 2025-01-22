import matplotlib.pyplot as plt
import numpy as np
from myutils.qr_rotate_utils import process_quad,process_quad_v2
import yaml


class BaseAffineClass:
    def __init__(self, config_path=None):
        # 基础属性
        self.scale = 100
        self.origin_row_height = 0.5 * self.scale
        self.mark1_pos = (0, 0)
        self.mark2_pos = (0, 0)
        self.mark3_pos = (0, 0)
        self.mark4_pos = (0, 0)

        # 从配置文件加载
        if config_path:
            self.load_config_from_yaml(config_path)

    def load_config_from_yaml(self, config_path):
        """从YAML文件加载配置"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def calculate_distance(self, point1, point2):
        """计算两点间距离"""
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def get_box_points(self, point1, point2):
        """获取框的四个角点"""
        x1, y1 = point1
        x2, y2 = point2
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        return (x_min, y_min), (x_max, y_max)

    def get_dst_point_four(self):
        # return self.mark1_pos, self.mark2_pos, self.mark3_pos,self.mark4_pos
        return self.mark4_pos,self.mark1_pos, self.mark2_pos, self.mark3_pos
    def _draw_markers_and_distances(self, ax):
        """绘制标记点和距离标注"""
        ax.plot(self.mark1_pos[0], self.mark1_pos[1], 'go', label='Mark 1')
        ax.plot(self.mark2_pos[0], self.mark2_pos[1], 'bo', label='Mark 2')
        ax.plot(self.mark3_pos[0], self.mark3_pos[1], 'ro', label='Mark 3')
        ax.plot(self.mark4_pos[0], self.mark4_pos[1], 'yo', label='Mark 4')

        distances = {
            'Mark1-Mark2': self.calculate_distance(self.mark1_pos, self.mark2_pos),
            'Mark1-Mark3': self.calculate_distance(self.mark1_pos, self.mark3_pos),
            'Mark2-Mark3': self.calculate_distance(self.mark2_pos, self.mark3_pos)
        }

        for i, (label, dist) in enumerate(distances.items()):
            ax.text(0.05, 0.95 - i * 0.05,
                    f'Distance {label}: {dist:.2f}',
                    transform=ax.transAxes, fontsize=8, color='green')


    def _set_plot_properties(self, ax, top_left):
        """设置图表属性"""
        ax.set_aspect('equal', 'box')  # 使得坐标轴比例相同，否则变形
        ax.set_xlim([0, top_left[0] + self.cols * self.col_width])
        ax.set_ylim([0, top_left[1] + self.rows * self.new_row_height])
        ax.invert_yaxis()
        ax.set_title('Table and Box Visualization')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        plt.grid(False)
        plt.legend()


class TableAffineClass(BaseAffineClass):
    """处理表格定位的类"""
    def __init__(self, config_path=None):
        super().__init__(config_path)
        # 表格特有的属性
        self.col_width = 2.0 * self.scale
        self.area_threshold = 0.10
        self.rows = 4
        self.cols = 12
        
        # 计算衍生属性
        self.top_pos = (self.mark1_pos[0] + (1 * self.col_width + 0.5 * self.origin_row_height),
                       self.mark1_pos[1] + (4.5 * self.origin_row_height))
        self.qr_height = (3 + 0.6) * self.origin_row_height
        self.new_row_height = 6 * self.origin_row_height
        # 绘制三个标记点
        # mark1: 表格顶部左侧的坐标

        # mark2: 从底部的左侧向下偏移
        top_left, top_right, bottom_left, bottom_right = self.get_table_corners(self.top_pos, self.new_row_height,
                                                                                self.col_width, self.rows, self.cols)
        self.mark2_pos = (self.mark1_pos[0], bottom_left[1] + self.origin_row_height * 1.5)
        self.mark3_pos = (bottom_right[0] + 0.5 * self.origin_row_height, bottom_right[1] + self.origin_row_height * 1.5)
        self.mark4_pos = (self.mark3_pos[0], self.mark1_pos[1])

    def _draw_table_grid(self, ax, top_left, top_right):
        """绘制表格网格线"""
        # 绘制横线
        for row in range(self.rows + 1):
            y = top_left[1] + row * self.new_row_height
            ax.plot([top_left[0], top_right[0]], [y, y], color='black')

        # 绘制竖线
        for col in range(self.cols + 1):
            x = top_left[0] + col * self.col_width
            ax.plot([x, x],
                    [top_left[1], top_left[1] + self.rows * self.new_row_height],
                    color='black')
    def get_table_corners(self,top_pos, row_height, col_width, rows, cols):
        """
        计算表格的四个顶点坐标
        """
        table_left = top_pos[0]  # mark3 给定表格左侧的横坐标
        table_top = top_pos[1]  # mark1 给定表格顶部的纵坐标
        table_right = table_left + cols * col_width  # 表格右侧的横坐标
        table_bottom = table_top + rows * row_height  # 表格底部的纵坐标

        # 返回四个顶点
        top_left = (table_left, table_top)
        top_right = (table_right, table_top)
        bottom_left = (table_left, table_bottom)
        bottom_right = (table_right, table_bottom)

        return top_left, top_right, bottom_left, bottom_right
    # 这里保留原有的表格相关方法
    def calculate_overlap_ratio(self, x1, y1, x2, y2, cell_x1, cell_y1, cell_x2, cell_y2,method='area_ratio'):
        """
              计算框与单元格的重叠比例
              method: 'area_ratio' 使用面积比（交集面积/二维码面积）
                     'iou' 使用IOU
              """
        # [1941.6661376953125, 1019.8320922851562, 2139.2294921875, 1201.1162109375]
        # 计算交集区域
        inter_x1 = max(x1, cell_x1)
        inter_y1 = max(y1, cell_y1)
        inter_x2 = min(x2, cell_x2)
        inter_y2 = min(y2, cell_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        # 计算面积
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        qr_area = (x2 - x1) * (y2 - y1)

        if method == 'area_ratio':
            # 返回交集面积占二维码面积的比例
            return intersection / qr_area
        else:  # method == 'iou'
            cell_area = (cell_x2 - cell_x1) * (cell_y2 - cell_y1)
            union = qr_area + cell_area - intersection
            return intersection / union if union > 0 else 0.0
    # 获取框所在的格子
    def get_cell_position_v3(self, x1, y1, x2, y2, top_pos, row_height, col_width, rows, cols, keep_threshold=0):
        """
        获取框所在的格子位置，使用面积比判断，并根据实际占比情况动态调整判断标准
        """
        # 计算表格的边界
        table_left = top_pos[0]
        table_top = top_pos[1]
        table_right = table_left + cols * col_width
        table_bottom = table_top + rows * row_height

        # 如果框完全在表格外，直接返回空列表
        if x2 < table_left or x1 > table_right or y2 < table_top or y1 > table_bottom:
            print('超出范围')
            return [], []

        # 获取可能的行列范围（扩大一个单元格的搜索范围）
        row_start = max(0, int((y1 - table_top) // row_height) - 1)
        row_end = min(rows - 1, int(np.ceil((y2 - table_top) / row_height)))
        col_start = max(0, int((x1 - table_left) // col_width) - 1)
        col_end = min(cols - 1, int(np.ceil((x2 - table_left) / col_width)))

        # 首先计算所有相交的单元格和它们的面积比
        all_intersections = []
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                # 计算单元格的坐标
                cell_x1 = table_left + col * col_width
                cell_y1 = table_top + row * row_height
                cell_x2 = cell_x1 + col_width
                cell_y2 = cell_y1 + row_height

                # 计算面积比
                area_ratio = self.calculate_overlap_ratio(
                    x1, y1, x2, y2,
                    cell_x1, cell_y1, cell_x2, cell_y2,
                    method='area_ratio'
                )

                if area_ratio > keep_threshold:
                    all_intersections.append((row, col, area_ratio))

        # 根据相交情况进行判断
        if not all_intersections:
            return [], []

        # 按面积比降序排序
        all_intersections.sort(key=lambda x: x[2], reverse=True)

        # 动态确定判断标准
        intersecting_cells = len(all_intersections)
        cell_positions = []
        area_ratios = []

        if intersecting_cells >= 4:
            # 一拖四的情况
            significant_cells = sum(1 for _, _, ratio in all_intersections if ratio >= 0.1)
            if significant_cells >= 3:
                # 保持一拖四
                threshold = 0.1*0.8
            elif significant_cells >= 2:
                # 退化为一拖二
                threshold = 0.25
                all_intersections = all_intersections[:2]  # 只保留最大的两个
            else:
                # 退化为一拖一
                threshold = 0.4
                all_intersections = all_intersections[:1]  # 只保留最大的一个

        elif intersecting_cells >= 2:
            # 一拖二的情况
            significant_cells = sum(1 for _, _, ratio in all_intersections if ratio >= 0.4)
            if significant_cells >= 2:
                # 保持一拖二
                threshold = 0.4
            else:
                # 退化为一拖一
                threshold = 0.6
                all_intersections = all_intersections[:1]  # 只保留最大的一个

        else:
            # 一拖一的情况
            threshold = 0.6

        # 应用最终的判断标准
        for row, col, ratio in all_intersections:
            if ratio >= threshold:
                cell_positions.append((row, col))
                area_ratios.append(ratio)

        return cell_positions, area_ratios

    def get_single_cell_position(self, x1, y1, x2, y2, top_pos, row_height, col_width, rows, cols):
        """
        获取框所在的单一格子位置，基于最大面积比
        """
        # 计算表格的边界
        table_left = top_pos[0]
        table_top = top_pos[1]
        table_right = table_left + cols * col_width
        table_bottom = table_top + rows * row_height

        # 如果框完全在表格外，直接返回空列表
        if x2 < table_left or x1 > table_right or y2 < table_top or y1 > table_bottom:
            print('超出范围')
            return None, 0.0

        # 获取可能的行列范围（扩大一个单元格的搜索范围）
        row_start = max(0, int((y1 - table_top) // row_height) - 1)
        row_end = min(rows - 1, int(np.ceil((y2 - table_top) / row_height)))
        col_start = max(0, int((x1 - table_left) // col_width) - 1)
        col_end = min(cols - 1, int(np.ceil((x2 - table_left) / col_width)))

        # 计算所有相交的单元格和它们的面积比
        max_area_ratio = 0.0
        best_cell = None
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                # 计算单元格的坐标
                cell_x1 = table_left + col * col_width
                cell_y1 = table_top + row * row_height
                cell_x2 = cell_x1 + col_width
                cell_y2 = cell_y1 + row_height

                # 计算面积比
                area_ratio = self.calculate_overlap_ratio(
                    x1, y1, x2, y2,
                    cell_x1, cell_y1, cell_x2, cell_y2,
                    method='area_ratio'
                )

                # 更新最大面积比和对应的单元格
                if area_ratio > max_area_ratio:
                    max_area_ratio = area_ratio
                    best_cell = (row, col)

        return [best_cell],[max_area_ratio]

    def get_table_boxes(self, boxes):
        """
        可视化表格和框的位置，并显示IOU值
        """

        # 获取表格四个顶点
        top_left, top_right, bottom_left, bottom_right = self.get_table_corners(
            self.top_pos, self.new_row_height, self.col_width, self.rows, self.cols
        )
        best_cells=[]
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # 解决变换过来，点不再是左上角和右下角的问题
            (x1,y1),(x2,y2)=self.get_box_points((x1,y1),(x2,y2))
            cell_positions, area_ratios = self.get_single_cell_position(  #横这种方式下，cell_position_v3返回的是空的。
                x1, y1, x2, y2, self.top_pos, self.new_row_height,
                self.col_width, self.rows, self.cols
            )

            best_cells.append(cell_positions[0])


        ret=dict()
        ret['best_cells']=best_cells
        ret['col_width']=self.col_width
        ret['row_height']=self.new_row_height
        ret['rows']=self.rows
        ret['cols']=self.cols
        ret['top_left']=top_left
        # 返回有多少格子，
        return ret


class MarkerAffineClass(BaseAffineClass):
    """只处理锚点定位的类"""
    def __init__(self, config_path=None):
        super().__init__(config_path)

    def get_angle_via_quadxy(self,  quad_xyes=None,  tilt=False):
        """可视化锚点和框的位置"""
        angle_rad=None
        if quad_xyes is not None:
            angle_rad=np.full(len(quad_xyes),0.00000000)
        rec_centers=[]
        # 处理框的绘制
        width_a=np.zeros(len(quad_xyes))
        height_a=np.zeros(len(quad_xyes))

        if tilt and quad_xyes is not None: # 这里还有些问题，需要看看
            for i,quad_xy in enumerate(quad_xyes):
                quad_xy = np.array(quad_xy, dtype=np.float32)
                rectangle, one_angle_rad,(width,height) = process_quad_v2(quad_xy)
                width_a[i]=width
                height_a[i]=height
                rectangle=rectangle.tolist()
                rec_center=np.mean(rectangle,axis=0)
                rec_center=rec_center.tolist()
                # print('rec_center',rec_center)
                rec_centers.append(rec_center)
                # print('one_angle_rad',one_angle_rad)

                # print('one_angle_rad',one_angle_rad)
                angle_rad[i]=one_angle_rad
        width_ret=np.mean(width_a)
        height_ret=np.mean(height_a)
        # 返回角度
        return angle_rad,rec_centers,width_ret,height_ret





if __name__ == '__main__':


    # 获取框所在的格子并可视化
    # self.visualize_table_and_box(self.mark1_pos, x1, y1, x2, y2, top_pos, row_height, new_row_height, self.col_width, rows=rows,
    #                              cols=cols)
    # boxAffineClass=BoxAffineClass()
    # x1, y1, = 500, 370
    # x2, y2 = x1 + boxAffineClass.qr_height, y1 + boxAffineClass.qr_height  # 框的坐标
    # box=(x1,y1,x2,y2)
    # boxAffineClass.visualize_table_and_box(box)
    pass
