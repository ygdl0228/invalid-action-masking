# @Time    : 2023/4/5 20:56
# @Author  : ygd
# @FileName: test.py
# @Software: PyCharm

# @Time    : 2023/4/3 14:37
# @Author  : ygd
# @FileName: env.py
# @Software: PyCharm

import math
import matplotlib.pyplot as plt
import matplotlib
import json


class route:
    def __init__(self, road_length, YC_interval, QC_interval, buffer_length, node_nums_x, node_nums_YC,
                 node_nums_QC, max_speed, min_speed, acceleration):
        self.road_length = road_length
        self.YC_interval = YC_interval
        self.QC_interval = QC_interval
        self.buffer_length = buffer_length
        self.node_nums_x = node_nums_x
        self.node_nums_YC = node_nums_YC
        self.node_nums_QC = node_nums_QC
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.acceleration = acceleration
        self.cur_time = 0
        self.move_direction = {'up': [0, 1], 'down': [0, -1], 'right': [1, 0], 'left': [-1, 0]}
        self.clf_fig = 'data.json'
        self.nodes = {}
        self.nodes_QC = {}
        self.nodes_YC = {}
        self.nodes_buffer = {}
        self.edges = []
        self.AGV = {}
        self.AGV_info = {'loc': []}
        self.speed = 0

    def creat_map(self, draw_arrow):
        node_nums = 1
        # YC node
        for j in range(self.node_nums_YC):
            for i in range(self.node_nums_x):
                self.nodes_YC[node_nums] = [i * self.road_length, j * self.YC_interval]
                plt.plot(i * self.road_length, j * self.YC_interval, 'o', color='b')
                # plt.text(i * road_length + 0.5, j * YC_interval - 0.5, f"{node_nums}", fontsize=10)
                node_nums += 1

        # YC最高点
        MAX_YC_Y = sorted(self.nodes_YC.items(), key=lambda x: x[1][1], reverse=True)[0][1][1]

        # QC node
        for j in range(self.node_nums_QC):
            for i in range(self.node_nums_x):
                self.nodes_QC[node_nums] = [i * self.road_length, j * self.QC_interval + MAX_YC_Y + self.buffer_length]
                plt.plot(i * self.road_length, j * self.QC_interval + MAX_YC_Y + self.buffer_length, 'o', color='r')
                # plt.text(i * road_length + 0.5, j * QC_interval - 0.5, f"{node_nums}", fontsize=10)
                node_nums += 1

        self.nodes.update(self.nodes_YC)
        self.nodes.update(self.nodes_QC)

        # YC从左到右
        for i in range(1, len(self.nodes_YC), self.node_nums_x):
            for j in range(i, self.node_nums_x + i - 1):
                start = self.nodes_YC[j]
                end = self.nodes_YC[j + 1]
                self.edges.append([start, end])

        # YC双向
        for i in range(1, self.node_nums_x + 1):
            for j in range(self.node_nums_YC - 1):
                start = self.nodes_YC[i + self.node_nums_x * j]
                end = self.nodes_YC[i + self.node_nums_x * (j + 1)]
                self.edges.append([start, end])
                self.edges.append([end, start])

        # YC最大坐标序号的值
        MAX_YC_Y_NODES = sorted(self.nodes_YC.items(), key=lambda x: x[0], reverse=True)[0][0]

        # QC从右到左
        for i in range(self.node_nums_QC):
            for j in range(i * self.node_nums_x + 1 + MAX_YC_Y_NODES,
                           self.node_nums_x + i * self.node_nums_x + MAX_YC_Y_NODES):
                end = self.nodes_QC[j]
                start = self.nodes_QC[j + 1]
                self.edges.append([start, end])

        # QC双向
        for i in range(1, self.node_nums_x + 1):
            for j in range(self.node_nums_YC,
                           self.node_nums_YC + self.node_nums_QC - 1):
                start = self.nodes_QC[i + self.node_nums_x * j]
                end = self.nodes_QC[i + self.node_nums_x * (j + 1)]
                self.edges.append([start, end])
                self.edges.append([end, start])

        # buffer
        # 双向
        for i in range(self.node_nums_x * (self.node_nums_YC - 1) + 1, self.node_nums_x * self.node_nums_YC + 1):
            start = self.nodes_YC[i]
            end = self.nodes_QC[i + self.node_nums_x]
            if i & 1:
                self.edges.append([start, end])
            else:
                self.edges.append([end, start])

        if draw_arrow:
            for start, end in self.edges:
                plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=0.2, head_length=0.2,
                          length_includes_head=True)
        else:
            for start, end in self.edges:
                plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                          length_includes_head=True)

        plt.axis('scaled')
        plt.show()


# road_length, YC_interval, QC_interval, buffer_length, node_nums_x, node_nums_YC,
# node_nums_QC, max_speed, min_speed, acceleration
if __name__ == '__main__':
    env = route(1, 1, 4, 8, 16, 6, 3, 10, 1, 1)
    env.creat_map('False')

# while 1:
#     try:
#         route.move_AGV(route.action_space()[0],1)
#     except:
#         break
# route.save_info()
