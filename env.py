# @Time    : 2023/4/3 14:37
# @Author  : ygd
# @FileName: env.py
# @Software: PyCharm

import math
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use("TkAgg")


class route:
    def __init__(self, road_length, YC_interval, QC_interval, buffer_legth, node_nums_x, node_nums_YC, node_nums_buffer,
                 node_nums_QC, max_speed, min_speed, acceleration):
        self.road_length = road_length
        self.YC_interval = YC_interval
        self.QC_interval = QC_interval
        self.buffer_legth = buffer_legth
        self.node_nums_x = node_nums_x
        self.node_nums_YC = node_nums_YC
        self.node_nums_buffer = node_nums_buffer
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
        self.AGV_info = {'loc':[]}
        self.speed = 0

    def creat_map(self, draw_arrow):
        node_nums = 1
        # YC
        for j in range(self.node_nums_YC):
            for i in range(self.node_nums_x):
                self.nodes_YC[node_nums] = [i * self.road_length, j * self.YC_interval]
                # plt.plot(i * self.road_length, j * self.YC_interval, 'o', color='black')
                # plt.text(i * self.road_length + 0.5, j * self.YC_interval - 0.5, f"{node_nums}", fontsize=10)
                node_nums += 1

        # buffer
        for j in range(self.node_nums_YC + 1, self.node_nums_buffer + self.node_nums_YC + 1):
            for i in range(self.node_nums_x):
                self.nodes_buffer[node_nums] = [i * self.road_length, j * self.buffer_legth]
                # plt.plot(i * self.road_length, j * self.buffer_legth, 'o', color='red')
                # plt.text(i * self.road_length + 0.5, j * self.buffer_legth - 0.5, f"{node_nums}", fontsize=10)
                node_nums += 1

        # QC
        for j in range(self.node_nums_YC + self.node_nums_buffer + 1,
                       self.node_nums_QC + self.node_nums_buffer + self.node_nums_YC + 1):
            for i in range(self.node_nums_x):
                self.nodes_QC[node_nums] = [i * self.road_length, j * self.QC_interval]
                # plt.plot(i * self.road_length, j * self.QC_interval, 'o', color='blue')
                # plt.text(i * self.road_length + 0.5, j * self.QC_interval - 0.5, f"{node_nums}", fontsize=10)
                node_nums += 1

        self.nodes.update(self.nodes_YC)
        self.nodes.update(self.nodes_buffer)
        self.nodes.update(self.nodes_QC)
        # YC
        # 从左到右
        for i in range(1, len(self.nodes_YC), self.node_nums_x):
            for j in range(i, self.node_nums_x + i - 1):
                start = self.nodes_YC[j]
                end = self.nodes_YC[j + 1]
                self.edges.append([start, end])

        # 双向
        for i in range(1, self.node_nums_x + 1):
            for j in range(self.node_nums_YC - 1):
                start = self.nodes_YC[i + self.node_nums_x * j]
                end = self.nodes_YC[i + self.node_nums_x * (j + 1)]
                self.edges.append([start, end])
                self.edges.append([end, start])

        # YC buffer
        for i in range(1, self.node_nums_x + 1):
            for j in range(self.node_nums_YC - 1, self.node_nums_YC):
                if i & 1:
                    start = self.nodes_YC[i + self.node_nums_x * j]
                    end = self.nodes_buffer[i + self.node_nums_x * (j + 1)]
                    self.edges.append([start, end])

                else:
                    start = self.nodes_YC[i + self.node_nums_x * j]
                    end = self.nodes_buffer[i + self.node_nums_x * (j + 1)]
                    self.edges.append([end, start])

        # buffer
        # 双向
        for i in range(1, self.node_nums_x + 1):
            for j in range(self.node_nums_YC, self.node_nums_buffer + self.node_nums_YC - 1):
                start = self.nodes_buffer[i + self.node_nums_x * j]
                end = self.nodes_buffer[i + self.node_nums_x * (j + 1)]
                if i & 1:
                    self.edges.append([start, end])

                else:
                    self.edges.append([end, start])

        # buffer QC
        for i in range(1, self.node_nums_x + 1):
            for j in range(self.node_nums_YC + self.node_nums_buffer - 1, self.node_nums_YC + self.node_nums_buffer):
                if i & 1:
                    start = self.nodes_buffer[i + self.node_nums_x * j]
                    end = self.nodes_QC[i + self.node_nums_x * (j + 1)]
                    self.edges.append([start, end])

                else:
                    start = self.nodes_buffer[i + self.node_nums_x * j]
                    end = self.nodes_QC[i + self.node_nums_x * (j + 1)]
                    self.edges.append([end, start])

        # QC
        for i in range(len(self.nodes_QC) + len(self.nodes_buffer) + 1,
                       len(self.nodes_QC) + len(self.nodes_buffer) + len(self.nodes_YC),
                       self.node_nums_x):
            for j in range(i, self.node_nums_x + i - 1):
                start = self.nodes_QC[j]
                end = self.nodes_QC[j + 1]
                self.edges.append([start, end])

        # 双向
        for i in range(1, self.node_nums_x + 1):
            for j in range(self.node_nums_YC + self.node_nums_buffer,
                           self.node_nums_YC + self.node_nums_buffer + self.node_nums_QC - 1):
                start = self.nodes_QC[i + self.node_nums_x * j]
                end = self.nodes_QC[i + self.node_nums_x * (j + 1)]
                self.edges.append([start, end])
                self.edges.append([end, start])
        if draw_arrow == 'True':
            for start, end in self.edges:
                plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=2, head_length=5,
                          length_includes_head=True)
        elif draw_arrow == 'False':
            for start, end in self.edges:
                plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], length_includes_head=True)

        plt.xlim((-5, self.node_nums_x * self.road_length * 1.1))
        plt.ylim((-5, (
                self.node_nums_QC * self.QC_interval + self.node_nums_YC * self.YC_interval + self.node_nums_buffer * self.buffer_legth) * 1.2))
        plt.show()

    def AGV_get_task(self):
        self.AGV['start'] = 1
        self.AGV['end'] = 115
        self.AGV['loc'] = self.nodes[self.AGV['start']]
        self.AGV['speed'] = 1
        self.AGV['inter'] = [1, 1]

    # 方向 加速减速还是匀速 ad=[-1,1]
    def move_AGV(self, action, ad):
        print(self.cur_time)
        print(self.AGV)
        self.AGV_info['loc'].append(self.AGV['loc'])
        self.AGV['speed'], dis = self.distance(self.AGV['speed'], ad)
        dx = self.move_direction[action][0] * dis
        dy = self.move_direction[action][1] * dis
        self.cur_time += 1
        self.AGV['loc'] = [self.AGV['loc'][0] + dx, self.AGV['loc'][1] + dy]
        print(self.AGV_info)

    def save_info(self):
        json_data = json.dumps(self.AGV_info)
        with open('data.json', 'w+') as f:
            f.write(json_data)

    def distance(self, cur_v, ad):
        if self.AGV['speed'] + ad * self.acceleration >= self.min_speed and self.AGV[
            'speed'] + ad * self.acceleration <= self.max_speed:
            cur_v = self.AGV['speed']
            next_v = cur_v + ad * self.acceleration
        elif self.AGV['speed'] + ad * self.acceleration < self.min_speed:
            cur_v = self.AGV['speed']
            next_v = self.min_speed
        elif self.AGV['speed'] + ad * self.acceleration > self.max_speed:
            cur_v = self.AGV['speed']
            next_v = self.max_speed
        dis = (cur_v + next_v) / 2
        return next_v, dis

    def action_space(self):
        action_space = []
        if self.AGV['loc'] in self.nodes.values():
            self.AGV['inter'] = [self.get_keys(self.nodes, self.AGV['loc']) for _ in range(2)]
            for edge in self.edges:
                start_edge, end_edge = edge
                if start_edge == self.AGV['loc']:
                    action_space.append(self.run_direction(start_edge, end_edge))
        else:
            for edge in self.edges:
                if self.distance_to_line_segment(self.AGV['loc'], edge) == 0:
                    start_edge, end_edge = edge
                    self.AGV['inter'] = [self.get_keys(self.nodes, start_edge), self.get_keys(self.nodes, end_edge)]
                    action_space.append(self.run_direction(start_edge, end_edge))
                    continue
        return action_space

    def get_keys(self, d, value):
        for k, v in d.items():
            if v == value:
                return k

    def run_direction(self, start, end):
        if end[0] - start[0] == 0 and end[1] - start[1] > 0:
            return 'up'
        elif end[0] - start[0] == 0 and end[1] - start[1] < 0:
            return 'down'
        elif end[1] - start[1] == 0 and end[0] - start[0] > 0:
            return 'right'
        elif end[1] - start[1] == 0 and end[0] - start[0] < 0:
            return 'left'

    # def AGV_infor(self):

    # 判断点到直线的距离 从而判断AGV在哪条路段上
    def distance_to_line_segment(self, P, L):
        x0, y0 = P
        x1, y1 = L[0]
        x2, y2 = L[1]
        dx = x2 - x1
        dy = y2 - y1
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx ** 2 + dy ** 2)
        if t < 0:
            x, y = x1, y1
        elif t > 1:
            x, y = x2, y2
        else:
            x, y = x1 + t * dx, y1 + t * dy
        return math.sqrt((x - x0) ** 2 + (y - y0) ** 2)


'''road_length, YC_interval, QC_interval, buffer_legth, node_nums_x, node_nums_YC, node_nums_buffer,
                 node_nums_QC, max_speed, min_speed, acceleration'''
route=route(7,4,4,4,16,6,4,7,10,1,1)
route.creat_map('False')
route.AGV_get_task()

while 1:
    try:
        route.move_AGV(route.action_space()[0],1)
    except:
        break
route.save_info()






