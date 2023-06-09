# @Time    : 2023/4/3 14:37
# @Author  : ygd
# @FileName: env.py
# @Software: PyCharm

import math
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from agent import DQN
from agent import ReplayBuffer


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
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.clf_fig = 'data.json'
        self.nodes = {}
        self.nodes_reverse = {}
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
                self.edges.append([end, start])
            else:
                self.edges.append([start, end])

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

    def AGV_get_task(self):
        self.AGV['start'] = np.random.choice(list(self.nodes.keys())[-self.node_nums_x:])
        self.AGV['end'] = np.random.choice(list(self.nodes.keys())[:self.node_nums_x])
        self.AGV['loc'] = self.nodes[self.AGV['start']]
        self.AGV['destination'] = self.nodes[self.AGV['end']]
        self.AGV['speed'] = 1
        self.AGV['inter'] = [1, 1]
        print(f'起点：', self.AGV['start'], f'终点：', self.AGV['end'])

    def reset(self):
        self.creat_map('False')
        self.AGV_get_task()
        return self.env_infor()

    def distance_angle(self, origin, destination):

        x1, y1 = origin
        x2, y2 = destination
        angle = 0.0
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        return [angle * 180 / math.pi, abs(x1 - x2) + abs(y1 - y2)]

    # 方向 加速减速还是匀速 ad=[-1,1]
    def move_AGV(self, action, ad):
        self.AGV_info['loc'].append(self.AGV['loc'])
        self.AGV['speed'], dis = self.distance(self.AGV['speed'], ad)
        dx = self.move_direction[action][0] * dis
        dy = self.move_direction[action][1] * dis
        self.cur_time += 1
        self.AGV['loc'] = [self.AGV['loc'][0] + dx, self.AGV['loc'][1] + dy]

    def save_info(self):
        json_data = json.dumps(self.AGV_info)
        with open('data.json', 'w+') as f:
            f.write(json_data)

    def step(self, action):
        done = False
        action_dir = self.action_map[action]
        self.move_AGV(action_dir, 0)
        reward = -1 - self.env_infor()[1]
        if self.AGV['loc'] == self.AGV['destination']:
            done = True
        return self.env_infor(), reward, done

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

    def get_avail_agent_action(self):
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
        avail_action_mask = [1, 1, 1, 1]

        if 'up' not in action_space:
            avail_action_mask[0] = 0
        if 'down' not in action_space:
            avail_action_mask[1] = 0
        if 'left' not in action_space:
            avail_action_mask[2] = 0
        if 'right' not in action_space:
            avail_action_mask[3] = 0
        return action_space, avail_action_mask

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

    def env_infor(self):
        state = self.distance_angle(self.AGV['loc'], self.AGV['destination']) + self.AGV['loc'] + self.AGV[
            'destination']
        return state

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


'''road_length, YC_interval, QC_interval, buffer_legth, node_nums_x, node_nums_YC,
                 node_nums_QC, max_speed, min_speed, acceleration'''
if __name__ == '__main__':
    lr = 1e-2
    lr_decay = 0.97
    min_lr = 1e-7
    num_episodes = 50000
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.8
    epsilon_decay = 0.97
    min_epsilon = 1e-4
    target_update = 10
    buffer_size = 1000000000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = 6
    action_dim = 4
    dqn_type = 'D3QN'
    agent = DQN(state_dim, hidden_dim, action_dim, lr, lr_decay, min_lr, gamma, epsilon, epsilon_decay, min_epsilon,
                target_update, device, dqn_type)

    return_list = []
    for eps in range(num_episodes):
        print(f'第{eps + 1}次训练')
        episode_return = 0
        env = route(1, 1, 1, 2, 16, 6, 3, 10, 1, 1)
        state = env.reset()
        done = False
        while not done:
            avail_action, avail_action_mask = env.get_avail_agent_action()
            action = agent.take_action(state, avail_action, eps)
            next_state, reward, done = env.step(action)
            _, next_avail_action_mask = env.get_avail_agent_action()
            replay_buffer.add(state, action, reward, next_state, next_avail_action_mask, done)
            state = next_state
            episode_return += reward
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_naam, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'next_avail_action_mask': b_naam,
                    'dones': b_d
                }
                agent.update(transition_dict, eps)
        return_list.append(episode_return)
        print(return_list)
        plt.plot(return_list)
        plt.show()
        plt.savefig('return_list')



