# @Time    : 2023/4/3 14:37
# @Author  : ygd
# @FileName: agent.py
# @Software: PyCharm


import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, next_avail_action_mask, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, next_avail_action_mask, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_avail_action_mask, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), next_avail_action_mask, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.x1 = torch.nn.Linear(state_dim, 64)
        self.x2 = torch.nn.Linear(64, hidden_dim)
        self.x3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.x1(x)
        x = F.relu(x)
        x = self.x2(x)
        x = F.relu(x)
        x = self.x3(x)
        return x


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, lr_decay, min_lr, gamma,
                 epsilon, epsilon_decay, min_epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.gamma = gamma  # 折扣因子
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.min_epsilon = min_epsilon
        self.min_lr = min_lr
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def take_action(self, state, avail_action, eps):
        # epsilon-贪婪策略采取动作
        # up down left right
        avail_action_dim = []
        avail_action_mask = [1, 1, 1, 1]
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        q_values = self.q_net(state)
        if 'up' not in avail_action:
            avail_action_mask[0] = 0
            q_values[0][0] = -float('inf')
        if 'down' not in avail_action:
            avail_action_mask[1] = 0
            q_values[0][1] = -float('inf')
        if 'left' not in avail_action:
            avail_action_mask[2] = 0
            q_values[0][2] = -float('inf')
        if 'right' not in avail_action:
            avail_action_mask[3] = 0
            q_values[0][3] = -float('inf')
        for i in range(len(q_values[0])):
            if q_values[0][i] != -float('inf'):
                avail_action_dim.append(i + 1)
        if np.random.random() < max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** eps)):
            action = np.random.choice(avail_action_dim)
        else:
            action = q_values.argmax().item() + 1
        return action

    def update(self, transition_dict, eps):
        optimizer = torch.optim.Adam(self.q_net.parameters(),
                                     lr=max(self.min_lr, self.learning_rate * (self.lr_decay ** eps)))
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        next_avail_action_mask = torch.tensor(transition_dict['next_avail_action_mask'],
                                              dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions - 1)  # Q值
        # 下个状态的最大Q值
        q_target = self.target_q_net(next_states)

        for i in range(len(q_target)):
            for j in range(4):
                if next_avail_action_mask[i][j] == 0:
                    q_target[i][j] = -float('inf')

        # max(1) 返回每一行的最大值
        max_next_q_values = q_target.max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
