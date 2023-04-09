# @Time    : 2023/4/3 14:37
# @Author  : ygd
# @FileName: main.py
# @Software: PyCharm

import numpy as np
import torch
from agent import ReplayBuffer,DQN
from env import route
import matplotlib.pyplot as plt
from datetime import datetime

def train():
    lr = 1e-2
    lr_decay = 0.97
    min_lr = 1e-7
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.9
    epsilon = 0.2
    epsilon_decay = 0.97
    min_epsilon = 1e-4
    target_update = 100
    buffer_size = 1000000000
    minimal_size = 100
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = 6
    action_dim = 4
    dqn_type = 'D3QN'
    output_dir= f'./output/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
    agent = DQN(state_dim, hidden_dim, action_dim, lr, lr_decay, min_lr, gamma, epsilon, epsilon_decay, min_epsilon,
                target_update, device, dqn_type,output_dir)
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

            torch.save(agent.q_net, agent.weight_dir+f'{eps}_model.pkl')
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

if __name__ == "__main__":
    train()
