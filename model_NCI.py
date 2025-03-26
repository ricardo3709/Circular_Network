import torch
import torch.nn as nn
import numpy as np
from collections import deque
import csv

class DQN(nn.Module):
    # DQN模型定义部分保持不变
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class Q_Network(nn.Module):
    def __init__(self, batch_size, state_dim, action_dim, gamma, epsilon, 
                 epsilon_decay, epsilon_min, learning_rate, total_eps, 
                 sim_env, total_its, replay_buffer, eval_freq, update_freq,
                 save_freq):
        
        super(Q_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.total_eps = total_eps
        self.sim_env = sim_env
        self.tot_its = total_its
        self.replay_buffer = replay_buffer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.update_freq = update_freq
        self.log_path = 'logs/training_log.txt'
        self.loss_path = 'logs/loss_log.csv'
        self.action_state_tuple_list = []

        self.states = np.zeros((self.tot_its, state_dim))
        self.actions = np.zeros(self.tot_its)

        self.weight_init()

    def weight_init(self):
        """
        Initialize the weights of the policy and target networks using Xavier initialization
        to prevent initial bias towards specific actions.
        """
        for module in self.policy_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
        
        # Copy the initialized weights to the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def to(self, device):
        """
        将模型移动到指定设备
        """
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        return self

    def eval(self, device):
        """
        评估当前模型的性能
        """
        total_reward = 0
        action_sum = 0
        
        print(f'Evaluation')
        for _ in range(100):
            # 初始化环境
            state = self.sim_env.reset()
            
            # 创建历史状态缓冲区，初始化为当前状态的复制
            state_history = deque(maxlen=3)  # 存储3个历史状态
            for _ in range(3):
                state_history.append(state.copy())
            
            # 拼接状态
            concatenated_state = np.concatenate([state] + list(state_history))
            concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32).to(device)
            
            episode_reward = 0
            
            for it in range(self.tot_its):
                # 选择动作
                with torch.no_grad():
                    q_values = self.policy_net(concatenated_state)
                    action = q_values.argmax().item()
                    action_sum += action
                
                # 执行动作
                next_state, reward, done = self.sim_env.step(action)
                episode_reward += reward
                
                # 更新历史状态
                state_history.append(state.copy())
                
                # 更新当前状态
                state = next_state
                concatenated_state = np.concatenate([state] + list(state_history))
                concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32).to(device)
            
            total_reward += episode_reward
        
        percentage_of_non_greedy_actions = action_sum / (100 * self.tot_its)
        avg_reward = total_reward / 100

        return avg_reward, percentage_of_non_greedy_actions

    def test(self, req_list, policy=None, device="cpu"):
        """
        使用给定的请求列表测试模型
        """
        # 设置随机种子
        np.random.seed(0)
        torch.manual_seed(0)

        # 初始化环境
        state = self.sim_env.reset()
        
        # 创建历史状态缓冲区，初始化为当前状态的复制
        state_history = deque(maxlen=3)
        for _ in range(3):
            state_history.append(state.copy())
        
        # 拼接状态
        concatenated_state = np.concatenate([state] + list(state_history))
        concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32).to(device)
        
        total_reward = 0
        total_action = 0

        for it in range(self.tot_its):
            # 选择动作
            if policy is not None:
                action = policy(concatenated_state)
            else:
                with torch.no_grad():
                    q_values = self.policy_net(concatenated_state)
                    action = q_values.argmax().item()
                    total_action += action

            # 使用预定义的请求位置执行动作
            req_position = req_list[it]
            next_state, reward, done = self.sim_env.step(action, req_position)
            
            # 更新历史状态
            state_history.append(state.copy())
            
            # 更新当前状态
            state = next_state
            concatenated_state = np.concatenate([state] + list(state_history))
            concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32).to(device)
            
            total_reward += reward

        percentage_of_non_greedy_actions = total_action / self.tot_its

        return total_reward, percentage_of_non_greedy_actions
                    
    def save(self, file_name):
        """
        保存模型参数到文件
        """
        torch.save(self.policy_net.state_dict(), 'saved_models/' + file_name+'_policy')
        torch.save(self.target_net.state_dict(), 'saved_models/' + file_name+'_target')
    
    def load(self, file_name):
        """
        从文件加载模型参数
        """
        policy_state_dict = torch.load('saved_models/' + file_name+'_policy', map_location=torch.device('cpu'))
        target_state_dict = torch.load('saved_models/' + file_name+'_target', map_location=torch.device('cpu'))
        
        self.policy_net.load_state_dict(policy_state_dict)
        self.target_net.load_state_dict(target_state_dict)