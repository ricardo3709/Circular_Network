import torch
import pickle
import torch.nn as nn
import numpy as np
# from tqdm import tqdm
from collections import deque

import csv

class DQN(nn.Module):
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
    
# class DQN_Target(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DQN_Target, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 32)
#         self.fc2 = nn.Linear(32, 32)
#         self.fc3 = nn.Linear(32, 32)
#         self.fc4 = nn.Linear(32, 32)
#         self.fc5 = nn.Linear(32, 16)
#         self.fc6 = nn.Linear(16, action_dim)
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
#         x = torch.relu(self.fc5(x))
#         x = self.fc6(x)
#         return x

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
                # Xavier/Glorot initialization for weights
                nn.init.xavier_uniform_(module.weight)
                # Initialize biases to small values close to zero
                nn.init.constant_(module.bias, 0.01)
        
        # Copy the initialized weights to the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        for ep in range(int(self.total_eps)):
            # 初始化环境
            state = self.sim_env.reset()
            
            state_history = deque([state.copy()]*3, maxlen=3)  # 存储3个历史状态

            total_reward = 0.0
            ep_losses = []

            print(f'Episode: {ep}')

            for it in range(self.tot_its):
                # [state_t] + [state_{t-1}, state_{t-2}, state_{t-3}]
                concat_state = np.concatenate([state] + list(state_history))
                state_tensor = torch.tensor(concat_state, dtype=torch.float32)
                
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, 2)
                else:
                    with torch.no_grad():
                        q_values = self.policy_net(state_tensor)
                        action = q_values.argmax().item()

                # 执行动作
                next_state, reward, done, is_greedy = self.sim_env.step(action)
                total_reward += reward

                state_history.appendleft(state.copy())  # [state_{t}, tate_{t-1}, state_{t-2}]

                next_concat = np.concatenate([next_state] + list(state_history)) # [state_{t+1}] + [state_{t}, tate_{t-1}, state_{t-2}]]
                next_tensor = torch.tensor(next_concat, dtype=torch.float32)

                self.replay_buffer.push(
                    state_tensor.numpy(),
                    action,
                    reward,
                    next_tensor.numpy(),
                    done
                )
                
                # 更新历史状态缓冲区
                state = next_state.copy()  # 更新当前状态
                

                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.tensor(states, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.int64)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.tensor(next_states, dtype=torch.float32)
                    dones = torch.tensor(dones, dtype=torch.float32)

                    # 当前Q值
                    q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    # 目标Q值
                    next_q_values = self.target_net(next_states).max(1).values
                    target_q_values = rewards + (1-dones)*self.gamma * next_q_values

                    # 计算损失
                    loss = self.loss(q_values, target_q_values)
                    ep_losses.append(loss.item())

                    # 每次迭代更新策略网络
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if done:
                    break
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 记录损失
            with open(self.loss_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep, np.mean(ep_losses), total_reward, self.epsilon])

            # 每update_freq轮更新目标网络
            if ep % self.update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 每eval_freq轮评估模型
            if ep % self.eval_freq == 0:
                avg_ep_reward, percentage_non_greedy = self.eval()  # 注意：eval方法也需要修改
                with open(self.log_path, 'a') as f:
                    f.write(f'Episode: {ep}, Episode Average Reward: {avg_ep_reward}, Non-Greedy:{percentage_non_greedy:.2%}, Loss:{np.mean(ep_losses)}\n')
            
            # 每save_freq轮保存模型
            if ep % self.save_freq == 0:
                self.save(f'Circular_DQN_{ep}')

    def eval(self):
        total_reward = 0
        action_sum = 0
        eval_ep = 50
        
        print(f'Evaluation')
        for _ in range(eval_ep):
            # 初始化环境
            state = self.sim_env.reset()
            
            # 创建历史状态缓冲区，初始化为当前状态的复制
            state_history = deque(maxlen=3)  # 存储3个历史状态
            for _ in range(3):
                state_history.append(state.copy())
            
            # 拼接状态
            concatenated_state = np.concatenate([state] + list(state_history))
            concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32)
            
            episode_reward = 0
            done = False
            
            for it in range(self.tot_its):
                # 选择动作
                with torch.no_grad():
                    q_values = self.policy_net(concatenated_state)
                    action = q_values.argmax().item()
                
                # 执行动作
                next_state, reward, done, is_greedy = self.sim_env.step(action)
                episode_reward += reward
                
                # 更新历史状态
                state_history.append(state.copy())
                
                # # 更新当前状态
                # state = next_state
                # concatenated_state = np.concatenate([state] + list(state_history))
                # concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32)
                # 创建下一个拼接状态
                next_concatenated_state = np.concatenate([next_state] + list(state_history))
                next_concatenated_state = torch.tensor(next_concatenated_state, dtype=torch.float32) 

                # 更新当前状态
                state = next_state
                concatenated_state = next_concatenated_state

                episode_reward += reward
                if not is_greedy:
                    action_sum += 1
                
            total_reward += episode_reward
        
        percentage_of_non_greedy_actions = action_sum / (eval_ep * self.tot_its)
        avg_reward = total_reward / eval_ep

        return avg_reward, percentage_of_non_greedy_actions

    def test(self, req_list, policy=None):
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
        concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32)
        
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

            # 使用预定义的请求位置执行动作
            req_position = req_list[it]
            next_state, reward, done, is_greedy = self.sim_env.step(action, req_position)
            if not is_greedy:
                total_action += 1
            # 更新历史状态
            state_history.append(state.copy())
            
            # 更新当前状态
            state = next_state
            concatenated_state = np.concatenate([state] + list(state_history))
            concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32)
            
            total_reward += reward

        percentage_of_non_greedy_actions = total_action / self.tot_its

        return total_reward, percentage_of_non_greedy_actions
                    
    def save(self, file_name):
        torch.save(self.policy_net.state_dict(), 'saved_models/' + file_name+'_policy')
        torch.save(self.target_net.state_dict(), 'saved_models/' + file_name+'_target')
    
    def load(self, file_name):
        self.policy_net.load_state_dict(torch.load('saved_models/' + file_name+'_policy', weights_only=True))
        self.target_net.load_state_dict(torch.load('saved_models/' + file_name+'_target', weights_only=True))

    def train_mp(self):
        """
        Train the model using multiprocessing
        """
        
        for ep in range(int(self.total_eps)):
            # 初始化环境
            state = self.sim_env.reset()
            
            # 创建历史状态缓冲区，初始化为当前状态的复制
            state_history = deque(maxlen=3)  # 只需存储3个历史状态，与当前状态一起就是4步
            for _ in range(3):
                state_history.append(state.copy())  # 用当前状态填充历史
            
            # 拼接状态
            concatenated_state = np.concatenate([state] + list(state_history))
            concatenated_state = torch.tensor(concatenated_state, dtype=torch.float32)
            
            total_reward = 0
            done = False
            it = 0
            ep_losses = []
            print(f'Episode: {ep}')\

            # for it in tqdm(range(self.tot_its)):
            for it in range(self.tot_its):
                # if it%100 == 0:
                #     print(f'Iteration: {it}')
                # 选择动作
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, 2)
                else:
                    with torch.no_grad():
                        q_values = self.policy_net(concatenated_state)
                        action = q_values.argmax().item()

                # 执行动作
                next_state, reward, done = self.sim_env.step(action)
                
                # 更新历史状态缓冲区
                state_history.append(state.copy())  # 添加当前状态到历史
                
                # 创建下一个拼接状态
                next_concatenated_state = np.concatenate([next_state] + list(state_history))
                next_concatenated_state = torch.tensor(next_concatenated_state, dtype=torch.float32)
                
                total_reward += reward
                
                # 存储经验
                self.replay_buffer.push(
                    concatenated_state.numpy(), 
                    action, 
                    reward, 
                    next_concatenated_state.numpy(), 
                    done
                )
                
                # 更新当前状态
                state = next_state
                concatenated_state = next_concatenated_state

                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.tensor(states, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.int64)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.tensor(next_states, dtype=torch.float32)
                    dones = torch.tensor(dones, dtype=torch.float32)

                    # 当前Q值
                    q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    # 目标Q值
                    next_q_values = self.target_net(next_states).max(1).values
                    target_q_values = rewards + (1-dones)*self.gamma * next_q_values

                    # 计算损失
                    loss = self.loss(q_values, target_q_values)
                    ep_losses.append(loss.item())

                    # 每次迭代更新策略网络
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if it == self.tot_its:
                    done = True
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 记录损失
            with open(self.loss_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep, np.mean(ep_losses), total_reward, self.epsilon])

            # 每update_freq轮更新目标网络
            if ep % self.update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 每eval_freq轮评估模型
            if ep % self.eval_freq == 0:
                total_reward, percentage_non_greedy = self.eval()  # 注意：eval方法也需要修改
                with open(self.log_path, 'a') as f:
                    f.write(f'Episode: {ep}, Reward: {total_reward}, Non-Greedy:{percentage_non_greedy:.2%}, Loss:{np.mean(ep_losses)}\n')
            
            # 每save_freq轮保存模型
            if ep % self.save_freq == 0:
                self.save(f'Circular_DQN_{ep}')