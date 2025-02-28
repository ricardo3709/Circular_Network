import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) # Q values 
        return x
    
class DQN_Target(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_Target, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Q_Network(nn.Module):
    def __init__(self, batch_size, state_dim, action_dim, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, total_eps):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN_Target(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.total_eps = total_eps
        

    def train(self):
        for ep in range(int(self.total_eps)):
            state = self.sim_env.reset()
            total_reward = 0
            done = False
            step = 0

            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, 2)
                else:
                    action = self.policy_net(state)
                next_state, reward, done = self.sim_env.step(action)
                total_reward += reward
                self.replay_buffer.push(state, action, reward, next_state)
                state = next_state
                step += 1
                if len(self.replay_buffer) > self.batch_size:
                    self.update()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode: {ep}, Total Reward: {total_reward}")
