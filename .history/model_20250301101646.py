import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

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
        x = self.fc4(x) # Q values for each action (2)
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
    def __init__(self, batch_size, state_dim, action_dim, gamma, epsilon, 
                 epsilon_decay, epsilon_min, learning_rate, total_eps, 
                 sim_env, total_its, replay_buffer, eval_freq):
        
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
        self.sim_env = sim_env
        self.tot_its = total_its
        self.replay_buffer = replay_buffer
        self.eval_freq = eval_freq


    def train(self):
        for ep in range(int(self.total_eps)):
            state = self.sim_env.reset()
            total_reward = 0
            done = False
            it = 0
            print(f'Episode: {ep}')

            for it in tqdm(range(self.tot_its)):
            # while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, 2)
                else:
                    with torch.no_grad():
                        q_values = self.policy_net(state)
                        action = q_values.argmax().item()

                next_state, reward, done = self.sim_env.step(action)
                total_reward += reward
                self.replay_buffer.push(state, action, reward, next_state)
                state = next_state

                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    states, actions, rewards, next_states = zip(*batch)

                    states = torch.tensor(states, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.int64)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.tensor(next_states, dtype=torch.float32)

                    # current Q values
                    q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    # target Q values
                    next_q_values = self.target_net(next_states).max(1).values
                    target_q_values = rewards + self.gamma * next_q_values

                    # loss
                    loss = self.loss(q_values, target_q_values)

                    # update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # it += 1
                # if it > self.tot_its:
                #     done = True
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if ep % 100 == 0:
                print(f'Episode: {ep}, Total Reward: {total_reward}')

    def eval(self):
        state = self.sim_env.reset()
        total_reward = 0
        done = False

        print(f'Evaluation')
        for it in tqdm(range(self.tot_its)):
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
            next_state, reward, done = self.sim_env.step(action)
            total_reward += reward
            state = next_state

        return total_reward
    
    def save(self, file_name):
        torch.save(self.policy_net.state_dict(), file_name+'_policy')
        torch.save(self.target_net.state_dict(), file_name+'_target')
    
    def load(self, file_name):
        self.policy_net.load_state_dict(torch.load(file_name+'_policy', weights_only=True))
        self.target_net.load_state_dict(torch.load(file_name+'_target', weights_only=True))
