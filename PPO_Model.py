import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from tqdm import tqdm
import csv
import pickle
import time
from collections import deque

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        residual = x  # 保存输入用于后续相加
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual  # 残差连接
        out = F.relu(out)
        return out
    
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.res1 = ResidualBlock(512)

        self.fc2 = nn.Linear(512, 256)
        self.res2 = ResidualBlock(256)

        self.fc3 = nn.Linear(256, 128)
        self.res3 = ResidualBlock(128)

        self.fc4 = nn.Linear(128, 64)

        self.fc5 = nn.Linear(64, action_dim)
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res1(x)

        x = F.relu(self.fc2(x))
        x = self.res2(x)

        x = F.relu(self.fc3(x))
        x = self.res3(x)

        x = F.relu(self.fc4(x))

        return F.softmax(self.fc5(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.res1 = ResidualBlock(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.res2 = ResidualBlock(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.res3 = ResidualBlock(128)
        
        self.fc4 = nn.Linear(128, 64)
        # 最后一层输出单个标量
        self.fc5 = nn.Linear(64, 1)

        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res1(x)
        
        x = F.relu(self.fc2(x))
        x = self.res2(x)
        
        x = F.relu(self.fc3(x))
        x = self.res3(x)
        
        x = F.relu(self.fc4(x))
        # 输出状态值，不经过激活函数
        return self.fc5(x)
    
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        batches = [b for b in batches if len(b) > 1]
        
        return np.array(self.states), np.array(self.actions), \
               np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), batches
               
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class PPO:
    def __init__(self, state_dim, action_dim, 
                 lr_actor=0.0003, lr_critic=0.0003, 
                 gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, 
                 n_epochs=10, entropy_coef=0.01,
                 sim_env=None, total_its=1000, eval_freq=100, save_freq=100):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.memory = PPOMemory(batch_size)
        self.sim_env = sim_env
        self.total_its = total_its
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_path = 'logs/ppo_training_log.txt'
        self.loss_path = 'logs/ppo_loss_log.csv'
        
    def choose_action(self, observation):
        # 添加模式切换
        self.actor.eval()  # 切换到评估模式
        self.critic.eval()
        
        with torch.no_grad():  # 禁用梯度计算
            if not torch.is_tensor(observation):
                observation = torch.tensor(observation, dtype=torch.float)
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            
            state = observation
            probabilities = self.actor(state)
            value = self.critic(state)
        
        # 恢复训练模式
        self.actor.train()
        self.critic.train()
        
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def learn(self):
        actor_losses = []
        critic_losses = []
        entropies = []
        
        # Generate batches of experiences
        states, actions, old_log_probs, vals, rewards, dones, batches = self.memory.generate_batches()
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)
        vals = torch.tensor(vals, dtype=torch.float)
        
        # Calculate advantages
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_adv = 0
        for t in reversed(range(len(rewards)-1)):
            if dones[t]:
                last_adv = 0
            delta = rewards[t] + self.gamma * vals[t+1] * (1-dones[t]) - vals[t]
            advantages[t] = last_adv = delta + self.gamma * self.gae_lambda * (1-dones[t]) * last_adv
        
        # 归一化优势函数 (新增)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 修改后正确实现
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages_tensor + vals.squeeze()
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 重复训练n_epochs次
        for _ in range(self.n_epochs):
            # 每个epoch重新打乱batch顺序
            np.random.shuffle(batches)
            
            # Process each batch
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = torch.tensor(advantages[batch], dtype=torch.float32)
                batch_returns = returns[batch]
                
                # Get current action probabilities and values
                probabilities = self.actor(batch_states)
                critic_value = self.critic(batch_states).squeeze()
                
                # Create categorical distribution
                dist = torch.distributions.Categorical(probabilities)
                
                # Get new log probabilities for actions
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate entropy for the current policy
                entropy = dist.entropy().mean()
                
                # Calculate probability ratio
                prob_ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = batch_advantages * torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                
                # Calculate actor loss (negative because we want to maximize)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - self.entropy_coef * entropy
                
                # Calculate critic loss (MSE)
                # returns = batch_advantages + vals[batch]
                critic_loss = F.mse_loss(critic_value, batch_returns)
                
                # Update actor and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                actor_loss.backward()
                critic_loss.backward()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # Clear memory after learning
        self.memory.clear_memory()
    
        return actor_losses, critic_losses, entropies
    
    def train(self, num_episodes=10000):
        # Set up logging
        with open(self.log_path, 'w') as f:
            f.write('Starting PPO Training\n')
            
        with open(self.loss_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Reward', 'Non_Greedy_Percentage', 'Actor_Loss', 'Critic_Loss', 'Entropy'])
        
        # Set up environment and start training
        for episode in range(num_episodes):
            obs = self.sim_env.reset()
            done = False
            total_reward = 0
            non_greedy_count = 0

            obs_history = deque(maxlen=3)
            for _ in range(3):
                obs_history.append(obs.copy())

            concatenated_obs = np.concatenate([obs] + list(obs_history))
            concatenated_obs = torch.tensor(concatenated_obs, dtype=torch.float32)
            
            print(f'Episode: {episode}')
            
            # 收集一整个episode的数据
            # for t in tqdm(range(self.total_its)):
            for t in range(self.total_its):
                # if t%100 == 0:
                #     print(f'Iteration: {t}')
                # Choose action
                action, prob, val = self.choose_action(concatenated_obs)
                
                # Count non-greedy actions (action=1)
                if action == 1:
                    non_greedy_count += 1
                
                # Take action in environment
                next_obs, reward, done = self.sim_env.step(action)

                # print(f"reward: {reward}, action: {action}, done: {done}")

                # Update observation history
                obs_history.append(obs.copy())
                next_concatenated_obs = np.concatenate([next_obs] + list(obs_history))
                next_concatenated_obs = torch.tensor(next_concatenated_obs, dtype=torch.float32)
                
                # Store the transition
                self.memory.store_memory(concatenated_obs.numpy(), action, prob, val, reward, done)
                
                # Update observation and total reward
                obs = next_obs
                concatenated_obs = next_concatenated_obs
                total_reward += reward
            
            # 每个episode结束后进行一次学习
            ep_actor_losses, ep_critic_losses, ep_entropy = self.learn()
            
            # Calculate percentage of non-greedy actions
            non_greedy_percentage = non_greedy_count / self.total_its

            # Calculate Avg Losses
            avg_actor_loss = np.mean(ep_actor_losses)
            avg_critic_loss = np.mean(ep_critic_losses)
            avg_entropy = np.mean(ep_entropy)
            
            # Log results
            with open(self.loss_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_reward, non_greedy_percentage, avg_actor_loss, avg_critic_loss, avg_entropy])
                
            
            # Evaluate the model
            if episode % self.eval_freq == 0:
                eval_reward, eval_non_greedy = self.eval()
                eval_reward /= self.total_its
                with open(self.log_path, 'a') as f:
                    f.write(f'Episode: {episode}, Avg Reward: {eval_reward}, Non-Greedy: {eval_non_greedy:.2%}\n')
            
            # Save the model
            if episode % self.save_freq == 0:
                self.save(f'Circular_PPO_{episode}')
    
    def eval(self):
        """Evaluate the current policy"""
        self.actor.eval()  # 新增评估模式切换
        self.critic.eval()
        
        obs = self.sim_env.reset()
        total_reward = 0
        non_greedy_count = 0

        eval_runs = 10
        
        print('Evaluation')
        # for _ in tqdm(range(10)):  # 10 evaluation runs
        for run in range(eval_runs):
            obs = self.sim_env.reset()
            obs_history = deque(maxlen=3)
            for _ in range(3):
                obs_history.append(obs.copy())
            concatenated_obs = np.concatenate([obs] + list(obs_history))
            concatenated_obs = torch.tensor(concatenated_obs, dtype=torch.float32)

            for t in range(self.total_its):
                with torch.no_grad():  
                    # 若输入为一维，则增加 batch 维度
                    if concatenated_obs.dim() == 1:
                        state_input = concatenated_obs.unsqueeze(0)
                    else:
                        state_input = concatenated_obs
                    probabilities = self.actor(state_input)
                    # 采用确定性策略：选择概率最大的动作
                    action = torch.argmax(probabilities, dim=1).item()
                    
                # Count non-greedy actions
                if action == 1:
                    non_greedy_count += 1
                
                # Take action
                next_obs, reward, _ = self.sim_env.step(action)
                total_reward += reward

                # Update observation history
                obs_history.append(obs.copy())
                obs = next_obs

                next_concatenated_obs = np.concatenate([next_obs] + list(obs_history))
                next_concatenated_obs = torch.tensor(next_concatenated_obs, dtype=torch.float32)
                concatenated_obs = next_concatenated_obs
        
        # Calculate average reward and non-greedy percentage
        avg_reward = total_reward / eval_runs
        non_greedy_percentage = non_greedy_count / (self.total_its * eval_runs)
        
        self.actor.train()  # 恢复训练模式
        self.critic.train()
        return avg_reward, non_greedy_percentage
    
    def test(self, req_list, policy=None):
        """Test the policy on a single episode"""
        np.random.seed(0)
        torch.manual_seed(0)


        obs = self.sim_env.reset()
        total_reward = 0
        non_greedy_count = 0

        obs_history = deque(maxlen=3)
        for _ in range(3):
            obs_history.append(obs.copy())
        
        concatenated_obs = np.concatenate([obs] + list(obs_history))
        concatenated_obs = torch.tensor(concatenated_obs, dtype=torch.float32)

        total_reward = 0
        total_action = 0
        
        for t in range(self.total_its):

            if policy is None:
                with torch.no_grad():
                    probabilities = self.actor(concatenated_obs)
                    action = torch.argmax(probabilities, dim=-1).item()
                    total_action += action
            else:
                action = policy(obs)
            
            req_position = req_list[t]
            next_obs, reward, _ = self.sim_env.step(action, req_position)

            obs_history.append(obs.copy())
            
            obs = next_obs
            concatenated_obs = np.concatenate([obs] + list(obs_history))
            concatenated_obs = torch.tensor(concatenated_obs, dtype=torch.float32)
            total_reward += reward
        
        percentage_of_non_greedy_actions = total_action / self.total_its
        
        return total_reward, percentage_of_non_greedy_actions
    
    def save(self, filename):
        """Save model parameters"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, f'saved_models/{filename}.pt')
    
    def load(self, filename):
        """Load model parameters"""
        checkpoint = torch.load(f'saved_models/{filename}.pt')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

