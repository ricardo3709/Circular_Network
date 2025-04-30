import numpy as np
from simulator import Simulator
from replay_buffer import ReplayBuffer
import csv
from PPO_Model import PPO
import torch
from model import Q_Network

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def main():
    # clear the content of logs
    with open('logs/training_log.txt', 'w') as f:
        f.write('')

    # delete the content of loss_log.csv
    with open('logs/loss_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Avg_Loss', 'Reward', 'Epsilon'])

    # HyperParameter for DQN
    n_vehs = 50
    batch_size = 128
    state_dim = (n_vehs*2+1+4)*4 # 10 vehicels, 10 gaps, 1 request position, gaps mean, gaps variance, 2 closest vehicles distance to request, 4 cat states
    action_dim = 2 # 0 or 1
    gamma = 0.99
    total_eps = 10001 # Total simulation episodes
    n_slots = 1000 # Number of slots in the ring, must be 10^n
    sim_env = Simulator(n_vehs, n_slots)
    total_its = 1000 # Total iterations per episode

    # Set torch random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # PPO Model
    lr_actor = 1e-3
    lr_critic = 3e-3
    state_dim = (n_vehs*2+1+4)*4 
    
    # 1/(1-gamma*gae_lambda) = 1/(1-0.999*0.999) = 500 steps to consider
    ppo_agent = PPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma, gae_lambda=0.99,
        policy_clip=0.1, batch_size=batch_size, n_epochs=5, entropy_coef=0.001,
        sim_env=sim_env, total_its=total_its, eval_freq=50, save_freq=50
    )

    # 训练模型
    ppo_agent.train(num_episodes=total_eps)


if __name__ == "__main__":
    main()
