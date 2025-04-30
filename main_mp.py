import numpy as np
from simulator import Simulator
from replay_buffer import ReplayBuffer
import csv
from PPO_Model import PPO
import torch
from model import Q_Network
import random
# from model_res_DQN import Q_Network
import torch.multiprocessing as mp
import argparse

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def collect_data(args, rank):
    print("---------------------------------------")
    print(f"Process {rank} - Collecting Data")
    print("---------------------------------------")

    process_seed = args.seed + rank * 1000
    torch.manual_seed(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)

    # HyperParameter for DQN
    n_vehs = args.n_vehs
    batch_size = args.batch_size
    state_dim = args.state_dim
    action_dim = args.action_dim
    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    epsilon_min = args.epsilon_min
    learning_rate = args.learning_rate
    total_eps = args.total_eps
    n_slots = args.n_slots
    sim_env = Simulator(n_vehs, n_slots)
    total_its = args.total_its
    eval_freq = args.eval_freq
    update_freq = args.update_freq
    save_freq = args.save_freq

    replay_buffer = ReplayBuffer(int(update_freq * total_its)) # Replay buffer size equals to the number of its between updates

    model_path = f'temp_model/Circular_DQN.pth'
    model = Q_Network.load(model_path)

    # actually, the speed of collecting data is not the bottleneck
    # pause here, will continue if we really need mp. 


def main():
    # clear the content of logs
    with open('logs/training_log.txt', 'w') as f:
        f.write('')

    # delete the content of loss_log.csv
    with open('logs/loss_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Avg_Loss', 'Reward', 'Epsilon'])

    # HyperParameter for DQN
    n_vehs = 100
    batch_size = 64
    state_dim = (n_vehs*2+1+4)*4 # 10 vehicels, 10 gaps, 1 request position, gaps mean, gaps variance, 2 closest vehicles distance to request, 4 cat states
    action_dim = 2 # 0 or 1
    gamma = 0.999
    epsilon = 1.0
    epsilon_decay = 0.9999
    epsilon_min = 0.10
    learning_rate = 3e-4
    total_eps = 20001 # Total simulation episodes
    n_slots = 1000 # Number of slots in the ring, must be 10^n
    sim_env = Simulator(n_vehs, n_slots)
    total_its = 1000 # Total iterations per episode
    eval_freq = 50 # Evaluate the model every 100 episodes
    update_freq = 10 # Update the target network every 2 episodes
    save_freq = 50 # Save the model every 1000 episodes

    replay_buffer = ReplayBuffer(int(2e7))
    # replay_buffer = ReplayBuffer(1000)

    # torch random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # DQN Model
    model = Q_Network(batch_size, state_dim, action_dim, gamma, epsilon, epsilon_decay, 
                      epsilon_min, learning_rate, total_eps, sim_env, total_its, 
                      replay_buffer, eval_freq, update_freq, save_freq)
    # Train the model
    model.train()

    # # # PPO Model
    # # lr_actor = 1e-4
    # # lr_critic = 3e-4
    # # state_dim = 48 #discrete state
    # # ppo_agent = PPO(
    # #     state_dim, action_dim, lr_actor, lr_critic, gamma, gae_lambda=0.97,
    # #     policy_clip=0.2, batch_size=256, n_epochs=10, entropy_coef=0.05,
    # #     sim_env=sim_env, total_its=total_its, eval_freq=500, save_freq=500
    # # )

    # # 训练模型
    # ppo_agent.train(num_episodes=total_eps)


if __name__ == "__main__":
    main()
