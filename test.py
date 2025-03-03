import numpy as np
from simulator import Simulator
from replay_buffer import ReplayBuffer
import csv
from model import Q_Network
import torch
from tqdm import tqdm

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    action = 0
    return action

def test():
     # HyperParameter for DQN
    sectors = 4 # Number of sectors of ring
    n_vehs = 10
    n_vehs_in_state = n_vehs
    batch_size = 64
    state_dim = sectors + n_vehs_in_state + 2 + 1 # 4 sectors + 10 vehicles + mean and std of vehicles + request position
    action_dim = 2 # 0 or 1
    gamma = 0.995
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    learning_rate = 1e-3
    total_eps = 1e5 # Total simulation episodes
    sim_env = Simulator(n_vehs, sectors, n_vehs_in_state)
    total_its = 1000 # Total iterations per episode
    eval_freq = 100 # Evaluate the model every 100 episodes
    update_freq = 2 # Update the target network every 2 episodes
    save_freq = 1000 # Save the model every 1000 episodes
    replay_buffer = ReplayBuffer(int(1))

    model = Q_Network(batch_size, state_dim, action_dim, gamma, epsilon, epsilon_decay, 
                      epsilon_min, learning_rate, total_eps, sim_env, total_its, 
                      replay_buffer, eval_freq, update_freq, save_freq)
    
    # load the model
    epoch = 6000
    model_name = f'Circular_DQN_{epoch}'
    model.load(model_name)

    total_reward_policy = 0
    total_reward_greedy = 0

    # Test the model
    for it in tqdm(range(1000)): # test 1000 times, get the average reward
        reward_policy = model.test()
        total_reward_policy += reward_policy
        sim_env.reset()

        # Test the model with greedy policy
        reward_greedy = model.test(greedy_policy)
        total_reward_greedy += reward_greedy
        sim_env.reset()
    
    print(f'Average Reward with Policy: {total_reward_policy/1000}')
    print(f'Average Reward with Greedy Policy: {total_reward_greedy/1000}')

if __name__ == "__main__":
    test()
    


