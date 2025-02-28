import numpy as np
from vehicle import Vehicle
from request import Request
from simulator import Simulator
from replay_buffer import ReplayBuffer
from model import Q_Network

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def main():
    # HyperParameter for DQN
    n_vehs = 10
    batch_size = 64
    state_dim = n_vehs + 1
    action_dim = 2 # 0 or 1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    learning_rate = 1e-3
    total_eps = 1e5 # Total simulation episodes
    sim_env = Simulator(n_vehs)
    total_its = 1000 # Total iterations per episode
    eval_freq = 100 # Evaluate the model every 100 episodes
    update_freq = 2 # Update the target network every 2 episodes
    save_freq = 1000 # Save the model every 1000 episodes

    # replay_buffer = ReplayBuffer(total_eps * total_its / 10)

    
    model = Q_Network(batch_size, state_dim, action_dim, gamma, epsilon, epsilon_decay, 
                      epsilon_min, learning_rate, total_eps, sim_env, total_its, 
                      replay_buffer, eval_freq, update_freq, save_freq)

    # Train the model
    model.train()

