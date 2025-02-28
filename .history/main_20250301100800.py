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
    # Initialize the simulator
    n_vehs = 10
    tot_eps = 1e5 # Total simulation episodes

    sim_env = Simulator(n_vehs)

    replay_buffer = ReplayBuffer(tot_eps)


    # HyperParameter for DQN
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01

    batch_size = 64
    state_dim = n_vehs + 1
    action_dim = 2 # 0 or 1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay: Any,
    epsilon_min: Any,
    learning_rate: Any,
    total_eps: Any,
    sim_env: Any,
    total_its: Any

    model = Q_Network()

    # Train the model
    train(tot_eps, sim_env, replay_buffer, policy)
