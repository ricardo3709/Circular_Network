import numpy as np
from vehicle import Vehicle
from request import Request
from simulator import Simulator
from replay_buffer import ReplayBuffer
from model import DQN

def train(tot_eps, sim_env, replay_buffer, policy):
    for ep in range(int(tot_eps)):
        state = sim_env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2)

    
def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def main():
    # Initialize the simulator
    n_vehs = 10
    state_dim = n_vehs + 1
    action_dim = 1 # 0 or 1
    tot_eps = 1e5 # Total simulation episodes

    sim_env = Simulator(n_vehs)

    policy = DQN(state_dim, action_dim)

    replay_buffer = ReplayBuffer(tot_eps)


    # HyperParameter for DQN
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01


    # Train the model
    train(tot_eps, sim_env, replay_buffer, policy)
