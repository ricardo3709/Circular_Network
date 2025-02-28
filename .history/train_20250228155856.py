import numpy as np
from vehicle import Vehicle
from request import Request
from simulator import Simulator
from replay_buffer import ReplayBuffer
from model import DQN

def train(tot_sim_steps, sim_env, replay_buffer, policy):
    for _ in range(tot_sim_steps):
        # Get the current state
        state = sim_env.get_state()

        # Select an action
        # random 
        action = 

        # Take an action and get the next state and reward
        next_state, reward, done = sim_env.step(action)

        # Store the transition in the replay buffer
        # replay_buffer.push(state, action, reward, next_state)

        # Update the state
        state = next_state

        if done:
            sim_env.reset()
    
def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def main():
    # Initialize the simulator
    n_vehs = 10
    state_dim = n_vehs + 1
    action_dim = 1 # 0 or 1
    tot_sim_steps = 1e7 # Total simulation steps, 1 million

    sim_env = Simulator(n_vehs)

    policy = DQN(state_dim, action_dim)

    replay_buffer = ReplayBuffer(tot_sim_steps)

    # HyperParameter for DQN
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon
    

    # Train the model
    train(tot_sim_steps, sim_env, replay_buffer, policy)
