import numpy as np
from vehicle import Vehicle
from request import Request
from simulator import Simulator

def train(tot_sim_steps, sim_env):
    for _ in range(tot_sim_steps):
        # Get the current state
        state = sim_env.get_state()

        # Select an action
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

def main():
    # Initialize the simulator
    n_vehs = 10
    sim_env = Simulator(n_vehs)
    tot_sim_steps = 1000

    # Train the model
    train(tot_sim_steps, sim_env)
