import numpy as np
from vehicle import Vehicle
from request import Request
from simulator import Simulator

def train():
    

def main():
    # Initialize the simulator
    n_vehs = 10
    sim_env = Simulator(n_vehs)
    state = sim_env.reset()
    tot_sim_steps = 1000

    # Train the model
    train(tot_sim_steps, sim_env)
