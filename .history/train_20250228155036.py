import numpy as np
from vehicle import Vehicle
from request import Request
from simulator import Simulator

def train():
    pass

def main():
    # Initialize the simulator
    n_vehs = 10
    sim_env = Simulator(n_vehs)
    state = sim_env.reset()

    # Train the model
    train()

    # 
