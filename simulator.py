import numpy as np
from vehicle import Vehicle
from request import Request
import torch
import torch.nn as nn

class Simulator:
    def __init__(self, n_vehs):
        self.n_vehs = n_vehs  # Number of Vehs
        self.vehicles = [Vehicle() for _ in range(n_vehs)]  # Initialize vehicles
        self.request = Request()
        self.reset()

    def uniform_init_vehicles(self):
        # Uniformly initialize the position of vehicles
        # [0,1)
        positions = np.linspace(0, 1, self.n_vehs, endpoint=False)
        for veh, pos in zip(self.vehicles, positions):
            veh.set_position(pos)

    def reset(self):
        # Uniformly initialize the position of drivers
        # [0,1)
        self.uniform_init_vehicles()
        # Randomly create a new request
        self.request = Request()
        return self.get_state()

    def step(self, action):
        # action: the closest veh or second closest veh
        # action: 0 or 1
        # action: 0 -> closest veh
        # action: 1 -> second closest veh
        # reward: -distance

        # Calculate the distance between the vehicle and the request
        for veh in self.vehicles:
            veh.distance = self.get_distance(veh, self.request)
        # Get the 2 closest vehicles
        target_vehs = sorted(self.vehicles, key=lambda x: x.distance)[:2]

        # get reward based on the action
        reward = -target_vehs[action].distance

        # Update States
        # Remove selected veh and random initialize a new veh
        self.vehicles.remove(target_vehs[action])
        self.vehicles.append(Vehicle())

        # Randomly create a new request
        self.request = Request()

        return self.get_state(), reward, False  # False: not done
    
    def get_distance(self, veh, request):
        # Calculate the distance between the vehicle and the request
        distance = min(abs(veh.position - request.position), 1 - abs(veh.position - request.position))
        return distance

    def get_state(self):
        # Vehicle density
        sectors = 4
        density = [0] * sectors
        for veh in self.vehicles:
            sector_idx = int(veh.position * sectors)
            density[sector_idx] += 1
        
        # Mean and std of vehicle positions
        positions = [veh.position for veh in self.vehicles]
        mean_pos = np.mean(positions)
        std_pos = np.std(positions)
        
        # Distance between vehicles and request
        distances = [self.get_distance(veh, self.request) for veh in self.vehicles]
        sorted_indices = np.argsort(distances)
        
        # Positions of vehicles sorted by distance to the request
        sorted_positions = [positions[i] for i in sorted_indices]
        
        # Combine all the information to form the state
        state = density + [mean_pos, std_pos] + sorted_positions + [self.request.position]
        
        return state

    def get_state_simple(self):
        # Return the current state of the simulator
        # State: [position of vehs, position of request]
        state = [veh.position for veh in self.vehicles]
        state.append(self.request.position)
        return state



        
    
