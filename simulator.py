import numpy as np
from vehicle import Vehicle
from request import Request
import torch
import torch.nn as nn

class Simulator:
    def __init__(self, n_vehs, sectors, n_vehs_in_state):
        self.n_vehs = n_vehs  # Number of Vehs
        self.n_sectors = sectors  # Number of sectors
        self.vehicles = [Vehicle() for _ in range(n_vehs)]  # Initialize vehicles
        self.request = Request()
        self.n_vehs_in_state = n_vehs_in_state
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

        # # Calculate Uniform Distribution Reward (Before)
        # positions_before = [veh.position for veh in self.vehicles]
        # gaps_before = [(positions_before[(i+1)%len(positions_before)] - pos) % 1.0 
        #           for i, pos in enumerate(positions_before)]
        # max_gap_before = max(gaps_before)
        # std_gap_before = np.std(gaps_before)

        # get reward based on the action
        reward = -target_vehs[action].distance

        # Update States
        # Remove selected veh and random initialize a new veh
        self.vehicles.remove(target_vehs[action])
        self.vehicles.append(Vehicle())

        # # Calculate Uniform Distribution Reward (After)
        # positions_after = sorted([veh.position for veh in self.vehicles])
        # gaps_after = [(positions_after[(i+1)%len(positions_after)] - pos) % 1.0 
        #             for i, pos in enumerate(positions_after)]
        # max_gap_after = max(gaps_after)
        # std_gap_after = np.std(gaps_after)

        # # Calculate the reward based on the gap
        # gap_change = max_gap_before - max_gap_after
        # std_change = std_gap_before - std_gap_after

        # gap_weight = 0.1
        # std_weight = 0.05

        # uniformly_reward = (gap_weight * gap_change + std_weight * std_change)*0.3

        # print(f"Distance Reward: {reward}")
        # print(f"Uniformly Reward: {uniformly_reward}")

        # reward += uniformly_reward 

        # Randomly create a new request
        self.request = Request()

        return self.get_state(), reward, False  # False: not done
    
    def get_distance(self, veh, request):
        # Calculate the distance between the vehicle and the request
        distance = min(abs(veh.position - request.position), 1 - abs(veh.position - request.position))
        return distance

    def get_state(self):
        # compute the interval of vehicles 
        positions = [veh.position for veh in self.vehicles]
        # always start from the 0 position
        positions.sort()
        # get the gap between vehicles
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        # get the gap between the last vehicle and the first vehicle
        gaps.append(1 - positions[-1] + positions[0])
        state = gaps + [self.request.position]
        return state

    # def get_state(self):
    #     # Vehicle density
    #     sectors = self.n_sectors
    #     density = [0] * sectors
    #     for veh in self.vehicles:
    #         sector_idx = int(veh.position * sectors)
    #         density[sector_idx] += 1
        
    #     density = [x / self.n_vehs for x in density] # Normalize the density
        
    #     # Mean and std of vehicle positions
    #     positions = [veh.position for veh in self.vehicles]
    #     mean_pos = np.mean(positions)
    #     std_pos = np.std(positions)
        
    #     # Distance between vehicles and request
    #     distances = [self.get_distance(veh, self.request) for veh in self.vehicles]
    #     distances.sort()
    #     # distances = np.array(distances).sort() # Get the 5 closest vehicles
    #     # sorted_indices = np.argsort(distances)
        
    #     # Positions of vehicles sorted by distance to the request
    #     # sorted_positions = [positions[i] for i in sorted_indices]
        
    #     # Combine all the information to form the state
    #     state = density + [mean_pos, std_pos] + distances[:self.n_vehs_in_state] + [self.request.position]
        
    #     return state

    def get_state_simple(self):
        # Return the current state of the simulator
        # State: [position of vehs, position of request]
        state = [veh.position for veh in self.vehicles]
        state.append(self.request.position)
        return state



        
    
