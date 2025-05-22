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
        self.gaps = np.zeros(self.n_vehs)
        self.prev_gaps = np.zeros(self.n_vehs)
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
        # return self.get_state()
        return self.get_state_discrete()

    def step(self, action, request_position=None):
        # action: the closest veh or second closest veh
        # action: 0 or 1
        # action: 0 -> closest veh
        # action: 1 -> second closest veh

        if request_position is not None: # For testing
            self.request.position = request_position

        # Calculate the distance between the vehicle and the request
        for veh in self.vehicles:
            veh.distance = self.get_distance(veh, self.request)
        # Get the 2 closest vehicles
        target_vehs = sorted(self.vehicles, key=lambda x: x.distance)[:2]

        # get reward based on the action
        # reward = -target_vehs[action].distance
        reward = self.get_reward(action, target_vehs)

        # Update States
        # Remove selected veh and random initialize a new veh
        self.vehicles.remove(target_vehs[action])
        self.vehicles.append(Vehicle())

        # Randomly create a new request
        self.request = Request()

        # return self.get_state(), reward, False  # False: not done
        return self.get_state_discrete(), reward, False
    
    def get_distance(self, veh, request):
        # Calculate the distance between the vehicle and the request
        distance = min(abs(veh.position - request.position), 1 - abs(veh.position - request.position))
        return distance
    
    def get_reward(self, action, target_vehs):
        reward = -target_vehs[action].distance
        return reward
        # just base reward
        veh_distances = [veh.distance for veh in self.vehicles]
        veh_distances.sort()

        reward = -veh_distances[action]
        
        return reward

        # # 基础运输效率奖励（短期激励）
        # base_reward = 1 / (self.vehicles[action].distance + 1e-6)
        
        # # 长期系统平衡奖励（新增）
        # gap_ratios = self.gaps / np.mean(self.gaps)
        # balance_reward = np.exp(-np.std(gap_ratios))  # 均匀分布奖励
        
        # # 探索引导奖励（新增）
        # explore_bonus = 0.5 * (1 - self.prev_gaps[action]/np.max(self.prev_gaps))
        
        # # 组合奖励（动态权重）
        # return (0.6 * base_reward + 0.3 * balance_reward + 0.1 * explore_bonus)


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
    
    def get_state_discrete(self):
        # 1. position of vehicles
        positions = [v.position for v in self.vehicles]
        sorted_pos = np.sort(positions)

        # 2. gaps between vehicles
        gaps = []
        for i in range(len(positions)-1):
            gap = sorted_pos[i+1] - sorted_pos[i]
            gaps.append(gap)

        # get last gap
        gaps.append(1 - sorted_pos[-1] + sorted_pos[0])

        # calculate the variance and mean of gaps
        gaps_variance = np.var(gaps)
        gaps_mean = np.mean(gaps)

        # 3. position of request
        req_pos = self.request.position

        # 4. 2 closest vehicles distance to request
        distances = [self.get_distance(veh, self.request) for veh in self.vehicles]
        distances.sort()
        distances = distances[:2]

        state = np.concatenate([sorted_pos, gaps, [req_pos], distances, [gaps_variance, gaps_mean]])
    
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



        
    
