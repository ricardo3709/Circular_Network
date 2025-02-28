import numpy as np
from vehicle import Vehicle

class Simulator:
    def __init__(self, n_vehs):
        self.n_vehs = n_vehs  # Number of Vehs
        self.vehicles = []
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
        return self.get_state()

    def step(self, action):
        # action: the closest veh or second closest veh
        # action: 0 or 1
        # reward: -distance

        # get two closest vehs
        self.vehicles.sort(key=lambda veh: abs(veh.position - self.request))

        # Calculate the distance between the vehicle and the request
        
        # 计算司机与请求的距离
        driver_pos = self.drivers[action]
        distance = min(abs(driver_pos - self.request), 1 - abs(driver_pos - self.request))
        reward = -distance  # 负距离作为奖励
        # 更新状态：移除被选中的司机，添加新司机和请求
        self.drivers = np.delete(self.drivers, action)
        self.drivers = np.append(self.drivers, np.random.rand())
        self.request = np.random.rand()
        return self.get_state(), reward, False  # False表示未结束
    
    def get_distance(self, veh, request):
        # Calculate the distance between the vehicle and the request
        distance = min(abs(driver_pos - self.request), 1 - abs(driver_pos - self.request))

    def get_state(self):
        # Return the current state of the simulator
        # State: [position of vehs, position of request]
        state = [veh.position for veh in self.vehicles]
        state.append(self.request)
        return state



        
    
