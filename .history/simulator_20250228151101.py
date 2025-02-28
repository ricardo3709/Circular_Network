import numpy as np
from vehicle import Vehicle

class Simulator:
    def __init__(self, n_drivers):
        self.n_drivers = n_drivers  # 司机数量
        self.vehicles = []
        self.reset()

    def reset(self):
        # Uniformly initialize the position of drivers
        # [0,1)
        self.dr
        return self.get_state()

    def step(self, action):
        # 计算司机与请求的距离
        driver_pos = self.drivers[action]
        distance = min(abs(driver_pos - self.request), 1 - abs(driver_pos - self.request))
        reward = -distance  # 负距离作为奖励
        # 更新状态：移除被选中的司机，添加新司机和请求
        self.drivers = np.delete(self.drivers, action)
        self.drivers = np.append(self.drivers, np.random.rand())
        self.request = np.random.rand()
        return self.get_state(), reward, False  # False表示未结束

    def get_state(self):
        # 返回当前状态：司机位置和请求位置
        return np.concatenate([self.drivers, [self.request]])

    def uniform_init_vehicles(self):
        
    
