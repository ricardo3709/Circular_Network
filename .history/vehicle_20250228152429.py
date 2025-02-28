import numpy as np

class Vehicle:
    def __init__(self):
        self.position = np.random.rand()  # Randomly initialize the position of the vehicle [0,1)
        self.distance = 0 # Distance between the vehicle and the request
    
    def set_position(self, position):
        self.position = position