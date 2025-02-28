import numpy as np

class Vehicle:
    def __init__(self):
        self.position = np.random.rand()  # Randomly initialize the position of the vehicle
    
    def set_position(self, position):
        self.position = position