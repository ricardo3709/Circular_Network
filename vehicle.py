import numpy as np

class Vehicle:
    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.position = np.random.randint(0,self.n_slots)/self.n_slots  # Initialize the position of the vehicle, circle is finite, 1000 intervals
        self.distance = 0 # Distance between the vehicle and the request
    
    def set_position(self, position):
        self.position = position