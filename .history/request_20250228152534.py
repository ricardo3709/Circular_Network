import numpy as np

class Request:
    def __init__(self):
        self.position = np.random.rand()  # Randomly initialize the position of the request [0,1)
        self.distance = 0  # Distance between the vehicle and the request