import numpy as np

class Request:
    def __init__(self, n_slots=1000):
        self.n_slots = n_slots
        self.position = np.random.randint(0,self.n_slots)/self.n_slots  # Randomly initialize the position of the request [0,1)