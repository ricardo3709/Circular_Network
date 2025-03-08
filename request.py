import numpy as np

class Request:
    def __init__(self):
        self.position = np.random.randint(0,1000)/1000  # Randomly initialize the position of the request [0,1)