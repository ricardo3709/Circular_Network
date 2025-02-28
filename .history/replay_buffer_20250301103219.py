from collections import deque
import random

class ReplayBuffer:
    def __init__(self, tot_sim_steps):
        self.buffer = deque(maxlen=tot_sim_steps)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)