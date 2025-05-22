from collections import deque
import random
import numpy as np
import torch.multiprocessing as mp

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.pos = 0
        self.size = 0


    def push(self, state, action, reward, next_state, done):
        # self.buffer.append((state, action, reward, next_state, done))
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # return random.sample(self.buffer, batch_size)
        n = self.size
        batch_size = min(batch_size, n)
        idx = np.random.randint(0, n, size=batch_size)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


    def __len__(self):
        # return len(self.buffer)
        return self.size


class SharedReplayBuffer:
    def __init__(self, capacity):
        """
        支持多进程共享的回放缓冲区
        """
        self.capacity = capacity
        self.manager = mp.Manager()
        self.buffer = self.manager.list()  # 使用Manager.list作为共享缓冲区
        self.position = mp.Value('i', 0)  # 共享整数值，跟踪当前位置
        self.size = mp.Value('i', 0)  # 共享整数值，跟踪当前缓冲区大小
        self.lock = mp.Lock()  # 用于同步访问

    def push(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        """
        with self.lock:  # 加锁以确保线程安全
            experience = (state, action, reward, next_state, done)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position.value] = experience
            
            self.position.value = (self.position.value + 1) % self.capacity
            self.size.value = min(self.size.value + 1, self.capacity)

    def sample(self, batch_size):
        """
        从缓冲区采样一批经验
        """
        with self.lock:  # 加锁以确保线程安全
            batch_size = min(batch_size, self.size.value)
            return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        """
        返回缓冲区中的经验数量
        """
        return self.size.value
    
    def clear(self):
        """
        清空缓冲区
        """
        with self.lock:
            self.buffer.clear()
            self.position.value = 0
            self.size.value = 0