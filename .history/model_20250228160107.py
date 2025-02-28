import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)

        # HyperParameter for DQN
        self.batch_size = 64
        gamma = 0.99
        epsilon = 1.0
        epsilon_decay = 0.999
        epsilon_min = 0.01
        gamma = 0.99

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

 def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.001,
        policy_noise=0.3,
        noise_clip=0.5,
        policy_freq=2
    )