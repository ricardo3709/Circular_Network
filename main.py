import numpy as np
from simulator import Simulator
from replay_buffer import ReplayBuffer
import csv
from PPO_Model import PPO
from model import Q_Network

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def main():
    # clear the content of logs
    with open('logs/training_log.txt', 'w') as f:
        f.write('')

    # delete the content of loss_log.csv
    with open('logs/loss_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Avg_Loss', 'Reward', 'Epsilon'])

    # HyperParameter for DQN
    sectors = 4 # Number of sectors of ring
    n_vehs = 10
    n_vehs_in_state = n_vehs
    batch_size = 64
    # state_dim = sectors + n_vehs_in_state + 2 + 1 # 4 sectors + 10 vehicles + mean and std of vehicles + request position
    state_dim = (n_vehs*2+1+4)*4 # 10 vehicels, 10 gaps, 1 request position, gaps mean, gaps variance, 2 closest vehicles distance to request, 4 cat states
    action_dim = 2 # 0 or 1
    gamma = 0.999
    epsilon = 1.0
    epsilon_decay = 0.9999
    epsilon_min = 0.10
    learning_rate = 3e-4
    total_eps = 20001 # Total simulation episodes
    sim_env = Simulator(n_vehs, sectors, n_vehs_in_state)
    total_its = 1000 # Total iterations per episode
    eval_freq = 100 # Evaluate the model every 100 episodes
    update_freq = 10 # Update the target network every 2 episodes
    save_freq = 100 # Save the model every 1000 episodes

    replay_buffer = ReplayBuffer(int(1e7))
    # replay_buffer = ReplayBuffer(1000)

    # DQN Model
    model = Q_Network(batch_size, state_dim, action_dim, gamma, epsilon, epsilon_decay, 
                      epsilon_min, learning_rate, total_eps, sim_env, total_its, 
                      replay_buffer, eval_freq, update_freq, save_freq)
    # Train the model
    model.train()

    # # # PPO Model
    # # lr_actor = 1e-4
    # # lr_critic = 3e-4
    # # state_dim = 48 #discrete state
    # # ppo_agent = PPO(
    # #     state_dim, action_dim, lr_actor, lr_critic, gamma, gae_lambda=0.97,
    # #     policy_clip=0.2, batch_size=256, n_epochs=10, entropy_coef=0.05,
    # #     sim_env=sim_env, total_its=total_its, eval_freq=500, save_freq=500
    # # )

    # # 训练模型
    # ppo_agent.train(num_episodes=total_eps)


if __name__ == "__main__":
    main()
