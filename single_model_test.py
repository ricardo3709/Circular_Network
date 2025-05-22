import numpy as np
from simulator import Simulator
from replay_buffer import ReplayBuffer
import csv
from model import Q_Network
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    action = 0
    return action

def test():
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
    replay_buffer = ReplayBuffer(int(1))

    model = Q_Network(batch_size, state_dim, action_dim, gamma, epsilon, epsilon_decay, 
                      epsilon_min, learning_rate, total_eps, sim_env, total_its, 
                      replay_buffer, eval_freq, update_freq, save_freq)
    
    # load the model
    epoch = 7100
    model_name = f'Circular_DQN_{epoch}'
    model_path = f'{n_vehs}vehs/{model_name}'
    model.load_test_model(model_path)

    total_reward_policy = 0
    total_reward_greedy = 0

    percentage_non_greedy_actions_list = []

    tot_test_eps = 5000
    policy_var_gaps_series = []
    policy_reward_series = []
    greedy_var_gaps_series = []
    greedy_reward_series = []
        
    # Test the model
    for it in tqdm(range(tot_test_eps)): # test 1000 times, get the average reward
        # 1. Generate positions of requests
        req_positions = generate_requests_positions(total_its)

        # 2. Test with the policy
        reward_policy, percentage_non_greedy_actions, ep_policy_var_gaps_series, ep_policy_reward_series = model.test(req_positions)
        total_reward_policy += reward_policy
        policy_var_gaps_series.append(ep_policy_var_gaps_series)
        policy_reward_series.append(ep_policy_reward_series)
        percentage_non_greedy_actions_list.append(percentage_non_greedy_actions)
        sim_env.reset()

        # 3. Test with the greedy policy with same requests
        reward_greedy, _, ep_greedy_var_gaps_series, ep_greedy_reward_series = model.test(req_positions, greedy_policy)
        total_reward_greedy += reward_greedy
        greedy_var_gaps_series.append(ep_greedy_var_gaps_series)
        greedy_reward_series.append(ep_greedy_reward_series)
        sim_env.reset()

    avg_reward_policy = total_reward_policy / tot_test_eps
    avg_reward_greedy = total_reward_greedy / tot_test_eps
    non_greedy = np.mean(percentage_non_greedy_actions_list)
    improvement = (total_reward_greedy - total_reward_policy) / total_reward_greedy
    print(f"Avg Reward Policy: {avg_reward_policy:.4f}, Avg Reward Greedy: {avg_reward_greedy:.4f}, Non-Greedy Actions: {non_greedy:%}, Improvement: {improvement:%}")
        
    policy_var_gaps_series = avg_series_over_eps(policy_var_gaps_series)
    greedy_var_gaps_series = avg_series_over_eps(greedy_var_gaps_series)
    policy_reward_series = avg_series_over_eps(policy_reward_series)
    greedy_reward_series = avg_series_over_eps(greedy_reward_series)

    plot_var_gaps_series(policy_var_gaps_series, greedy_var_gaps_series)
    plot_reward_series(policy_reward_series, greedy_reward_series)

def generate_requests_positions(step):
    # Generate a list of length == step
    # contains random numbers [0,1)
    req_positions = np.random.randint(0, 1000, size=step) / 1000
    return req_positions

def avg_series_over_eps(series):
    # Average the series over the episodes
    return np.mean(series, axis=0)

def plot_var_gaps_series(policy_var_gaps_series, greedy_var_gaps_series):
    # moving average
    policy_var_gaps_series = np.convolve(policy_var_gaps_series, np.ones(30)/30, mode='valid')
    greedy_var_gaps_series = np.convolve(greedy_var_gaps_series, np.ones(30)/30, mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(policy_var_gaps_series, label='Policy')
    plt.plot(greedy_var_gaps_series, label='Greedy')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel('Step', fontsize=16)
    plt.ylabel('Variance of Gaps', fontsize=16)
    # plt.title('Variance of Gaps')
    plt.legend(fontsize=16)
    plt.savefig(f'var_gaps_series.png', dpi=300)
    # plt.show()

def plot_reward_series(policy_reward_series, greedy_reward_series):
    # moving average
    policy_reward_series = np.convolve(policy_reward_series, np.ones(30)/30, mode='valid')
    greedy_reward_series = np.convolve(greedy_reward_series, np.ones(30)/30, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(policy_reward_series, label='Policy')
    plt.plot(greedy_reward_series, label='Greedy')
    plt.xlabel('Step', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    # plt.title('Reward')
    plt.legend(fontsize=16)
    plt.savefig(f'reward_series.png', dpi=300)
    # plt.show()

    

if __name__ == "__main__":
    test()
    


