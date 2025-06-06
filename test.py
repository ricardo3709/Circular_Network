import numpy as np
from simulator import Simulator
from replay_buffer import ReplayBuffer
import csv
from model import Q_Network
import torch
from tqdm import tqdm

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
    
    # write the header to the csv file
    with open('logs/test_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg_Reward_Policy', 'Avg_Reward_Greedy', 'Avg_Non_Greedy_Actions', 'Improvement'])
    
    # # load the model
    # epoch = 9300
    # model_name = f'Circular_DQN_{epoch}'
    # model.load(model_name)

    # total_reward_policy = 0
    # total_reward_greedy = 0

    # percentage_non_greedy_actions_list = []

    # tot_test_eps = 50

    # test all models
    max_epoch = 7800
    for epoch in tqdm(range(0, max_epoch, 100)):
        model_name = f'Circular_DQN_{epoch}'
        model.load(model_name)

        total_reward_policy = 0
        total_reward_greedy = 0

        percentage_non_greedy_actions_list = []

        tot_test_eps = 50
            
        # Test the model
        for it in range(tot_test_eps): # test 1000 times, get the average reward
            # 1. Generate positions of requests
            req_positions = generate_requests_positions(total_its)

            # 2. Test with the policy
            reward_policy, percentage_non_greedy_actions, _, _ = model.test(req_positions)
            total_reward_policy += reward_policy
            percentage_non_greedy_actions_list.append(percentage_non_greedy_actions)
            sim_env.reset()

            # 3. Test with the greedy policy with same requests
            reward_greedy, _, _, _ = model.test(req_positions, greedy_policy)
            total_reward_greedy += reward_greedy
            sim_env.reset()

        avg_reward_policy = total_reward_policy / tot_test_eps
        avg_reward_greedy = total_reward_greedy / tot_test_eps
        non_greedy = np.mean(percentage_non_greedy_actions_list)
        improvement = (total_reward_greedy - total_reward_policy) / total_reward_greedy
        # save the results to a csv file
        with open('logs/test_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_reward_policy, avg_reward_greedy, non_greedy, improvement])
    
def generate_requests_positions(step):
    # Generate a list of length == step
    # contains random numbers [0,1)
    req_positions = np.random.randint(0, 1000, size=step) / 1000
    return req_positions
    

if __name__ == "__main__":
    test()
    


