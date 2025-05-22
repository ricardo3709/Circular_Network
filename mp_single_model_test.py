import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from simulator import Simulator
from model import Q_Network
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import platform

# 参数配置
sectors = 4
n_vehs = 10
n_vehs_in_state = n_vehs
batch_size = 64
state_dim = (n_vehs * 2 + 1 + 4) * 4
action_dim = 2
gamma = 0.999
epsilon = 1.0
epsilon_decay = 0.9999
epsilon_min = 0.10
learning_rate = 3e-4
total_eps = 20001
total_its = 1000
eval_freq = 100
update_freq = 10
save_freq = 100
replay_buffer = ReplayBuffer(int(1))

epoch = 7100
model_name = f'Circular_DQN_{epoch}'
model_path = os.path.join(f'{n_vehs}vehs', model_name)

tot_test_eps = 5000

def get_optimal_worker_count():
    """获取最优的工作线程数量"""
    if platform.processor() == 'arm':
        # M4 Pro 有 10 个性能核心
        return 4
    else:
        # 对于其他处理器，使用物理核心数
        return os.cpu_count()

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    action = 0
    return action

# 单次测试函数
def run_one_episode(_):
    # 为线程独立实例化环境和模型
    sim_env = Simulator(n_vehs, sectors, n_vehs_in_state)
    model = Q_Network(
        batch_size, state_dim, action_dim,
        gamma, epsilon, epsilon_decay,
        epsilon_min, learning_rate,
        total_eps, sim_env, total_its,
        replay_buffer, eval_freq,
        update_freq, save_freq
    )
    model.load_test_model(model_path)

    # 生成请求序列
    req_positions = generate_requests_positions(total_its)

    # 策略测试
    reward_policy, perc_non_greedy, var_gaps_p, reward_series_p = model.test(req_positions)
    sim_env.reset()

    # 贪心测试
    reward_greedy, _, var_gaps_g, reward_series_g = model.test(req_positions, greedy_policy)
    sim_env.reset()

    return reward_policy, perc_non_greedy, var_gaps_p, reward_series_p, reward_greedy, var_gaps_g, reward_series_g


def main():
    # 多线程执行
    max_workers = get_optimal_worker_count()
    print(f"Using {max_workers} threads for testing")
    
    # 计算每个线程需要处理的episode数
    eps_per_worker = tot_test_eps // max_workers
    remaining_eps = tot_test_eps % max_workers
    
    # 为每个线程分配任务
    worker_tasks = []
    start_ep = 0
    for i in range(max_workers):
        # 分配任务，确保所有episode都被处理
        eps_for_this_worker = eps_per_worker + (1 if i < remaining_eps else 0)
        worker_tasks.append((start_ep, start_ep + eps_for_this_worker, i))  # 添加rank
        start_ep += eps_for_this_worker

    # 每个线程的结果收集器
    worker_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for start, end, rank in worker_tasks:
            # 每个线程处理多个episode，传入rank
            future = executor.submit(process_worker_tasks, start, end, rank)
            futures.append(future)
        
        # 等待所有线程完成并收集结果
        for fut in as_completed(futures):
            worker_results.append(fut.result())

    # 汇总所有线程的结果
    total_reward_policy = 0.0
    total_reward_greedy = 0.0
    percentage_non_greedy_list = []
    all_var_gaps_p = []
    all_reward_series_p = []
    all_var_gaps_g = []
    all_reward_series_g = []

    for worker_result in worker_results:
        total_reward_policy += worker_result[0]
        total_reward_greedy += worker_result[1]
        percentage_non_greedy_list.extend(worker_result[2])
        all_var_gaps_p.extend(worker_result[3])
        all_reward_series_p.extend(worker_result[4])
        all_var_gaps_g.extend(worker_result[5])
        all_reward_series_g.extend(worker_result[6])

    # 结果汇总
    avg_reward_policy = total_reward_policy / tot_test_eps
    avg_reward_greedy = total_reward_greedy / tot_test_eps
    non_greedy = np.mean(percentage_non_greedy_list)
    improvement = (total_reward_greedy - total_reward_policy) / total_reward_greedy

    print(f"Avg Reward Policy: {avg_reward_policy:.4f}, "
          f"Avg Reward Greedy: {avg_reward_greedy:.4f}, "
          f"Non-Greedy Actions: {non_greedy:.2%}, "
          f"Improvement: {improvement:.2%}")

    # 可视化
    plot_var_gaps_series(
        avg_series_over_eps(all_var_gaps_p),
        avg_series_over_eps(all_var_gaps_g)
    )
    plot_reward_series(
        avg_series_over_eps(all_reward_series_p),
        avg_series_over_eps(all_reward_series_g)
    )

def process_worker_tasks(start_ep, end_ep, rank):
    """处理一个工作线程的所有任务"""
    total_reward_policy = 0.0
    total_reward_greedy = 0.0
    percentage_non_greedy_list = []
    all_var_gaps_p = []
    all_reward_series_p = []
    all_var_gaps_g = []
    all_reward_series_g = []

    # 只为rank 0的worker显示进度条
    episode_range = tqdm(range(start_ep, end_ep), desc=f"Worker {rank}") if rank == 0 else range(start_ep, end_ep)
    
    for i in episode_range:
        rp, perc, vg_p, rs_p, rg, vg_g, rs_g = run_one_episode(i)
        total_reward_policy += rp
        total_reward_greedy += rg
        percentage_non_greedy_list.append(perc)
        all_var_gaps_p.append(vg_p)
        all_reward_series_p.append(rs_p)
        all_var_gaps_g.append(vg_g)
        all_reward_series_g.append(rs_g)

    return (total_reward_policy, total_reward_greedy, percentage_non_greedy_list,
            all_var_gaps_p, all_reward_series_p, all_var_gaps_g, all_reward_series_g)

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

if __name__ == '__main__':
    main()
