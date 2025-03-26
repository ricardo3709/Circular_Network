import numpy as np
from simulator import Simulator
from replay_buffer import ReplayBuffer
import csv
import torch
import torch.nn as nn
from model_NCI import Q_Network, DQN
import os
import time
from mpi4py import MPI
import pickle
import sys

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def collect_experience(model, sim_env, config, epsilon):
    """收集一个episode的经验数据"""
    # 初始化环境
    state = sim_env.reset()
    
    # 创建历史状态缓冲区
    state_history = []
    for _ in range(3):
        state_history.append(state.copy())
    
    # 拼接状态
    concatenated_state = np.concatenate([state] + state_history)
    
    experiences = []
    total_reward = 0
    
    for it in range(config['total_its']):
        # 选择动作
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 2)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(concatenated_state)
                q_values = model.policy_net(state_tensor)
                action = q_values.argmax().item()
        
        # 执行动作
        next_state, reward, done = sim_env.step(action)
        total_reward += reward
        
        # 更新历史状态
        state_history.pop(0)
        state_history.append(state.copy())
        
        # 拼接下一个状态
        next_concatenated_state = np.concatenate([next_state] + state_history)
        
        # 存储经验
        experiences.append((
            concatenated_state.copy(),
            action,
            reward,
            next_concatenated_state.copy(),
            done
        ))
        
        # 更新状态
        state = next_state
        concatenated_state = next_concatenated_state
    
    return experiences, total_reward

def data_collection_worker(model_config, comm):
    """数据收集工作进程"""
    # 获取MPI信息
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 创建环境和模型
    sim_env = Simulator(
        model_config['n_vehs'], 
        model_config['sectors'], 
        model_config['n_vehs_in_state']
    )
    
    # 创建模型实例 (仅用于推理)
    policy_net = DQN(model_config['state_dim'], model_config['action_dim'])
    
    # 运行的epsilon值
    epsilon = model_config['epsilon']
    
    # 工作进程分配的episode范围
    episodes_per_worker = model_config['total_eps'] // (size - 1)
    start_ep = (rank - 1) * episodes_per_worker
    end_ep = rank * episodes_per_worker if rank < size - 1 else model_config['total_eps']
    
    print(f"Worker {rank} will collect episodes from {start_ep} to {end_ep}")
    
    # 开始收集数据
    ep = start_ep
    
    # 创建本地日志文件
    log_file = f"logs/worker_{rank}_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"Worker {rank} started\n")
    
    while ep < end_ep:
        print(f"Worker {rank} at episode {ep}")
        # 检查是否有新模型参数
        if comm.Iprobe(source=0, tag=10):
            model_params = comm.recv(source=0, tag=10)
            policy_net.load_state_dict(model_params)
            with open(log_file, 'a') as f:
                f.write(f"Episode {ep}: Updated model parameters\n")
        
        # 检查是否收到停止信号
        if comm.Iprobe(source=0, tag=0):
            stop_signal = comm.recv(source=0, tag=0)
            if stop_signal:
                with open(log_file, 'a') as f:
                    f.write("Received stop signal\n")
                break
        
        # 收集一个episode的经验数据
        experiences, total_reward = collect_experience(
            policy_net, sim_env, model_config, epsilon
        )

        print("finished collecting experiences")
        
        # 发送经验数据给训练进程
        # comm.send(experiences, dest=0, tag=20)
        request = comm.isend(experiences, dest=0, tag=20)
        request.wait()

        print("sent experiences to trainer")
        
        with open(log_file, 'a') as f:
            f.write(f"Episode {ep}: Collected experiences, total reward: {total_reward}\n")
        
        # 更新epsilon
        epsilon = max(model_config['epsilon_min'], epsilon * model_config['epsilon_decay'])
        
        ep += 1
    
    # 发送完成信号
    comm.send(None, dest=0, tag=20)
    with open(log_file, 'a') as f:
        f.write("Worker completed\n")

def trainer_process(model_config, comm):
    """训练进程"""
    # 获取MPI信息
    size = comm.Get_size()
    
    # 使用GPU (如果可用)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
    
    # 创建环境和模型
    sim_env = Simulator(
        model_config['n_vehs'], 
        model_config['sectors'], 
        model_config['n_vehs_in_state']
    )
    
    # 创建模型
    model = Q_Network(
        model_config['batch_size'], 
        model_config['state_dim'], 
        model_config['action_dim'],
        model_config['gamma'], 
        model_config['epsilon'], 
        model_config['epsilon_decay'],
        model_config['epsilon_min'], 
        model_config['learning_rate'], 
        model_config['total_eps'],
        sim_env, 
        model_config['total_its'], 
        None,  # 不使用类内部的replay buffer
        model_config['eval_freq'],
        model_config['update_freq'], 
        model_config['save_freq']
    )
    
    # 将模型移到设备上
    model.policy_net.to(device)
    model.target_net.to(device)
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(model_config['buffer_capacity'])
    
    # 清理日志文件
    os.makedirs('logs', exist_ok=True)
    with open('logs/training_log.txt', 'w') as f:
        f.write("Training started\n")

    with open('logs/loss_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Loss', 'Buffer_Size', 'Epsilon'])
    
    # 活动工作进程数量
    active_workers = size - 1
    completed_workers = 0
    
    # 训练迭代计数器
    iteration = 0
    
    # 发送初始模型参数给所有工作进程
    initial_params = model.policy_net.state_dict()
    for i in range(1, size):
        cpu_params = {k: v.cpu() for k, v in initial_params.items()}
        comm.send(cpu_params, dest=i, tag=10)
    
    print(f"Sent initial model to {size-1} workers")
    
    # 训练循环
    loss_history = []
    
    while active_workers > 0:
        # 检查是否有工作进程发送经验数据
        for i in range(1, size):
            if comm.Iprobe(source=i, tag=20):
                experiences = comm.recv(source=i, tag=20)
                
                # 如果收到None，表示工作进程已完成
                if experiences is None:
                    active_workers -= 1
                    completed_workers += 1
                    print(f"Worker {i} completed. {active_workers} workers still active.")
                    continue
                
                # 将经验数据添加到回放缓冲区
                for exp in experiences:
                    replay_buffer.push(*exp)
        
        # 如果回放缓冲区不够大，等待更多数据
        if len(replay_buffer) < model_config['batch_size']:
            time.sleep(0.1)
            continue
        
        # 从回放缓冲区采样batch
        batch = replay_buffer.sample(model_config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量并移动到设备
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        
        # 计算当前Q值
        q_values = model.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        next_q_values = model.target_net(next_states).max(1).values
        target_q_values = rewards + (1 - dones) * model_config['gamma'] * next_q_values
        
        # 计算损失
        loss = model.loss(q_values, target_q_values)
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 更新网络
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        # 记录训练信息
        if iteration % 100 == 0:
            avg_loss = np.mean(loss_history)
            loss_history = []
            print(f"Iteration {iteration}, Loss: {avg_loss}, Buffer size: {len(replay_buffer)}, Active workers: {active_workers}")
            
            with open('logs/loss_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, avg_loss, len(replay_buffer), model.epsilon])
        
        # 定期更新目标网络
        if iteration % model_config['update_freq'] == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())
        
        # 定期发送更新后的模型参数给所有工作进程
        if iteration % model_config['trainer_update_freq'] == 0:
            updated_params = model.policy_net.state_dict()
            for i in range(1, size):
                if i != completed_workers + 1:  # 跳过已完成的工作进程
                    cpu_params = {k: v.cpu() for k, v in updated_params.items()}
                    comm.send(cpu_params, dest=i, tag=10)
            print(f"Sent updated model at iteration {iteration}")
        
        # 定期保存模型
        if iteration % model_config['save_freq'] == 0:
            save_path = f'saved_models/Circular_DQN_iter_{iteration}'
            os.makedirs('saved_models', exist_ok=True)
            model.save(f'Circular_DQN_iter_{iteration}')
            print(f"Saved model at iteration {iteration}")
        
        # 定期评估模型
        if iteration % model_config['eval_freq'] == 0:
            reward, non_greedy = model.eval(device)
            with open('logs/training_log.txt', 'a') as f:
                f.write(f"Iteration {iteration}, Reward: {reward}, Non-Greedy: {non_greedy:.2%}, Loss: {avg_loss if 'avg_loss' in locals() else 'N/A'}\n")
        
        iteration += 1
        
        # 如果达到最大迭代次数，提前退出
        if iteration >= model_config['max_trainer_iterations']:
            print(f"Reached maximum iterations ({iteration}). Sending stop signal to workers.")
            for i in range(1, size):
                if i != completed_workers + 1:  # 跳过已完成的工作进程
                    comm.send(True, dest=i, tag=0)
            break
    
    print("Training completed")
    
    # 保存最终模型
    model.save('Circular_DQN_final')
    print("Saved final model")

def main():
    # 初始化MPI环境
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        print("Error: At least 2 MPI processes are required (1 for training, 1+ for data collection)")
        return
    
    # 超参数
    sectors = 4
    n_vehs = 10
    n_vehs_in_state = n_vehs
    batch_size = 64
    state_dim = (n_vehs*2+1+4)*4
    action_dim = 2
    gamma = 0.999
    epsilon = 1.0
    epsilon_decay = 0.9999
    epsilon_min = 0.10
    learning_rate = 3e-4
    total_eps = 20001
    total_its = 1000
    eval_freq = 1000
    update_freq = 10
    save_freq = 1000
    
    # 新的并行训练参数
    trainer_update_freq = 50  # 训练进程多少次迭代后更新工作进程的模型
    max_trainer_iterations = total_eps * total_its // batch_size  # 训练的最大迭代次数
    # buffer_capacity = 200 * 1024 * 1024  # 约2亿个样本，适合384GB内存
    buffer_capacity = 1024
    
    # 模型配置
    model_config = {
        'sectors': sectors,
        'n_vehs': n_vehs,
        'n_vehs_in_state': n_vehs_in_state,
        'batch_size': batch_size,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'gamma': gamma,
        'epsilon': epsilon,
        'epsilon_decay': epsilon_decay,
        'epsilon_min': epsilon_min,
        'learning_rate': learning_rate,
        'total_eps': total_eps,
        'total_its': total_its,
        'eval_freq': eval_freq,
        'update_freq': update_freq,
        'save_freq': save_freq,
        'trainer_update_freq': trainer_update_freq,
        'max_trainer_iterations': max_trainer_iterations,
        'buffer_capacity': buffer_capacity
    }
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    # 根据rank分配角色
    if rank == 0:
        print(f"Process {rank} starting as trainer")
        # 训练进程
        trainer_process(model_config, comm)
    else:
        print(f"Process {rank} starting as data collector")
        # 数据收集进程
        data_collection_worker(model_config, comm)
    
    # 同步所有进程
    comm.Barrier()
    if rank == 0:
        print("All processes completed successfully.")

if __name__ == "__main__":
    main()