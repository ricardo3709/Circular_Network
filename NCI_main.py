import numpy as np
from simulator import Simulator
from replay_buffer import SharedReplayBuffer as ReplayBuffer
import csv
from PPO_Model import PPO
from model_NCI import Q_Network
import torch
import torch.multiprocessing as mp
import os
import time
from queue import Empty

def greedy_policy(state):
    # Greedy policy
    # Select the vehicle with the minimum distance
    return np.argmin(state[:-1])

def data_collection_worker(worker_id, model_config, shared_buffer, model_queue, stop_event):
    """
    数据收集工作进程
    
    worker_id: 工作进程ID
    model_config: 模型配置参数
    shared_buffer: 共享的经验回放缓冲区
    model_queue: 接收最新模型参数的队列
    stop_event: 停止事件，用于通知工作进程退出
    """
    # 创建环境和模型
    sim_env = Simulator(
        model_config['n_vehs'], 
        model_config['sectors'], 
        model_config['n_vehs_in_state']
    )
    
    # 创建本地模型（只用于推理）
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
        shared_buffer, 
        model_config['eval_freq'],
        model_config['update_freq'], 
        model_config['save_freq']
    )
    
    device = torch.device("cpu")  # 数据收集工作进程只使用CPU
    model.policy_net.to(device)
    model.target_net.to(device)
    
    # 工作进程分配的episode范围
    total_workers = model_config['n_collection_workers']
    episodes_per_worker = model_config['total_eps'] // total_workers
    start_ep = worker_id * episodes_per_worker
    end_ep = (worker_id + 1) * episodes_per_worker if worker_id < total_workers - 1 else model_config['total_eps']
    
    # 开始收集数据
    ep = start_ep
    latest_model_params = None

    print(f"Worker {worker_id} starting data collection from episode {start_ep} to {end_ep}")
    
    while ep < end_ep and not stop_event.is_set():
        # 检查是否有新的模型参数
        try:
            # 非阻塞方式检查队列
            latest_model_params = model_queue.get_nowait()
            model.policy_net.load_state_dict(latest_model_params)
            print(f"Worker {worker_id} updated model parameters")
        except Empty:
            # 队列为空，继续使用当前模型
            pass
        
        # 初始化环境
        state = sim_env.reset()
        
        # 创建历史状态缓冲区
        state_history = []
        for _ in range(3):  # 存储3个历史状态
            state_history.append(state.copy())
        
        # 拼接状态
        concatenated_state = np.concatenate([state] + state_history)
        concatenated_state_tensor = torch.tensor(concatenated_state, dtype=torch.float32).to(device)
        
        total_reward = 0
        
        # 执行一个episode的数据收集
        for it in range(model_config['total_its']):
            # 选择动作
            if latest_model_params is None or np.random.rand() < model_config['epsilon']:
                # 如果还没有收到模型参数或者需要探索，则随机选择动作
                action = np.random.randint(0, 2)
            else:
                # 使用当前策略网络选择动作
                with torch.no_grad():
                    q_values = model.policy_net(concatenated_state_tensor)
                    action = q_values.argmax().item()
            
            # 执行动作
            next_state, reward, done = sim_env.step(action)
            total_reward += reward
            
            # 更新历史状态
            state_history.pop(0)  # 移除最旧的状态
            state_history.append(state.copy())  # 添加当前状态
            
            # 创建下一个状态的表示
            next_concatenated_state = np.concatenate([next_state] + state_history)
            next_concatenated_state_tensor = torch.tensor(next_concatenated_state, dtype=torch.float32).to(device)
            
            # 存储经验到共享缓冲区
            shared_buffer.push(
                concatenated_state.copy(), 
                action, 
                reward, 
                next_concatenated_state.copy(), 
                done
            )
            
            # 更新状态
            state = next_state
            concatenated_state = next_concatenated_state
            concatenated_state_tensor = next_concatenated_state_tensor
            
            if it >= model_config['total_its'] - 1:
                done = True
        
        if worker_id == 0 and ep % 10 == 0:
            print(f"Worker {worker_id}: Completed episode {ep}, total reward: {total_reward}, buffer size: {len(shared_buffer)}")
        
        ep += 1
    
    print(f"Worker {worker_id} completed data collection")

def trainer_process(model_config, shared_buffer, model_queue, stop_event):
    """
    模型训练进程
    
    model_config: 模型配置参数
    shared_buffer: 共享的经验回放缓冲区
    model_queue: 发送最新模型参数的队列
    stop_event: 停止事件，用于通知训练进程退出
    """
    # 设置训练进程使用GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # 创建环境和模型
    sim_env = Simulator(
        model_config['n_vehs'], 
        model_config['sectors'], 
        model_config['n_vehs_in_state']
    )
    
    # 创建训练模型
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
        shared_buffer, 
        model_config['eval_freq'],
        model_config['update_freq'], 
        model_config['save_freq']
    )
    
    model.policy_net.to(device)
    model.target_net.to(device)
    
    # 清理日志
    with open('logs/training_log.txt', 'w') as f:
        f.write('')

    with open('logs/loss_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Avg_Loss', 'Epsilon'])
    
    print("Trainer process initialized. Waiting for sufficient data...")
    
    # 等待足够的数据收集
    while len(shared_buffer) < model_config['batch_size'] and not stop_event.is_set():
        time.sleep(1)
    
    print(f"Starting training with buffer size: {len(shared_buffer)}")
    
    # 开始训练过程
    iteration = 0
    update_freq = model_config['trainer_update_freq']  # 每隔多少次迭代更新一次工作进程的模型
    collected_losses = []
    
    while iteration < model_config['max_trainer_iterations'] and not stop_event.is_set():
        # 如果经验缓冲区为空，等待数据
        if len(shared_buffer) < model_config['batch_size']:
            time.sleep(0.1)
            continue
        
        # 从经验回放缓冲区中采样批次
        batch = shared_buffer.sample(model_config['batch_size'])
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
        collected_losses.append(loss.item())
        
        # 更新网络
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        # 记录训练信息
        if iteration % 100 == 0:
            avg_loss = np.mean(collected_losses) if collected_losses else 0
            print(f"Trainer: Iteration {iteration}, Loss: {avg_loss}, Buffer size: {len(shared_buffer)}")
            with open('logs/loss_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, avg_loss, model_config['epsilon']])
            collected_losses = []
        
        # 定期更新目标网络
        if iteration % model_config['update_freq'] == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())
        
        # 定期发送更新后的模型参数给工作进程
        if iteration % update_freq == 0:
            # 将模型参数转移到CPU再发送
            cpu_state_dict = {k: v.cpu() for k, v in model.policy_net.state_dict().items()}
            model_queue.put(cpu_state_dict)
            print(f"Trainer: Sent updated model parameters at iteration {iteration}")
        
        # 定期保存模型
        if iteration % model_config['save_freq'] == 0:
            model.save(f'Circular_DQN_iter_{iteration}')
        
        # 定期评估模型
        if iteration % model_config['eval_freq'] == 0:
            reward, non_greedy = model.eval(device)
            with open('logs/training_log.txt', 'a') as f:
                f.write(f"Iteration {iteration}, Reward: {reward}, Non-Greedy: {non_greedy:.2%}, Loss: {np.mean(collected_losses) if collected_losses else 0}\n")
        
        iteration += 1
    
    print("Trainer process completed")
    # 发送最终模型
    cpu_state_dict = {k: v.cpu() for k, v in model.policy_net.state_dict().items()}
    model_queue.put(cpu_state_dict)
    # 设置停止事件，通知其他进程退出
    stop_event.set()

def main():
    # 设置多处理方法
    mp.set_start_method('spawn', force=True)
    
    # 清理日志
    with open('logs/training_log.txt', 'w') as f:
        f.write('')

    with open('logs/loss_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Avg_Loss', 'Epsilon'])

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
    total_eps = 50001
    total_its = 1000
    eval_freq = 1000
    update_freq = 10
    save_freq = 1000
    
    # 新的并行训练参数
    n_cpus = mp.cpu_count()
    n_collection_workers = min(n_cpus - 1, 23 if torch.cuda.is_available() else 11)  # 保留一个CPU用于训练
    trainer_update_freq = 50  # 训练进程多少次迭代后更新工作进程的模型
    max_trainer_iterations = total_eps * total_its // batch_size  # 训练的最大迭代次数
    
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
        'n_collection_workers': n_collection_workers,
        'trainer_update_freq': trainer_update_freq,
        'max_trainer_iterations': max_trainer_iterations
    }
    
    # 创建共享经验回放缓冲区
    replay_buffer_capacity = int(100*1024*1024)
    shared_buffer = ReplayBuffer(replay_buffer_capacity)
    
    # 创建模型参数队列和停止事件
    model_queue = mp.Queue()
    stop_event = mp.Event()
    
    # 启动训练进程
    trainer = mp.Process(target=trainer_process, args=(model_config, shared_buffer, model_queue, stop_event))
    trainer.start()
    
    # 启动数据收集进程
    collectors = []
    for i in range(n_collection_workers):
        collector = mp.Process(target=data_collection_worker, args=(i, model_config, shared_buffer, model_queue, stop_event))
        collector.start()
        collectors.append(collector)
    
    try:
        # 等待训练进程完成
        trainer.join()
        
        # 训练完成后，通知所有数据收集进程退出
        stop_event.set()
        
        # 等待所有数据收集进程完成
        for collector in collectors:
            collector.join()
            
        print("All processes completed successfully.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        stop_event.set()
        
        # 等待所有进程结束
        trainer.join()
        for collector in collectors:
            collector.join()
        
        print("All processes terminated.")
    except Exception as e:
        print(f"Error occurred: {e}")
        stop_event.set()
        
        # 等待所有进程结束
        trainer.join()
        for collector in collectors:
            collector.join()
        
        print("All processes terminated due to error.")

if __name__ == "__main__":
    main()