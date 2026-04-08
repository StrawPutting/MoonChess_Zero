try:
    import gym
    from gym import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("警告: gym 库未安装，控制台模式仍可正常使用", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import re
import os
import time
from collections import deque, namedtuple
from MoonChess import MoonChessEnv
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Value
import signal
import sys

# 设置多进程启动方式为spawn（避免CUDA问题）
mp.set_start_method('spawn', force=True)

# 定义经验数据结构（使用numpy而非tensor，方便多进程传递）
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'next_valid_mask', 'reward', 'done'))

class PrioritizedReplayMemory:
    """优先级经验回放缓冲区（主进程管理）"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.0001, epsilon=1e-6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        self.epsilon = epsilon
    
    def push(self, *args):
        """存入一条经验，初始优先级设为当前最大优先级"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = Transition(*args)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """按优先级采样一批经验，返回经验、索引和重要性权重"""
        if len(self.memory) < batch_size:
            return None, None, None
        
        # 计算采样概率
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # 采样
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # 计算重要性采样权重
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.memory[i] for i in indices]
        return batch, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, td_errors):
        """更新采样经验的优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """DQN神经网络，添加BatchNorm层（保持原结构）"""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        """前向传播"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_single = True
        else:
            is_single = False
        
        is_training = self.training
        batch_size = x.size(0)
        
        if is_training and batch_size == 1:
            self.eval()
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        
        if is_training and batch_size == 1:
            self.train()
        
        if is_single:
            x = x.squeeze(0)
        return x
    
    def get_state_dict_numpy(self):
        """将state_dict转为numpy格式（方便多进程传递）"""
        state_dict = self.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k].cpu().numpy()
        return state_dict
    
    def load_state_dict_numpy(self, state_dict_np):
        """从numpy格式加载state_dict"""
        state_dict = {}
        for k in state_dict_np.keys():
            state_dict[k] = torch.from_numpy(state_dict_np[k])
        self.load_state_dict(state_dict)

def encode_state(env, current_player, input_dim):
    """独立的状态编码函数（供子进程使用）"""
    board_state = env.board.copy().astype(np.float32)
    if current_player == -1:
        board_state = -board_state
    
    history_x = np.zeros(3, dtype=np.float32)
    for i in range(min(3, len(env.history_x))):
        history_x[2 - i] = env.history_x[-(i + 1)]
    history_x = np.where(np.arange(3) < (3 - len(env.history_x)), -1.0, history_x / 4.0 - 1.0)
    
    history_o = np.zeros(3, dtype=np.float32)
    for i in range(min(3, len(env.history_o))):
        history_o[2 - i] = env.history_o[-(i + 1)]
    history_o = np.where(np.arange(3) < (3 - len(env.history_o)), -1.0, history_o / 4.0 - 1.0)
    
    if current_player == -1:
        my_history = history_o
        opp_history = history_x
    else:
        my_history = history_x
        opp_history = history_o
    
    current_history = env.history_x if current_player == 1 else env.history_o
    disappear_pos = np.array([-1.0], dtype=np.float32)
    if len(current_history) >= 3:
        disappear_pos[0] = (current_history[-3] / 4.0) - 1.0
    
    state = np.concatenate([
        board_state,
        my_history,
        opp_history,
        disappear_pos
    ])
    
    return state

def get_valid_actions(env):
    """独立的有效动作获取函数（供子进程使用）"""
    valid = []
    for i in range(9):
        if env.board[i] == 0:
            current_history = env.history_x if env.current_player == 1 else env.history_o
            if len(current_history) < 3 or i != current_history[-3]:
                valid.append(i)
    return valid

def get_valid_mask(env, device):
    """独立的有效掩码获取函数（供子进程使用）"""
    mask = torch.zeros(9, dtype=torch.bool, device=device)
    valid_actions = get_valid_actions(env)
    for i in valid_actions:
        mask[i] = True
    return mask.cpu().numpy()

def select_action(state_np, epsilon, valid_actions, net, device):
    """独立的动作选择函数（供子进程使用）"""
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        state = torch.FloatTensor(state_np).to(device)
        was_training = net.training
        net.eval()
        with torch.no_grad():
            q_values = net(state.unsqueeze(0))
            q_values_np = q_values.cpu().numpy()[0]
            for i in range(9):
                if i not in valid_actions:
                    q_values_np[i] = -np.inf
            action = int(np.argmax(q_values_np))
        if was_training:
            net.train()
        return action

def select_opponent(episode, history_model_indices, latest_opponent_available):
    """独立的对手选择函数（供子进程使用）"""
    if random.random() < 0.05:
        return 'random', None
    
    if episode < 2500:
        p = 1.0
    else:
        progress = (episode - 2500) / (500000 - 2500)
        p = 0.8 - 0.3 * progress
        p = max(0.5, min(0.8, p))
    
    if random.random() < p:
        return 'latest', 0  # 0代表最新模型
    elif history_model_indices:
        idx = random.randint(0, len(history_model_indices) - 1)
        return f'history_{history_model_indices[idx]}', idx + 1  # 1+代表历史模型索引
    else:
        return 'latest', 0

def worker_process(
    worker_id,
    input_dim,
    output_dim,
    experience_queue,
    param_queue,
    history_model_queue,
    episode_counter,
    epsilon_value,
    stop_flag,
    win_reward,
    lose_penalty,
    draw_reward,
    invalid_penalty
):
    """子进程：经验收集工作进程"""
    # 子进程使用CPU
    device = torch.device("cpu")
    env = MoonChessEnv()
    policy_net = DQN(input_dim, output_dim).to(device)
    opponent_net = DQN(input_dim, output_dim).to(device)
    history_nets = {}
    
    # 初始化获取最新参数
    latest_params = None
    try:
        latest_params = param_queue.get(timeout=30)
        policy_net.load_state_dict_numpy(latest_params)
        print(f"Worker {worker_id}: 已加载初始模型", flush=True)
    except:
        print(f"Worker {worker_id}: 超时未收到模型，使用随机初始化", flush=True)
    
    # 接收历史模型（最多等5秒）
    hist_received = 0
    hist_start_time = time.time()
    while (time.time() - hist_start_time) < 5.0:
        try:
            msg = history_model_queue.get(timeout=0.2)
            if msg is None:  # 结束标记
                break
            hist_idx, hist_params = msg
            history_nets[hist_idx] = DQN(input_dim, output_dim).to(device)
            history_nets[hist_idx].load_state_dict_numpy(hist_params)
            history_nets[hist_idx].eval()
            hist_received += 1
        except:
            continue
    
    print(f"Worker {worker_id}: 已接收 {hist_received} 个历史模型", flush=True)
    print(f"工作进程 {worker_id} 初始化完成", flush=True)
    
    while not stop_flag.value:
        # 获取当前训练进度（只读，不修改）
        current_episode = episode_counter.value
        if current_episode >= 500000:
            break
        
        # 更新模型参数（优先获取最新参数）
        try:
            while True:
                new_params = param_queue.get_nowait()
                policy_net.load_state_dict_numpy(new_params)
        except:
            pass
        
        # 接收新的历史模型（如果有的话）
        try:
            while True:
                msg = history_model_queue.get_nowait()
                if msg is not None:
                    hist_idx, hist_params = msg
                    history_nets[hist_idx] = DQN(input_dim, output_dim).to(device)
                    history_nets[hist_idx].load_state_dict_numpy(hist_params)
                    history_nets[hist_idx].eval()
        except:
            pass
        
        # 获取当前epsilon
        current_epsilon = epsilon_value.value
        
        # 开始单局训练
        env.reset()
        done = False
        episode_memory = []
        truncated = False
        
        # 获取历史模型索引列表
        history_model_indices = [k for k in history_nets.keys() if k > 0]
        
        # 选择对手
        current_is_first = (current_episode % 2 == 1)
        opponent_type, opp_model_idx = select_opponent(
            current_episode, 
            history_model_indices,
            latest_opponent_available=True
        )
        
        # 加载对手模型
        if opponent_type == 'random':
            opp_net = None
        elif opp_model_idx == 0:
            opp_net = policy_net  # 最新模型
        else:
            opp_net = history_nets.get(opp_model_idx, policy_net)
        
        # 设置双方网络和epsilon
        if current_is_first:
            first_net = policy_net
            second_net = opp_net
            first_epsilon = current_epsilon
            second_epsilon = 0.0
        else:
            first_net = opp_net
            second_net = policy_net
            first_epsilon = 0.0
            second_epsilon = current_epsilon
        
        # 对局循环
        while not done and not stop_flag.value:
            current_player = env.current_player
            state_np = encode_state(env, current_player, input_dim)
            valid_actions = get_valid_actions(env)
            
            if not valid_actions:
                break
            
            # 选择动作
            if current_player == 1:
                net = first_net
                epsilon = first_epsilon
            else:
                net = second_net
                epsilon = second_epsilon
            
            if net is None or opponent_type == 'random':
                action = random.choice(valid_actions)
            else:
                action = select_action(state_np, epsilon, valid_actions, net, device)
            
            # 执行动作
            try:
                obs, raw_reward, done, truncated, _ = env.step(action)
            except:
                done = True
                truncated = True
                continue
            
            # 计算最终奖励
            final_reward = raw_reward
            if done:
                win, winner = env._check_win()
                if win:
                    final_reward = win_reward if current_player == winner else lose_penalty
                else:
                    final_reward = draw_reward
            
            # 编码下一个状态
            next_player = env.current_player
            next_state_np = encode_state(env, next_player, input_dim)
            
            # 获取下一个有效掩码
            if done:
                next_valid_mask_np = np.zeros(9, dtype=bool)
            else:
                next_valid_mask_np = get_valid_mask(env, device)
            
            # 存储经验
            episode_memory.append((
                state_np,
                action,
                next_state_np,
                next_valid_mask_np,
                final_reward,
                done
            ))
        
        # 修正最后一步奖励（平局/胜负）
        win, winner = env._check_win()
        if win:
            if len(episode_memory) >= 2:
                state, action, next_state, next_valid_mask, _, _ = episode_memory[-2]
                episode_memory[-2] = (state, action, next_state, next_valid_mask, lose_penalty, True)
        else:
            if len(episode_memory) > 0:
                state, action, next_state, next_valid_mask, _, done_flag = episode_memory[-1]
                episode_memory[-1] = (state, action, next_state, next_valid_mask, draw_reward, done_flag)
        
        # 将经验放入队列（完整的一局为单位）
        try:
            experience_queue.put(episode_memory, timeout=1.0)
        except:
            time.sleep(0.05)
    
    print(f"工作进程 {worker_id} 退出", flush=True)

class MoonChessDQNTrainerMP:
    """多进程月亮棋DQN训练器（保持原功能）"""
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.env = MoonChessEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 输入输出维度（保持原配置）
        self.input_dim = 9 + 3 + 3 + 1
        self.output_dim = 9
        
        # 初始化网络
        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.opponent_net = DQN(self.input_dim, self.output_dim).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        
        # 优先级经验回放（主进程管理）
        self.memory = PrioritizedReplayMemory(500000)
        
        # 训练超参数（保持原配置）
        self.batch_size = 64
        self.gamma = 0.95
        self.tau = 0.005
        
        self.epsilon = Value('d', 1.0)  # 共享epsilon（double类型）
        self.epsilon_min = 0.1
        self.epsilon_start = 1.0
        self.epsilon_decay_steps = 200000
        
        self.num_episodes = 500000
        # 主进程单独管理 episode 计数，不使用共享变量
        self.start_episode = 0
        self.current_episode = 0
        
        self.stop_flag = Value('b', False)    # 停止标志（bool类型）
        
        # 奖励配置（保持原配置）
        self.win_reward = 10.0
        self.lose_penalty = -10.0
        self.draw_reward = 0.0
        self.invalid_penalty = -1.0
        
        # 多进程通信队列 - 简化版本，不用Manager
        self.experience_queue = Queue(maxsize=500)  # 经验队列（一局为单位）
        self.param_queue = Queue(maxsize=self.num_workers * 3)  # 参数队列
        self.history_model_queue = Queue(maxsize=100)  # 历史模型队列
        
        # 历史模型库
        self.history_model_lib = []
        self.latest_opponent_net = None
        
        # 加载模型
        self.load_latest_model()
        self.load_history_models()
        
        # 工作进程列表
        self.workers = []
    
    def find_all_models(self):
        """查找所有模型文件（保持原逻辑）"""
        model_files = []
        pattern = re.compile(r'moonchess_policy_(\d+)\.pth')
        
        if not os.path.exists('Models2'):
            os.makedirs('Models2')
        
        for filename in os.listdir('Models2'):
            match = pattern.match(filename)
            if match:
                episode = int(match.group(1))
                if episode >= 2500:
                    model_files.append((episode, filename))
        
        model_files.sort(key=lambda x: x[0])
        return model_files
    
    def find_latest_model(self):
        """查找最新的模型文件（保持原逻辑）"""
        model_files = []
        pattern = re.compile(r'moonchess_policy_(\d+)\.pth')
        
        if not os.path.exists('Models2'):
            os.makedirs('Models2')
        
        for filename in os.listdir('Models2'):
            match = pattern.match(filename)
            if match:
                episode = int(match.group(1))
                model_files.append((episode, filename))
        
        if not model_files:
            return None
        
        model_files.sort(reverse=True, key=lambda x: x[0])
        return model_files[0]
    
    def load_history_models(self):
        """加载所有历史模型到历史模型库（保持原逻辑）"""
        model_files = self.find_all_models()
        for episode, filename in model_files:
            try:
                net = DQN(self.input_dim, self.output_dim).to(self.device)
                checkpoint = torch.load(f"Models2/{filename}", map_location=self.device)
                net.load_state_dict(checkpoint)
                net.eval()
                self.history_model_lib.append((episode, net))
                print(f"已加载历史模型: Models2/{filename} (第 {episode} 回合)", flush=True)
            except Exception as e:
                print(f"加载历史模型失败: {filename}, 错误: {e}", flush=True)
        
        if self.history_model_lib:
            self.latest_opponent_net = self.history_model_lib[-1][1]
            self.opponent_net.load_state_dict(self.latest_opponent_net.state_dict())
    
    def load_latest_model(self):
        """加载最新的训练模型（保持原逻辑）"""
        latest = self.find_latest_model()
        if latest:
            episode, filename = latest
            try:
                checkpoint = torch.load(f"Models2/{filename}", map_location=self.device)
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.opponent_net.load_state_dict(self.policy_net.state_dict())
                
                self.start_episode = episode
                self.current_episode = episode
                
                # 更新epsilon
                decay_progress = min(episode / self.epsilon_decay_steps, 1.0)
                with self.epsilon.get_lock():
                    self.epsilon.value = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (1 - decay_progress)
                
                print(f"已加载模型: Models2/{filename} (第 {episode} 回合)", flush=True)
                print(f"当前ε: {self.epsilon.value:.6f}", flush=True)
            except Exception as e:
                print(f"加载模型失败: {e}", flush=True)
                print("使用随机初始化的对手网络", flush=True)
                self.opponent_net = DQN(self.input_dim, self.output_dim).to(self.device)
        else:
            print("未找到现有模型，从头开始训练", flush=True)
    
    def optimize_model(self):
        """优化模型（保持原逻辑）"""
        batch_data, indices, weights = self.memory.sample(self.batch_size)
        if batch_data is None:
            return
        
        batch = Transition(*zip(*batch_data))
        
        # 转换为tensor
        state_batch = torch.stack([torch.FloatTensor(s).to(self.device) for s in batch.state])
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.stack([torch.FloatTensor(s).to(self.device) for s in batch.next_state])
        next_valid_mask_batch = torch.stack([torch.BoolTensor(m).to(self.device) for m in batch.next_valid_mask])
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        weights = weights.to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            next_policy_q = self.policy_net(next_state_batch)
            next_policy_q = torch.where(next_valid_mask_batch, next_policy_q, torch.tensor(-1e9, device=self.device))
            next_best_actions = next_policy_q.argmax(1).unsqueeze(1)
            
            next_target_q = self.target_net(next_state_batch)
            next_q_values = next_target_q.gather(1, next_best_actions).squeeze()
            
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
            target_q_values = torch.clamp(target_q_values, -15, 15)
        
        # 计算TD误差和损失
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors)
    
    def update_target_net(self):
        """更新目标网络（保持原逻辑）"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def evaluate_ai(self, num_games=50):
        """评估AI性能（保持原逻辑）"""
        self.policy_net.eval()
        
        first_win = 0
        first_draw = 0
        first_lose = 0
        second_win = 0
        second_draw = 0
        second_lose = 0
        
        half_games = num_games // 2
        
        for _ in range(half_games):
            eval_env = MoonChessEnv()
            eval_env.reset()
            done = False
            
            while not done:
                if eval_env.current_player == 1:
                    state_np = encode_state(eval_env, 1, self.input_dim)
                    state = torch.FloatTensor(state_np).to(self.device)
                    valid = get_valid_actions(eval_env)
                    action = select_action(state_np, 0.0, valid, self.policy_net, self.device)
                    _, _, done, truncated, _ = eval_env.step(action)
                else:
                    valid = get_valid_actions(eval_env)
                    action = random.choice(valid)
                    _, _, done, truncated, _ = eval_env.step(action)
                
                if truncated:
                    done = True
            
            win, winner = eval_env._check_win()
            if winner == 1:
                first_win += 1
            elif winner == -1:
                first_lose += 1
            else:
                first_draw += 1
        
        for _ in range(half_games):
            eval_env = MoonChessEnv()
            eval_env.reset()
            done = False
            
            while not done:
                if eval_env.current_player == -1:
                    state_np = encode_state(eval_env, -1, self.input_dim)
                    state = torch.FloatTensor(state_np).to(self.device)
                    valid = get_valid_actions(eval_env)
                    action = select_action(state_np, 0.0, valid, self.policy_net, self.device)
                    _, _, done, truncated, _ = eval_env.step(action)
                else:
                    valid = get_valid_actions(eval_env)
                    action = random.choice(valid)
                    _, _, done, truncated, _ = eval_env.step(action)
                
                if truncated:
                    done = True
            
            win, winner = eval_env._check_win()
            if winner == -1:
                second_win += 1
            elif winner == 1:
                second_lose += 1
            else:
                second_draw += 1
        
        self.policy_net.train()
        
        return (first_win / half_games, first_draw / half_games, first_lose / half_games), \
               (second_win / half_games, second_draw / half_games, second_lose / half_games)
    
    def save_models(self, suffix):
        """保存模型（保持原逻辑）"""
        if not os.path.exists('Models2'):
            os.makedirs('Models2')
        save_path = f"Models2/moonchess_policy_{suffix}.pth"
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"模型已保存: {save_path}", flush=True)
    
    def start_workers(self):
        """启动工作进程 - 简化稳定版本"""
        # 创建共享的 episode_counter 只给 worker 读取
        shared_episode_counter = Value('i', self.current_episode)
        
        # 先启动所有 workers
        for i in range(self.num_workers):
            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    self.input_dim,
                    self.output_dim,
                    self.experience_queue,
                    self.param_queue,
                    self.history_model_queue,
                    shared_episode_counter,
                    self.epsilon,
                    self.stop_flag,
                    self.win_reward,
                    self.lose_penalty,
                    self.draw_reward,
                    self.invalid_penalty
                )
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        print(f"已启动 {self.num_workers} 个工作进程", flush=True)
        
        # 发送初始参数
        init_params = self.policy_net.get_state_dict_numpy()
        for _ in range(self.num_workers):
            self.param_queue.put(init_params)
        
        # 发送历史模型给 Workers（每个 worker 一份）
        if self.history_model_lib:
            for _ in range(self.num_workers):
                for idx, (episode, net) in enumerate(self.history_model_lib):
                    hist_params = net.get_state_dict_numpy()
                    self.history_model_queue.put((idx + 1, hist_params))
                # 发送结束标记
                self.history_model_queue.put(None)
        
        print(f"已发送 {len(self.history_model_lib)} 个历史模型给 Workers", flush=True)
        
        return shared_episode_counter
    
    def clear_experience_queue(self):
        """清空经验队列"""
        count_episodes = 0
        while not self.experience_queue.empty():
            try:
                self.experience_queue.get_nowait()
                count_episodes += 1
            except:
                break
        if count_episodes > 0:
            print(f"已清空经验队列，丢弃 {count_episodes} 局经验", flush=True)
    
    def train(self):
        """主训练循环"""
        print(f"开始多进程训练，使用设备: {self.device}", flush=True)
        print(f"训练回合数: {self.current_episode + 1} 到 {self.num_episodes}", flush=True)
        print(f"历史模型库大小: {len(self.history_model_lib)}", flush=True)
        print(f"奖励配置：赢={self.win_reward}, 输={self.lose_penalty}, 平局={self.draw_reward}", flush=True)
        print(f"工作进程数: {self.num_workers}", flush=True)
        
        # 启动工作进程
        shared_episode_counter = self.start_workers()
        
        # 主进程训练循环
        try:
            self.current_episode = self.start_episode
            while self.current_episode < self.num_episodes:
                # 从经验队列获取经验并存储（完整的一局为单位）
                # 必须拿到经验才算1个episode
                got_episode = False
                while not got_episode:
                    if self.stop_flag.value:
                        break
                    
                    try:
                        episode_memory = self.experience_queue.get(timeout=0.5)
                        got_episode = True
                        
                        # 存入经验回放
                        for exp in episode_memory:
                            state, action, next_state, next_valid_mask, reward, done = exp
                            state = state if isinstance(state, torch.Tensor) else torch.FloatTensor(state)
                            next_state = next_state if isinstance(next_state, torch.Tensor) else torch.FloatTensor(next_state)
                            next_valid_mask = next_valid_mask if isinstance(next_valid_mask, torch.Tensor) else torch.BoolTensor(next_valid_mask)
                            self.memory.push(state, action, next_state, next_valid_mask, reward, done)
                    except:
                        time.sleep(0.05)
                
                if self.stop_flag.value:
                    break
                
                # 更新共享的 episode counter 供 worker 读取
                with shared_episode_counter.get_lock():
                    shared_episode_counter.value = self.current_episode
                
                # 优化模型
                if len(self.memory) >= self.batch_size:
                    self.optimize_model()
                    self.update_target_net()
                
                # 更新epsilon
                if self.current_episode < self.epsilon_decay_steps:
                    new_epsilon = max(
                        self.epsilon_min,
                        self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (self.current_episode / self.epsilon_decay_steps)
                    )
                    with self.epsilon.get_lock():
                        self.epsilon.value = new_epsilon
                
                # 定期同步模型参数到子进程
                if (self.current_episode + 1) % 10 == 0:
                    try:
                        # 清空旧参数
                        while not self.param_queue.empty():
                            self.param_queue.get_nowait()
                        # 放入新参数
                        new_params = self.policy_net.get_state_dict_numpy()
                        for _ in range(self.num_workers):
                            self.param_queue.put(new_params)
                    except:
                        pass
                
                # 定期保存模型
                save_interval = 2500
                if self.current_episode < 2500:
                    if (self.current_episode + 1) % 500 == 0:
                        self.opponent_net.load_state_dict(self.policy_net.state_dict())
                        self.latest_opponent_net = DQN(self.input_dim, self.output_dim).to(self.device)
                        self.latest_opponent_net.load_state_dict(self.policy_net.state_dict())
                        self.latest_opponent_net.eval()
                        print(f"历史策略已更新 (第 {self.current_episode + 1} 回合)", flush=True)
                    
                    if (self.current_episode + 1) % save_interval == 0:
                        self.save_models(self.current_episode + 1)
                        # 清空经验队列
                        self.clear_experience_queue()
                else:
                    if (self.current_episode + 1) % save_interval == 0:
                        self.save_models(self.current_episode + 1)
                        new_net = DQN(self.input_dim, self.output_dim).to(self.device)
                        new_net.load_state_dict(self.policy_net.state_dict())
                        new_net.eval()
                        self.history_model_lib.append((self.current_episode + 1, new_net))
                        self.latest_opponent_net = new_net
                        self.opponent_net.load_state_dict(self.policy_net.state_dict())
                        print(f"模型已保存并加入历史模型库 (第 {self.current_episode + 1} 回合), 历史库大小: {len(self.history_model_lib)}", flush=True)
                        
                        # 把新历史模型发送给所有 Workers
                        new_hist_idx = len(self.history_model_lib)
                        new_hist_params = new_net.get_state_dict_numpy()
                        for _ in range(self.num_workers):
                            self.history_model_queue.put((new_hist_idx, new_hist_params))
                        
                        # 清空经验队列
                        self.clear_experience_queue()
                
                # 定期打印训练信息
                if (self.current_episode + 1) % 100 == 0:
                    print(f"回合: {self.current_episode + 1}/{self.num_episodes}, ε: {self.epsilon.value:.6f}, 经验池大小: {len(self.memory)}", flush=True)
                
                # 定期评估AI
                if (self.current_episode + 1) % 1000 == 0:
                    (first_win, first_draw, first_lose), (second_win, second_draw, second_lose) = self.evaluate_ai(num_games=30)
                    print(f"  对战随机 - 先手: 胜率 {first_win:.2f}, 平局 {first_draw:.2f}, 败率 {first_lose:.2f}", flush=True)
                    print(f"  对战随机 - 后手: 胜率 {second_win:.2f}, 平局 {second_draw:.2f}, 败率 {second_lose:.2f}", flush=True)
                
                # 增加episode计数
                self.current_episode += 1
            
            # 训练完成
            print("训练完成!", flush=True)
            self.save_models("final")
            
        except KeyboardInterrupt:
            print("\n接收到停止信号，正在关闭...", flush=True)
        finally:
            # 停止所有工作进程
            self.stop_flag.value = True
            for worker in self.workers:
                worker.join(timeout=5)
            print("所有工作进程已关闭", flush=True)
    
    def play_vs_agent(self):
        """人机对战（保持原功能）"""
        self.policy_net.eval()
        
        env = MoonChessEnv()
        env.reset()
        
        print("=== 月亮棋人机对战 ===", flush=True)
        print("你是X (先手)，AI是O (后手)", flush=True)
        print("棋盘位置编号：", flush=True)
        print("0 | 1 | 2", flush=True)
        print("3 | 4 | 5", flush=True)
        print("6 | 7 | 8", flush=True)
        
        while not env.done:
            env.render()
            
            if env.current_player == 1:
                try:
                    valid = get_valid_actions(env)
                    action = int(input(f"\n请输入落子位置 (有效位置: {valid}): "))
                    if action not in valid:
                        print("无效位置! 请选择有效位置。", flush=True)
                        continue
                    obs, reward, done, truncated, _ = env.step(action)
                except ValueError:
                    print("请输入有效的数字!", flush=True)
                    continue
            else:
                state_np = encode_state(env, -1, self.input_dim)
                state = torch.FloatTensor(state_np).to(self.device)
                valid = get_valid_actions(env)
                
                with torch.no_grad():
                    q_values = self.policy_net(state.unsqueeze(0))
                    q_values_np = q_values.cpu().numpy()[0]
                    print("\nAI各位置Q值:", flush=True)
                    for i in range(9):
                        validity = "✓" if i in valid else "✗"
                        print(f"  位置 {i}: {q_values_np[i]:.4f} {validity}", flush=True)
                    for i in range(9):
                        if i not in valid:
                            q_values_np[i] = -np.inf
                    action = int(np.argmax(q_values_np))
                
                print(f"\nAI选择了位置: {action}", flush=True)
                obs, reward, done, truncated, _ = env.step(action)
            
            if truncated:
                break
        
        env.render()
        win, winner = env._check_win()
        if win:
            print(f"\n游戏结束! {'你' if winner == 1 else 'AI'} 获胜!", flush=True)
        else:
            print("\n游戏结束! 平局!", flush=True)

def signal_handler(sig, frame):
    """信号处理：优雅退出"""
    print('\n接收到退出信号，正在清理...', flush=True)
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 创建训练器（可调整工作进程数）
    trainer = MoonChessDQNTrainerMP(num_workers=6)
    
    # 启动训练
    trainer.train()
    
    # 可选：启动人机对战
    # trainer.play_vs_agent()
