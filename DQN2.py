try:
    import gym
    from gym import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("警告: gym 库未安装，控制台模式仍可正常使用")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import re
import os
from collections import deque, namedtuple
from MoonChess import MoonChessEnv

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'next_valid_mask', 'reward', 'done'))

class PrioritizedReplayMemory:
    """优先级经验回放缓冲区"""
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
    """DQN神经网络，添加BatchNorm层"""
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

class MoonChessDQNTrainer:
    """月亮棋DQN训练器"""
    def __init__(self):
        self.env = MoonChessEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_dim = 9 + 3 + 3 + 1
        self.output_dim = 9
        
        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.opponent_net = DQN(self.input_dim, self.output_dim).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.opponent_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        
        self.memory = PrioritizedReplayMemory(500000)
        
        self.batch_size = 64
        self.gamma = 0.95
        self.tau = 0.005
        
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_start = 1.0
        self.epsilon_decay_steps = 200000
        
        self.num_episodes = 500000
        self.start_episode = 0
        
        self.history_model_lib = []
        self.latest_opponent_net = None
        
        self.win_reward = 10.0
        self.lose_penalty = -10.0
        self.draw_reward = 0.0
        self.invalid_penalty = -1.0
        
        self.load_latest_model()
        self.load_history_models()
    
    def find_all_models(self):
        """查找所有模型文件"""
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
        """查找最新的模型文件"""
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
        """加载所有历史模型到历史模型库"""
        model_files = self.find_all_models()
        for episode, filename in model_files:
            try:
                net = DQN(self.input_dim, self.output_dim).to(self.device)
                checkpoint = torch.load(f"Models2/{filename}", map_location=self.device)
                net.load_state_dict(checkpoint)
                net.eval()
                self.history_model_lib.append((episode, net))
                print(f"已加载历史模型: Models2/{filename} (第 {episode} 回合)")
            except Exception as e:
                print(f"加载历史模型失败: {filename}, 错误: {e}")
        
        if self.history_model_lib:
            self.latest_opponent_net = self.history_model_lib[-1][1]
            self.opponent_net.load_state_dict(self.latest_opponent_net.state_dict())
    
    def load_latest_model(self):
        """加载最新的训练模型"""
        latest = self.find_latest_model()
        if latest:
            episode, filename = latest
            try:
                checkpoint = torch.load(f"Models2/{filename}", map_location=self.device)
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.opponent_net.load_state_dict(self.policy_net.state_dict())
                self.start_episode = episode
                
                decay_progress = min(episode / self.epsilon_decay_steps, 1.0)
                self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (1 - decay_progress)
                
                print(f"已加载模型: Models2/{filename} (第 {episode} 回合)")
                print(f"当前ε: {self.epsilon:.6f}")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("使用随机初始化的对手网络")
                self.opponent_net = DQN(self.input_dim, self.output_dim).to(self.device)
        else:
            print("未找到现有模型，从头开始训练")
    
    def encode_state(self, env, current_player):
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
        
        return torch.FloatTensor(state).to(self.device)
    
    def select_action(self, state, epsilon, valid_actions, net=None):
        if net is None:
            net = self.policy_net
            
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
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
    
    def get_valid_actions(self, env):
        valid = []
        for i in range(9):
            if env.board[i] == 0:
                current_history = env.history_x if env.current_player == 1 else env.history_o
                if len(current_history) < 3 or i != current_history[-3]:
                    valid.append(i)
        return valid
    
    def get_valid_mask(self, env):
        mask = torch.zeros(9, dtype=torch.bool, device=self.device)
        valid_actions = self.get_valid_actions(env)
        for i in valid_actions:
            mask[i] = True
        return mask
    
    def optimize_model(self):
        batch_data, indices, weights = self.memory.sample(self.batch_size)
        if batch_data is None:
            return
        
        batch = Transition(*zip(*batch_data))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.stack(batch.next_state)
        next_valid_mask_batch = torch.stack(batch.next_valid_mask)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        weights = weights.to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        with torch.no_grad():
            next_policy_q = self.policy_net(next_state_batch)
            next_policy_q = torch.where(next_valid_mask_batch, next_policy_q, torch.tensor(-1e9, device=self.device))
            next_best_actions = next_policy_q.argmax(1).unsqueeze(1)
            
            next_target_q = self.target_net(next_state_batch)
            next_q_values = next_target_q.gather(1, next_best_actions).squeeze()
            
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
            target_q_values = torch.clamp(target_q_values, -15, 15)
        
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.memory.update_priorities(indices, td_errors)
    
    def update_target_net(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def select_opponent(self, episode):
        """选择对手"""
        if random.random() < 0.05:
            return 'random', None
        
        if episode < 2500:
            p = 1.0
        else:
            progress = (episode - 2500) / (500000 - 2500)
            p = 0.8 - 0.3 * progress
            p = max(0.5, min(0.8, p))
        
        if random.random() < p:
            return 'latest', self.latest_opponent_net if self.latest_opponent_net else self.opponent_net
        elif self.history_model_lib:
            idx = random.randint(0, len(self.history_model_lib) - 1)
            return f'history_{self.history_model_lib[idx][0]}', self.history_model_lib[idx][1]
        else:
            return 'latest', self.opponent_net
    
    def evaluate_ai(self, num_games=50):
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
                    state = self.encode_state(eval_env, 1)
                    valid = self.get_valid_actions(eval_env)
                    action = self.select_action(state, 0.0, valid)
                    _, _, done, truncated, _ = eval_env.step(action)
                else:
                    valid = self.get_valid_actions(eval_env)
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
                    state = self.encode_state(eval_env, -1)
                    valid = self.get_valid_actions(eval_env)
                    action = self.select_action(state, 0.0, valid)
                    _, _, done, truncated, _ = eval_env.step(action)
                else:
                    valid = self.get_valid_actions(eval_env)
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
    
    def train(self):
        print(f"开始训练，使用设备: {self.device}")
        print(f"训练回合数: {self.start_episode + 1} 到 {self.num_episodes}")
        print(f"历史模型库大小: {len(self.history_model_lib)}")
        print(f"奖励配置：赢={self.win_reward}, 输={self.lose_penalty}, 平局={self.draw_reward}")
        
        for episode in range(self.start_episode, self.num_episodes):
            self.env.reset()
            done = False
            episode_memory = []
            
            current_is_first = (episode % 2 == 1)
            
            opponent_type, opponent_net_instance = self.select_opponent(episode)
            
            if current_is_first:
                first_net = self.policy_net
                second_net = opponent_net_instance if opponent_net_instance else None
                first_epsilon = self.epsilon
                second_epsilon = 0.0
            else:
                first_net = opponent_net_instance if opponent_net_instance else None
                second_net = self.policy_net
                first_epsilon = 0.0
                second_epsilon = self.epsilon
            
            while not done:
                current_player = self.env.current_player
                state = self.encode_state(self.env, current_player)
                valid_actions = self.get_valid_actions(self.env)
                
                if not valid_actions:
                    break
                
                if current_player == 1:
                    net = first_net
                    epsilon = first_epsilon
                else:
                    net = second_net
                    epsilon = second_epsilon
                
                if net is None or opponent_type == 'random':
                    action = random.choice(valid_actions)
                else:
                    action = self.select_action(state, epsilon, valid_actions, net)
                
                obs, raw_reward, done, truncated, _ = self.env.step(action)
                
                if truncated:
                    done = True
                
                final_reward = raw_reward
                if done:
                    win, winner = self.env._check_win()
                    if win:
                        final_reward = self.win_reward if current_player == winner else self.lose_penalty
                    else:
                        final_reward = self.draw_reward
                
                next_player = self.env.current_player
                next_state = self.encode_state(self.env, next_player)
                
                if done:
                    next_valid_mask = torch.zeros(9, dtype=torch.bool, device=self.device)
                else:
                    next_valid_mask = self.get_valid_mask(self.env)
                
                episode_memory.append((state, action, next_state, next_valid_mask, final_reward, done))
                
                self.optimize_model()
                self.update_target_net()
            
            win, winner = self.env._check_win()
            if win:
                if len(episode_memory) >= 2:
                    state, action, next_state, next_valid_mask, _, _ = episode_memory[-2]
                    episode_memory[-2] = (state, action, next_state, next_valid_mask, self.lose_penalty, True)
            else:
                if len(episode_memory) > 0:
                    state, action, next_state, next_valid_mask, _, done_flag = episode_memory[-1]
                    episode_memory[-1] = (state, action, next_state, next_valid_mask, self.draw_reward, done_flag)
            
            for exp in episode_memory:
                self.memory.push(*exp)
            
            if episode < 2500:
                if (episode + 1) % 500 == 0:
                    self.opponent_net.load_state_dict(self.policy_net.state_dict())
                    self.latest_opponent_net = DQN(self.input_dim, self.output_dim).to(self.device)
                    self.latest_opponent_net.load_state_dict(self.policy_net.state_dict())
                    self.latest_opponent_net.eval()
                    print(f"历史策略已更新 (第 {episode + 1} 回合)")
            else:
                if (episode + 1) % 2500 == 0:
                    self.save_models(episode + 1)
                    new_net = DQN(self.input_dim, self.output_dim).to(self.device)
                    new_net.load_state_dict(self.policy_net.state_dict())
                    new_net.eval()
                    self.history_model_lib.append((episode + 1, new_net))
                    self.latest_opponent_net = new_net
                    self.opponent_net.load_state_dict(self.policy_net.state_dict())
                    print(f"模型已保存并加入历史模型库 (第 {episode + 1} 回合), 历史库大小: {len(self.history_model_lib)}")
            
            if episode < 2500:
                if (episode + 1) % 2500 == 0:
                    self.save_models(episode + 1)
            else:
                pass
            
            if episode < self.epsilon_decay_steps:
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (episode / self.epsilon_decay_steps)
                )
            
            if (episode + 1) % 100 == 0:
                win, winner = self.env._check_win()
                if win:
                    result = "先手胜" if winner == 1 else "后手胜"
                elif truncated:
                    result = "步数超限平局"
                else:
                    result = "平局"
                print(f"回合: {episode + 1}/{self.num_episodes}, {result}, 对手: {opponent_type}, ε: {self.epsilon:.6f}")
            
            if (episode + 1) % 1000 == 0:
                (first_win, first_draw, first_lose), (second_win, second_draw, second_lose) = self.evaluate_ai(num_games=30)
                print(f"  对战随机 - 先手: 胜率 {first_win:.2f}, 平局 {first_draw:.2f}, 败率 {first_lose:.2f}")
                print(f"  对战随机 - 后手: 胜率 {second_win:.2f}, 平局 {second_draw:.2f}, 败率 {second_lose:.2f}")
        
        print("训练完成!")
        self.save_models("final")
    
    def save_models(self, suffix):
        if not os.path.exists('Models2'):
            os.makedirs('Models2')
        save_path = f"Models2/moonchess_policy_{suffix}.pth"
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"模型已保存: {save_path}")
    
    def play_vs_agent(self):
        self.policy_net.eval()
        
        env = MoonChessEnv()
        env.reset()
        
        print("=== 月亮棋人机对战 ===")
        print("你是X (先手)，AI是O (后手)")
        print("棋盘位置编号：")
        print("0 | 1 | 2")
        print("3 | 4 | 5")
        print("6 | 7 | 8")
        
        while not env.done:
            env.render()
            
            if env.current_player == 1:
                try:
                    valid = self.get_valid_actions(env)
                    action = int(input(f"\n请输入落子位置 (有效位置: {valid}): "))
                    if action not in valid:
                        print("无效位置! 请选择有效位置。")
                        continue
                    obs, reward, done, truncated, _ = env.step(action)
                except ValueError:
                    print("请输入有效的数字!")
                    continue
            else:
                state = self.encode_state(env, -1)
                valid = self.get_valid_actions(env)
                
                with torch.no_grad():
                    q_values = self.policy_net(state.unsqueeze(0))
                    q_values_np = q_values.cpu().numpy()[0]
                    print("\nAI各位置Q值:")
                    for i in range(9):
                        validity = "✓" if i in valid else "✗"
                        print(f"  位置 {i}: {q_values_np[i]:.4f} {validity}")
                    for i in range(9):
                        if i not in valid:
                            q_values_np[i] = -np.inf
                    action = int(np.argmax(q_values_np))
                
                print(f"\nAI选择了位置: {action}")
                obs, reward, done, truncated, _ = env.step(action)
            
            if truncated:
                break
        
        env.render()
        win, winner = env._check_win()
        if win:
            print(f"\n游戏结束! {'你' if winner == 1 else 'AI'} 获胜!")
        else:
            print("\n游戏结束! 平局!")

if __name__ == "__main__":
    trainer = MoonChessDQNTrainer()
    trainer.train()
