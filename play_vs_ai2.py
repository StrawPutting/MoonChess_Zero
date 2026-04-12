import torch
import numpy as np
import copy
from MoonChess import MoonChessEnv
from DQN2 import DQN

class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_terminal = False
        self.terminal_value = 0
        
        win, winner = env._check_win()
        if win:
            self.is_terminal = True
            if winner == env.current_player:
                self.terminal_value = 1
            else:
                self.terminal_value = -1
        elif env.done:
            self.is_terminal = True
            self.terminal_value = 0
    
    def is_fully_expanded(self):
        valid_actions = get_valid_actions(self.env)
        return len(self.children) == len(valid_actions)
    
    def get_ucb(self, exploration_weight=1.0):
        if self.visit_count == 0:
            return float('inf')
        mean_value = self.value_sum / self.visit_count
        parent_visit = self.parent.visit_count if self.parent else 1
        ucb = mean_value + exploration_weight * np.sqrt(np.log(parent_visit) / self.visit_count)
        return ucb
    
    def expand(self):
        valid_actions = get_valid_actions(self.env)
        for action in valid_actions:
            if action not in self.children:
                new_env = copy.deepcopy(self.env)
                new_env.step(action)
                child_node = MCTSNode(new_env, parent=self, action=action)
                self.children[action] = child_node
                return child_node
        return None

def mcts_search(env, num_simulations=100, max_depth=3, exploration_weight=1.0):
    original_player = env.current_player
    root = MCTSNode(copy.deepcopy(env))
    
    for _ in range(num_simulations):
        node = root
        depth = 0
        
        while node.is_fully_expanded() and not node.is_terminal and depth < max_depth:
            node = select_best_child(node, exploration_weight)
            depth += 1
        
        if not node.is_terminal and depth < max_depth:
            node = node.expand()
            if node is None:
                continue
        
        value = simulate(node, max_depth - depth)
        
        backpropagate(node, value)
    
    best_action = None
    best_visit = -1
    action_stats = {}
    for action, child in root.children.items():
        if child.visit_count > 0:
            q_value = child.value_sum / child.visit_count
            action_stats[action] = (q_value, child.visit_count)
        if child.visit_count > best_visit:
            best_visit = child.visit_count
            best_action = action
    
    return best_action, action_stats

def select_best_child(node, exploration_weight):
    best_ucb = -float('inf')
    best_child = None
    for child in node.children.values():
        ucb = child.get_ucb(exploration_weight)
        if ucb > best_ucb:
            best_ucb = ucb
            best_child = child
    return best_child

def simulate(node, max_depth):
    env = copy.deepcopy(node.env)
    depth = 0
    
    while not env.done and depth < max_depth:
        valid_actions = get_valid_actions(env)
        if not valid_actions:
            break
        action = np.random.choice(valid_actions)
        env.step(action)
        depth += 1
    
    win, winner = env._check_win()
    if win:
        if winner == node.env.current_player:
            return 1
        else:
            return -1
    return 0

def backpropagate(node, value):
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value
        node = node.parent

def get_valid_actions(env):
    valid = []
    for i in range(9):
        if env.board[i] == 0:
            current_history = env.history_x if env.current_player == 1 else env.history_o
            if len(current_history) < 3 or i != current_history[-3]:
                valid.append(i)
    return valid

def encode_state(env, current_player, device):
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
    
    return torch.FloatTensor(state).to(device)

def play_vs_ai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN(16, 9).to(device)
    
    try:
        policy_net.load_state_dict(torch.load("Models/moonchess_policy_400000.pth", map_location=device))
        print("已加载训练好的模型!")
    except FileNotFoundError:
        print("警告: 未找到训练好的模型，使用随机初始化的模型!")
    
    policy_net.eval()
    
    env = MoonChessEnv()
    env.reset()
    
    print("\n欢迎来到月亮棋对战AI!")
    print("请选择你要扮演的角色:")
    print("1. X (先手)")
    print("2. O (后手)")
    
    choice = input("请输入 1 或 2: ")
    player_side = 1 if choice == "1" else -1
    
    print(f"\n你是 {'X' if player_side == 1 else 'O'}, AI是 {'O' if player_side == 1 else 'X'}")
    print("AI将使用3回合蒙特卡洛搜索进行决策!")
    
    while not env.done:
        env.render()
        
        if env.current_player == player_side:
            try:
                valid = get_valid_actions(env)
                action = int(input(f"请输入落子位置 (有效位置: {valid}): "))
                if action not in valid:
                    print("无效位置!")
                    continue
                obs, reward, done, truncated, _ = env.step(action)
            except ValueError:
                print("请输入数字!")
                continue
        else:
            print("\nAI正在思考 (使用3回合蒙特卡洛搜索)...")
            action, action_stats = mcts_search(env, num_simulations=500, max_depth=6, exploration_weight=1.0)
            
            valid = get_valid_actions(env)
            print("\n蒙特卡洛搜索结果:")
            for i in range(9):
                if i in valid:
                    if i in action_stats:
                        q_val, visits = action_stats[i]
                        print(f"  位置 {i}: Q值={q_val:.4f}, 访问次数={visits}")
                    else:
                        print(f"  位置 {i}: 未探索")
                else:
                    print(f"  位置 {i}: 无效位置")
            
            print(f"\nAI选择了: {action}")
            obs, reward, done, truncated, _ = env.step(action)
        
        if truncated:
            break
    
    env.render()
    win, winner = env._check_win()
    if win:
        if winner == player_side:
            print("恭喜你获胜了!")
        else:
            print("AI获胜了!")
    else:
        print("游戏结束! 平局!")

if __name__ == "__main__":
    play_vs_ai()
