import torch
import numpy as np
from MoonChess import MoonChessEnv
from DQN2 import DQN

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
            state = encode_state(env, env.current_player, device)
            valid = get_valid_actions(env)
            
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0))
                q_values_np = q_values.cpu().numpy()[0]
                print("\nAI各位置Q值:")
                for i in range(9):
                    validity = "✓" if i in valid else "✗"
                    print(f"  位置 {i}: {q_values_np[i]:.4f} {validity}")
                for i in range(9):
                    if i not in valid:
                        q_values_np[i] = -np.inf
                action = int(np.argmax(q_values_np))
            
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
