try:
    import gym
    from gym import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("警告: gym 库未安装，控制台模式仍可正常使用")

import numpy as np


if HAS_GYM:
    class BaseMoonChessEnv(gym.Env):
        pass
else:
    class BaseMoonChessEnv:
        pass


class MoonChessEnv(BaseMoonChessEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        if HAS_GYM:
            super().__init__()
        
        self.board = np.zeros(9, dtype=int)
        self.history_x = []
        self.history_o = []
        self.current_player = 1
        self.action_space = spaces.Discrete(9) if HAS_GYM else None
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=int) if HAS_GYM else None
        self.done = False
        self.step_count = 0
        self.max_steps = 1000
        # 新增：记录对手的奖励（用于训练时给失败方补惩罚）
        self.opponent_reward = 0  

    def reset(self, seed=None, options=None):
        if HAS_GYM:
            super().reset(seed=seed)
        
        self.board = np.zeros(9, dtype=int)
        self.history_x = []
        self.history_o = []
        self.current_player = 1
        self.done = False
        self.step_count = 0
        self.opponent_reward = 0  # 重置对手奖励
        return self.board.copy(), {}

    def _check_win(self):
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for pattern in win_patterns:
            if (self.board[pattern[0]] == self.board[pattern[1]] == self.board[pattern[2]] != 0):
                return True, self.board[pattern[0]]
        return False, 0

    def step(self, action):
        # 重置对手奖励
        self.opponent_reward = 0  
        
        #1，检查游戏是否结束
        if self.done:
            return self.board.copy(), 0, True, False, {}

        #2，检查是否在重复位置落子
        if self.board[action] != 0:
            return self.board.copy(), -1, self.done, False, {}

        #3，查看当前玩家历史记录。额外指示即将消失的位置不能落子（暂时无法触发，重复落子已经在步骤2中判定）
        current_history = self.history_x if self.current_player == 1 else self.history_o
        if len(current_history) >= 3:
            disappear_pos = current_history[-3]
            if action == disappear_pos:
                return self.board.copy(), -1, self.done, False, {}

        #4，落子，并记录玩家历史操作
        self.board[action] = self.current_player
        current_history.append(action)

        #5，棋子消失
        if len(current_history) >= 4:
            disappear_pos = current_history[-4]
            self.board[disappear_pos] = 0

        #6，检查是否获胜
        win, winner = self._check_win()
        if win:
            self.done = True
            # 当前玩家奖励
            reward = 10 if winner == self.current_player else -10
            # 关键：给对手设置负奖励（当前玩家赢=对手输）
            self.opponent_reward = -reward  
            return self.board.copy(), reward, True, False, {}

        #7，增加步数计数，检查是否超过步数上限
        self.step_count += 1
        if self.step_count >= self.max_steps:
            return self.board.copy(), 0, False, True, {}

        #8，切换玩家
        self.current_player = -self.current_player
        return self.board.copy(), 0, False, False, {}

    def render(self, mode='human'):
        print("\n")
        for i in range(3):
            row = []
            for j in range(3):
                idx = i * 3 + j
                if self.board[idx] == 1:
                    row.append('X')
                elif self.board[idx] == -1:
                    row.append('O')
                else:
                    row.append(str(idx))
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("---+---+---")
        print(f"\n当前玩家: {'X' if self.current_player == 1 else 'O'}")
        print(f"当前步数: {self.step_count}/{self.max_steps}")
        if len(self.history_x) > 0:
            print(f"X 的历史落子: {self.history_x}")
        if len(self.history_o) > 0:
            print(f"O 的历史落子: {self.history_o}")
        current_history = self.history_x if self.current_player == 1 else self.history_o
        if len(current_history) >= 3:
            print(f"当前玩家下一个将消失的位置: {current_history[-3]}")
        print("\n")

    def close(self):
        pass


def play_human():
    env = MoonChessEnv()
    env.reset()
    print("欢迎来到月亮棋！")
    print("规则：")
    print("- 两个玩家交替落子（X和O）")
    print("- 当落下第4个棋子后，第1个棋子会消失")
    print("- 之后每一步，3步之前的棋子会消失")
    print("- 棋子不能下在即将消失的位置")
    print("- 先形成三连线者获胜")
    print("- 最多1000步，超过则游戏结束")
    print("\n操作：输入0-8之间的数字来落子\n")

    while not env.done:
        env.render()
        try:
            action = int(input("请输入落子位置 (0-8): "))
            if action < 0 or action > 8:
                print("请输入0-8之间的数字！")
                continue
            obs, reward, done, truncated, _ = env.step(action)
            if reward == -1 and not done and not truncated:  # 修正：原reward=-10是获胜惩罚，但这里是无效落子
                print("无效的位置！请重新选择。")
            if truncated:
                print("游戏结束！达到步数上限！")
                break
        except ValueError:
            print("请输入有效的数字！")

    env.render()
    win, winner = env._check_win()
    if win:
        print(f"游戏结束！{'X' if winner == 1 else 'O'} 获胜！")
    else:
        print("游戏结束！平局！")


if __name__ == "__main__":
    play_human()