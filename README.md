This project utilizes DQN to train an agent for decision-making in the game of Moon Chess.

Moon Chess is a mini-game in Genshin Impact, and the game rules are as follows:
1. The game is played on a 3x3 board, and the user wins if their pieces form a straight line.
2. After a user makes a move, the pieces placed before the 3rd move will disappear.
3. Users cannot place a piece in the disappearing position.
(Note: The original rules of the game allow for moves to be made in the disappearing position, but doing so would inevitably result in a draw. Therefore, a revised rule was later introduced in the Genshin Impact community and the Thousand Star Realm, prohibiting moves in the disappearing position.)

Inference mode:
1. Use DQN for single-step inference.
2. Utilize Monte Carlo Tree Search (MCTS) + DQN for deep reasoning. Currently, the search depth is limited to 3 episodes.
3. You can consider creating a UI page that does not limit the search depth, just like AlphaGo.
    I have provided a simple UI interface created with Unity in the Release directory, including the packaged exe application and the UnityPackage file.

Training algorithm:
1. Construct a DQN model using a 5-layer fully connected network.
    Input features (except for the board state, all other features need to be normalized):
        Current board state: 1 represents own pieces, -1 represents opponent's pieces, and 0 represents empty spaces.
        Own historical move positions: Record the move positions of the last 3 moves. If the number of moves is less than 3, add a "-1".
        Historical move positions of the opponent: Record the move positions of the last 3 moves. If the number of moves is less than 3, add a -1.
        The location that is about to disappear.
2. Only win-loss rewards are provided, and the rewards are sparse.
3. Use a single DQN for self-play training, where the training opponent is an agent from a historical version.
    The model is saved every 2500 rounds. With probability p, it competes against the most recently saved model; with probability 0.05, it competes against a random model; and with probability (0.95-p), it competes against a random historical version of the model.
    The p-value gradually decreased from 0.8 to 0.5.
4. It provides two training modes: single-process and multi-process. The multi-process mode is only supported on Linux systems.
5. A priority experience replay (PER) buffer has been set up.

Training results:
    A trained model with 400,000 steps, available in both pth and onnx formats, can be found in the Models directory.



本项目利用DQN训练一个智能体，用于在月亮棋游戏中进行决策。

月亮棋是原神中的一款小游戏，游戏规则如下：
1，游戏在3*3的棋盘进行，用户的棋子连成一条线就算胜利
2，当用户落子之后，3步之前的棋子会消失
3，用户不能在消失位置落子
（注：游戏原版规则可以在消失位置落子，然而那样必定和棋。因此后来在原神社区和千星奇域中产生了不能在消失位置落子的改版规则）

推理模式：
1，使用DQN进行单步推理
2，使用蒙特卡洛搜索（MCTS）+DQN进行深度推理。目前限制了搜索深度为3回合
3，你可以考虑制作不限制搜索深度的UI页面，就像AlphaGo那样。
    我在Release目录下提供了一个用Unity制作的简单UI界面，包括打包好的exe应用，以及UnityPackage文件。

训练算法：
1，使用5层全连接网络构建DQN模型
    输入特征（除了棋盘状态，其他特征都需要归一化）：
        当前棋盘状态：1表示己方棋子，-1表示对方棋子，0表示空位
        己方历史落子位置：记录3步的落子位置，不满3步的补充-1
        对方历史落子位置：记录3步的落子位置，不满3步的补充-1
        即将消失的位置
2，只提供胜负奖励，奖励是稀疏的
3，使用单DQN进行自博弈训练，训练对手是历史版本的智能体。
    每2500轮保存一次模型，p的概率与最新保存的模型对战，0.05的概率与随机模型对战，（0.95-p）的概率与随机历史版本的模型对战
    p值从0.8逐渐下降到0.5
4，提供了单进程和多进程两种训练模式，多进程模式仅支持linux系统
5，设置了优先级经验回放（PER）缓冲区

训练结果：
    提供了400000步的训练模型，包括pth和onnx两个版本，可以在Models目录找到。