"""
蒙特卡洛树搜索 (MCTS) - 用于评估局面胜率
"""
import numpy as np
import torch
from config import BOARD_SIZE, BLACK, WHITE


class MCTSNode:
    """MCTS 树节点"""
    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action  # 到达此节点的动作
        self.prior = prior    # 先验概率
        self.children = {}    # 子节点字典 {action: node}
        
        self.visit_count = 0
        self.value_sum = 0.0  # 累计价值
        self.is_expanded = False
    
    @property
    def value(self):
        """平均价值（胜率）"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def q_value(self):
        """Q值（用于UCT选择）"""
        if self.visit_count == 0:
            return float('inf')  # 未访问的节点优先探索
        return self.value
    
    def select_child(self, c_puct=1.0):
        """使用 UCT 选择最佳子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # UCT 公式: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
            uct_score = child.q_value + c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, actions, priors):
        """扩展节点"""
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, action=action, prior=prior)
        self.is_expanded = True
    
    def update(self, value):
        """更新节点价值"""
        self.visit_count += 1
        self.value_sum += value
    
    def backup(self, value):
        """反向传播价值"""
        self.update(value)
        if self.parent is not None:
            # 价值取反（因为是对手视角）
            self.parent.backup(-value)


class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self, agent, env, num_simulations=800, c_puct=1.0):
        self.agent = agent
        self.env = env
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def get_action_probs(self, board, current_player, temperature=1.0):
        """
        执行 MCTS 搜索，返回动作概率分布
        
        返回: (action_probs, root_value)
        action_probs: [board_size * board_size] 每个动作的概率
        root_value: 当前局面的评估胜率
        """
        root = MCTSNode()
        
        # 使用神经网络评估根节点
        state_input = self.agent.board_to_input(board, current_player)
        with torch.no_grad():
            log_policy, value = self.agent.model(
                torch.FloatTensor(state_input).unsqueeze(0).to(self.agent.device)
            )
            policy = torch.exp(log_policy).cpu().numpy()[0]
            root_value = value.cpu().numpy()[0][0]
        
        # 扩展根节点
        legal_moves = self._get_legal_moves(board)
        if len(legal_moves) == 0:
            return np.zeros(BOARD_SIZE * BOARD_SIZE), root_value
        
        priors = np.array([policy[a] for a in legal_moves])
        priors = priors / priors.sum() if priors.sum() > 0 else np.ones(len(legal_moves)) / len(legal_moves)
        root.expand(legal_moves, priors)
        
        # 执行多次模拟
        for _ in range(self.num_simulations):
            node = root
            env_copy = self._copy_env(board, current_player)
            
            # Selection: 选择路径到叶子节点
            while node.is_expanded and len(node.children) > 0:
                action, node = node.select_child(self.c_puct)
                row, col = action // BOARD_SIZE, action % BOARD_SIZE
                env_copy.step((row, col))
            
            # Evaluation: 评估叶子节点
            if env_copy.done:
                # 游戏结束，使用实际结果
                if env_copy.winner == current_player:
                    value = 1.0
                elif env_copy.winner is None:
                    value = 0.0
                else:
                    value = -1.0
            else:
                # 使用神经网络评估
                state_input = self.agent.board_to_input(env_copy.board, env_copy.current_player)
                with torch.no_grad():
                    _, value_tensor = self.agent.model(
                        torch.FloatTensor(state_input).unsqueeze(0).to(self.agent.device)
                    )
                    value = value_tensor.cpu().numpy()[0][0]
                    # 如果是对手回合，价值取反
                    if env_copy.current_player != current_player:
                        value = -value
            
            # Expansion: 扩展叶子节点
            if not env_copy.done and not node.is_expanded:
                legal_moves = self._get_legal_moves(env_copy.board)
                if len(legal_moves) > 0:
                    state_input = self.agent.board_to_input(env_copy.board, env_copy.current_player)
                    with torch.no_grad():
                        log_policy, _ = self.agent.model(
                            torch.FloatTensor(state_input).unsqueeze(0).to(self.agent.device)
                        )
                        policy = torch.exp(log_policy).cpu().numpy()[0]
                    
                    priors = np.array([policy[a] for a in legal_moves])
                    priors = priors / priors.sum() if priors.sum() > 0 else np.ones(len(legal_moves)) / len(legal_moves)
                    node.expand(legal_moves, priors)
            
            # Backup: 反向传播
            node.backup(value)
        
        # 计算动作概率（基于访问次数）
        action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        # 应用温度参数
        if temperature == 0:
            # 选择访问次数最多的动作
            best_action = np.argmax(action_probs)
            action_probs = np.zeros_like(action_probs)
            action_probs[best_action] = 1.0
        else:
            action_probs = action_probs ** (1.0 / temperature)
            action_probs = action_probs / action_probs.sum() if action_probs.sum() > 0 else action_probs
        
        return action_probs, root.value
    
    def _get_legal_moves(self, board):
        """获取合法移动"""
        moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == 0:
                    moves.append(i * BOARD_SIZE + j)
        return moves
    
    def _copy_env(self, board, current_player):
        """复制环境状态"""
        from gobang_env import GobangEnv
        env_copy = GobangEnv()
        env_copy.board = board.copy()
        env_copy.current_player = current_player
        env_copy.done = False
        env_copy.winner = None
        env_copy.move_history = []
        return env_copy
