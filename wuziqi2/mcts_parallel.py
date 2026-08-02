"""
并行 MCTS - 同时运行多个游戏对局，提高 GPU 利用率
"""
import numpy as np
import torch
from config import BOARD_SIZE, BLACK, WHITE
from mcts import MCTSNode


class ParallelMCTS:
    """并行 MCTS - 同时处理多个游戏状态"""
    
    def __init__(self, agent, env_class, num_simulations=50, c_puct=1.0, batch_size=8):
        self.agent = agent
        self.env_class = env_class
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size  # 并行游戏数量
    
    def get_action_probs_batch(self, boards, current_players, temperature=1.0):
        """
        批量执行 MCTS，同时处理多个游戏状态
        
        boards: list of boards，每个是一个游戏状态
        current_players: list of players
        
        返回: list of (action_probs, value)
        """
        n_games = len(boards)
        results = []
        
        # 为每个游戏创建根节点
        roots = [MCTSNode() for _ in range(n_games)]
        
        # 批量评估根节点
        state_inputs = []
        for board, player in zip(boards, current_players):
            state_input = self.agent.board_to_input(board, player)
            state_inputs.append(state_input)
        
        state_batch = torch.FloatTensor(np.array(state_inputs)).to(self.agent.device)
        with torch.no_grad():
            log_policies, values = self.agent.model(state_batch)
            policies = torch.exp(log_policies).cpu().numpy()
            root_values = values.cpu().numpy()
        
        # 扩展根节点
        for i, (root, board, player) in enumerate(zip(roots, boards, current_players)):
            legal_moves = self._get_legal_moves(board)
            if len(legal_moves) == 0:
                continue
            
            priors = np.array([policies[i][a] for a in legal_moves])
            priors = priors / priors.sum() if priors.sum() > 0 else np.ones(len(legal_moves)) / len(legal_moves)
            root.expand(legal_moves, priors)
            results.append((None, root_values[i][0]))  # 临时结果
        
        # 执行多次模拟（批量）
        for sim in range(self.num_simulations):
            # 收集所有需要评估的叶子节点
            batch_nodes = []
            batch_envs = []
            batch_players = []
            
            for game_idx, (root, board, player) in enumerate(zip(roots, boards, current_players)):
                if not root.children:
                    continue
                
                node = root
                env = self._copy_env(board, player)
                
                # Selection
                while node.is_expanded and len(node.children) > 0:
                    action, node = node.select_child(self.c_puct)
                    env.step((action // BOARD_SIZE, action % BOARD_SIZE))
                
                batch_nodes.append((game_idx, node))
                batch_envs.append(env)
                batch_players.append(env.current_player)
            
            if len(batch_envs) == 0:
                break
            
            # 批量评估
            state_inputs = []
            for env in batch_envs:
                state_input = self.agent.board_to_input(env.board, env.current_player)
                state_inputs.append(state_input)
            
            state_batch = torch.FloatTensor(np.array(state_inputs)).to(self.agent.device)
            with torch.no_grad():
                log_policies, values = self.agent.model(state_batch)
                policies = torch.exp(log_policies).cpu().numpy()
                leaf_values = values.cpu().numpy()
            
            # 扩展和反向传播
            for (game_idx, node), env, policy, value in zip(batch_nodes, batch_envs, policies, leaf_values):
                if not env.done and not node.is_expanded:
                    legal_moves = self._get_legal_moves(env.board)
                    if len(legal_moves) > 0:
                        priors = np.array([policy[a] for a in legal_moves])
                        priors = priors / priors.sum() if priors.sum() > 0 else np.ones(len(legal_moves)) / len(legal_moves)
                        node.expand(legal_moves, priors)
                
                # 确定价值
                if env.done:
                    if env.winner == current_players[game_idx]:
                        actual_value = 1.0
                    elif env.winner is None:
                        actual_value = 0.0
                    else:
                        actual_value = -1.0
                else:
                    actual_value = value[0]
                    if env.current_player != current_players[game_idx]:
                        actual_value = -actual_value
                
                node.backup(actual_value)
        
        # 计算最终动作概率
        final_results = []
        for root, player in zip(roots, current_players):
            if not root.children:
                final_results.append((np.zeros(BOARD_SIZE * BOARD_SIZE), 0.0))
                continue
            
            action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
            for action, child in root.children.items():
                action_probs[action] = child.visit_count
            
            if temperature == 0:
                best_action = np.argmax(action_probs)
                action_probs = np.zeros_like(action_probs)
                action_probs[best_action] = 1.0
            else:
                action_probs = action_probs ** (1.0 / temperature)
                action_probs = action_probs / action_probs.sum() if action_probs.sum() > 0 else action_probs
            
            final_results.append((action_probs, root.value))
        
        return final_results
    
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
        env_copy = self.env_class()
        env_copy.board = board.copy()
        env_copy.current_player = current_player
        env_copy.done = False
        env_copy.winner = None
        env_copy.move_history = []
        return env_copy
