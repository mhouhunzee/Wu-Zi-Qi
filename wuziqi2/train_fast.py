"""
五子棋AI快速训练脚本 - 不记录详细日志
"""
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm

from config import (T, CYCLE, SEGMENT_SIZE, MODEL_DIR, LOG_DIR, BOARD_SIZE, BLACK, WHITE, K, REWARD, E, USE_RULES, LOG_BUFFER_SIZE, TRAIN_FREQUENCY,ADAPTIVE_TRAINING, WIN_RATE_THRESHOLD, TRAINING_BOOST_FACTOR, MAX_BOOST_FACTOR, CONSECUTIVE_LOW_WIN_ROUNDS,ENABLE_TRAINING_PROTECTION, WIN_RATE_DROP_THRESHOLD,CONSECUTIVE_DROPS_BEFORE_ROLLBACK, LEARNING_RATE_ADJUSTMENT,MODEL_KEEP_LAST_N, LEARNING_RATE, USE_MCTS, MCTS_NUM_SIMULATIONS, MCTS_C_PUCT, MCTS_TEMPERATURE, PARALLEL_BATCH_SIZE, USE_PARALLEL_MCTS)
from gobang_env import GobangEnv
from model import GobangAgent
from mcts_parallel import ParallelMCTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class FastTrainer:
    """快速训练管理器（最小化日志）"""
    
    def __init__(self):
        self.env = GobangEnv()
        self.black_agent = GobangAgent(BLACK)
        self.white_agent = GobangAgent(WHITE)
        
        # 获取脚本所在目录的绝对路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.script_dir, MODEL_DIR)
        self.log_dir = os.path.join(self.script_dir, LOG_DIR)
        
        # 只创建模型目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"[信息] 模型保存路径: {self.model_dir}")
        print(f"[信息] 日志保存路径: {self.log_dir}")
        
        # 最小化统计
        self.stats = {
            'cycle': [],
            'game_results': [],
            'training_phase': [],
            'losses': []
        }
        
        # 游戏日志文件路径
        self.game_log_file = os.path.join(self.log_dir, "game_logs.csv")
        # 初始化日志文件（如果不存在）
        if not os.path.exists(self.game_log_file):
            with open(self.game_log_file, 'w', encoding='utf-8') as f:
                f.write("game_id,winner,gamelength,moves\n")
        
        # 全局游戏计数器
        self.global_game_count = 0
        
        # 日志缓冲区
        self.log_buffer = []
        self.log_buffer_size = LOG_BUFFER_SIZE  # 从配置读取
        
        # 是否使用规则层
        self.use_rules = USE_RULES  # 从配置读取
        
        # 自适应训练状态跟踪
        self.adaptive_training = ADAPTIVE_TRAINING
        self.black_low_win_count = 0  # Black连续低胜率轮数
        self.white_low_win_count = 0  # White连续低胜率轮数
        self.black_boost_factor = 1.0  # Black当前训练倍增因子
        self.white_boost_factor = 1.0  # White当前训练倍增因子
        
        # 训练保护机制状态跟踪
        self.training_protection = ENABLE_TRAINING_PROTECTION
        self.black_consecutive_drops = 0  # Black连续胜率下降轮数
        self.white_consecutive_drops = 0  # White连续胜率下降轮数
        self.black_prev_win_rate = None   # Black上一轮胜率
        self.white_prev_win_rate = None   # White上一轮胜率
        self.black_rollback_count = 0     # Black回退次数
        self.white_rollback_count = 0     # White回退次数
        
        # 加载已有模型（如果存在）
        self._load_existing_models()
    
    def _find_latest_model(self, color):
        """查找指定颜色最新的模型文件"""
        import glob
        pattern = os.path.join(self.model_dir, f"{color}_cycle_*.pth")
        models = glob.glob(pattern)
        if not models:
            return None
        models.sort()
        return models[-1]
    
    def _load_existing_models(self):
        """加载已有的最新模型（如果存在）"""
        black_latest = self._find_latest_model('black')
        white_latest = self._find_latest_model('white')
        
        if black_latest and os.path.exists(black_latest):
            try:
                self.black_agent.load(black_latest)
                print(f"[加载] Black模型: {black_latest}")
            except Exception as e:
                print(f"[错误] 加载Black模型失败: {e}")
        
        if white_latest and os.path.exists(white_latest):
            try:
                self.white_agent.load(white_latest)
                print(f"[加载] White模型: {white_latest}")
            except Exception as e:
                print(f"[错误] 加载White模型失败: {e}")
    
    def _flush_log_buffer(self):
        """将缓冲区中的日志批量写入文件"""
        if self.log_buffer:
            with open(self.game_log_file, 'a', encoding='utf-8') as f:
                f.writelines(self.log_buffer)
            self.log_buffer.clear()
    
    def _log_game(self, cycle_num, game_num, winner, move_history):
        """
        记录单局游戏日志
        格式: game{cycle}_{game_num},winner{winner},gamelength{length},B(9,9),W(9,10),...
        """
        self.global_game_count += 1
        
        # 构建游戏ID
        game_id = f"game{cycle_num}_{game_num}"
        
        # 确定胜者标识
        if winner == BLACK:
            winner_str = "B"
        elif winner == WHITE:
            winner_str = "W"
        else:
            winner_str = "D"  # 平局
        
        # 对局长度
        game_length = len(move_history)
        
        # 构建落子记录
        moves = []
        for player, row, col in move_history:
            color = "B" if player == BLACK else "W"
            moves.append(f"{color}({row},{col})")
        moves_str = ",".join(moves)
        
        # 写入日志（使用缓冲区）
        log_line = f"{game_id},winner{winner_str},gamelength{game_length},{moves_str}\n"
        self.log_buffer.append(log_line)
        
        # 缓冲区满时批量写入
        if len(self.log_buffer) >= self.log_buffer_size:
            self._flush_log_buffer()
    
    def play_games_with_mcts_batch(self, black_agent, white_agent, mcts, num_games):
        """
        批量并行运行多个MCTS游戏
        
        返回: list of (winner, black_data, white_data)
        """
        results = []
        
        # 初始化多个游戏环境
        envs = [GobangEnv() for _ in range(num_games)]
        black_data_list = [[] for _ in range(num_games)]
        white_data_list = [[] for _ in range(num_games)]
        
        # 所有游戏是否结束
        active_games = list(range(num_games))
        
        while active_games:
            # 收集所有活跃游戏的状态
            boards = []
            players = []
            game_indices = []
            
            for idx in active_games:
                env = envs[idx]
                if not env.done:
                    boards.append(env.get_state())
                    players.append(env.current_player)
                    game_indices.append(idx)
            
            if not boards:
                break
            
            # 批量MCTS搜索
            mcts_results = mcts.get_action_probs_batch(boards, players, temperature=1.0)
            
            # 处理每个游戏的结果
            for i, game_idx in enumerate(game_indices):
                env = envs[game_idx]
                action_probs, value = mcts_results[i]
                
                # 记录数据
                if players[i] == BLACK:
                    state_input = black_agent.board_to_input(boards[i], BLACK)
                    black_data_list[game_idx].append((state_input, action_probs, value))
                else:
                    state_input = white_agent.board_to_input(boards[i], WHITE)
                    white_data_list[game_idx].append((state_input, action_probs, value))
                
                # 选择动作
                legal_moves = env.get_valid_moves()
                if len(legal_moves) == 0:
                    continue
                
                legal_action_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
                legal_probs = np.array([action_probs[idx] for idx in legal_action_indices])
                if legal_probs.sum() > 0:
                    legal_probs = legal_probs / legal_probs.sum()
                    action_idx = np.random.choice(len(legal_moves), p=legal_probs)
                    action = legal_moves[action_idx]
                else:
                    action = random.choice(legal_moves)
                
                # 执行动作
                env.step(action)
            
            # 移除已结束的游戏
            active_games = [idx for idx in active_games if not envs[idx].done]
        
        # 收集结果
        for i in range(num_games):
            winner = envs[i].get_game_result()
            
            black_states = [d[0] for d in black_data_list[i]]
            black_policies = [d[1] for d in black_data_list[i]]
            black_values = [d[2] for d in black_data_list[i]]
            
            white_states = [d[0] for d in white_data_list[i]]
            white_policies = [d[1] for d in white_data_list[i]]
            white_values = [d[2] for d in white_data_list[i]]
            
            results.append((winner, 
                          (black_states, black_policies, black_values),
                          (white_states, white_policies, white_values)))
        
        return results
    
    def play_game_with_mcts(self, black_agent, white_agent, num_simulations=50):
        """
        使用 MCTS 进行一局对局，生成训练数据
        训练目标：让模型学会评估局面胜率，并选择胜率最高的动作
        优化：减少模拟次数以提高速度
        """
        state = self.env.reset()
        done = False
        
        # 存储训练数据
        black_data = []  # [(state, policy, value), ...]
        white_data = []
        
        # 创建并行 MCTS 实例（减少模拟次数以提高速度）
        black_mcts = ParallelMCTS(black_agent, GobangEnv, num_simulations=num_simulations, batch_size=1)
        white_mcts = ParallelMCTS(white_agent, GobangEnv, num_simulations=num_simulations, batch_size=1)
        
        while not done:
            current_player = self.env.current_player
            board = self.env.get_state()
            
            # 使用 MCTS 搜索，获取动作概率和局面胜率
            if current_player == BLACK:
                results = black_mcts.get_action_probs_batch([board], [BLACK], temperature=1.0)
                action_probs, value = results[0]
                state_input = black_agent.board_to_input(board, BLACK)
                black_data.append((state_input, action_probs, value))
            else:
                results = white_mcts.get_action_probs_batch([board], [WHITE], temperature=1.0)
                action_probs, value = results[0]
                state_input = white_agent.board_to_input(board, WHITE)
                white_data.append((state_input, action_probs, value))
            
            # 根据概率选择动作（训练时加入探索）
            legal_moves = self.env.get_valid_moves()
            if len(legal_moves) == 0:
                break
            
            # 将 (row, col) 转换为 action index
            legal_action_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
            legal_probs = np.array([action_probs[idx] for idx in legal_action_indices])
            if legal_probs.sum() > 0:
                legal_probs = legal_probs / legal_probs.sum()
                action_idx = np.random.choice(len(legal_moves), p=legal_probs)
                action = legal_moves[action_idx]
            else:
                action = random.choice(legal_moves)
            
            # 执行动作
            state, reward, done, info = self.env.step(action)
        
        # 确定胜者
        winner = self.env.get_game_result()
        
        # 处理训练数据：分离 states, policies, values
        black_states = [d[0] for d in black_data]
        black_policies = [d[1] for d in black_data]
        black_values = [d[2] for d in black_data]
        
        white_states = [d[0] for d in white_data]
        white_policies = [d[1] for d in white_data]
        white_values = [d[2] for d in white_data]
        
        return winner, \
               (black_states, black_policies, black_values), \
               (white_states, white_policies, white_values)
    
    def _create_policy_targets(self, states, move_history, player_color):
        """为指定玩家的每一步创建策略目标"""
        policies = []
        move_idx = 0
        
        for i, (p, row, col) in enumerate(move_history):
            if p == player_color and move_idx < len(states):
                policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
                policy[row * BOARD_SIZE + col] = 1.0
                policies.append(policy)
                move_idx += 1
        
        return policies
    
    def train_agent(self, agent, training_data, num_epochs=5):
        """训练指定代理"""
        states, policies, final_value = training_data
        
        if len(states) == 0:
            return None
        
        # 为每一步分配价值
        values = []
        for i in range(len(states)):
            discount = 0.95 ** (len(states) - i - 1)
            values.append(final_value * discount)
        
        states = np.array(states)
        policies = np.array(policies)
        values = np.array(values).reshape(-1, 1)
        
        # 训练多个epoch
        total_loss = {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
        for epoch in range(num_epochs):
            batch_size = min(32, len(states))
            indices = np.random.choice(len(states), batch_size, replace=False)
            
            batch_states = states[indices]
            batch_policies = policies[indices]
            batch_values = values[indices]
            
            loss = agent.train_step(batch_states, batch_policies, batch_values)
            for key in total_loss:
                total_loss[key] += loss[key]
        
        # 平均损失
        for key in total_loss:
            total_loss[key] /= num_epochs
        
        return total_loss
    
    def calculate_segment_win_rates(self, results, segment_size=SEGMENT_SIZE):
        """计算分段胜率"""
        segments = []
        for i in range(0, len(results), segment_size):
            segment = results[i:i+segment_size]
            black_wins = segment.count(BLACK)
            white_wins = segment.count(WHITE)
            draws = segment.count(None)
            
            total = len(segment)
            segments.append({
                'black_win_rate': black_wins / total,
                'white_win_rate': white_wins / total,
                'draw_rate': draws / total,
                'start_game': i,
                'end_game': min(i + segment_size, len(results))
            })
        
        return segments
    
    def _calculate_win_rate(self, results, agent_color):
        """计算指定agent的胜率"""
        if not results:
            return 0.5
        wins = results.count(agent_color)
        return wins / len(results)
    
    def _aggregate_training_data_mcts(self, game_data_list):
        """聚合MCTS生成的训练数据"""
        all_states = []
        all_policies = []
        all_values = []
        
        for states, policies, values in game_data_list:
            if len(states) > 0:
                all_states.extend(states)
                all_policies.extend(policies)
                all_values.extend(values)
        
        return (np.array(all_states), np.array(all_policies), np.array(all_values).reshape(-1, 1))
    
    def _update_adaptive_training(self, white_results, black_results):
        """
        更新自适应训练状态
        返回: (white_games, black_games) 调整后的训练局数
        """
        if not self.adaptive_training:
            return T, T
        
        white_win_rate = self._calculate_win_rate(white_results, WHITE)
        black_win_rate = self._calculate_win_rate(black_results, BLACK)
        
        # 检查White胜率
        if white_win_rate < WIN_RATE_THRESHOLD:
            self.white_low_win_count += 1
        else:
            self.white_low_win_count = 0
            # 逐渐恢复训练量
            self.white_boost_factor = max(1.0, self.white_boost_factor * 0.9)
        
        # 检查Black胜率
        if black_win_rate < WIN_RATE_THRESHOLD:
            self.black_low_win_count += 1
        else:
            self.black_low_win_count = 0
            # 逐渐恢复训练量
            self.black_boost_factor = max(1.0, self.black_boost_factor * 0.9)
        
        # 应用增强训练
        white_games = T
        black_games = T
        
        if self.white_low_win_count >= CONSECUTIVE_LOW_WIN_ROUNDS:
            # White需要增强训练
            self.white_boost_factor = min(MAX_BOOST_FACTOR, 
                                         self.white_boost_factor * TRAINING_BOOST_FACTOR)
            white_games = int(T * self.white_boost_factor)
            print(f"  [自适应] White胜率{white_win_rate:.2%}过低，增强训练至{white_games}局")
        
        if self.black_low_win_count >= CONSECUTIVE_LOW_WIN_ROUNDS:
            # Black需要增强训练
            self.black_boost_factor = min(MAX_BOOST_FACTOR, 
                                         self.black_boost_factor * TRAINING_BOOST_FACTOR)
            black_games = int(T * self.black_boost_factor)
            print(f"  [自适应] Black胜率{black_win_rate:.2%}过低，增强训练至{black_games}局")
        
        return white_games, black_games
    
    def _check_training_protection(self, white_results, black_results):
        """
        检查是否需要触发训练保护（回退模型）
        返回: (need_rollback_white, need_rollback_black, adjusted_lr)
        """
        if not self.training_protection:
            return False, False, LEARNING_RATE
        
        # 计算当前胜率
        white_win_rate = self._calculate_win_rate(white_results, WHITE)
        black_win_rate = self._calculate_win_rate(black_results, BLACK)
        
        need_rollback_white = False
        need_rollback_black = False
        adjusted_lr = LEARNING_RATE
        
        # 检查White胜率是否下降
        if self.white_prev_win_rate is not None:
            white_drop = self.white_prev_win_rate - white_win_rate
            if white_drop > WIN_RATE_DROP_THRESHOLD:
                self.white_consecutive_drops += 1
                print(f"  [保护] White胜率下降{white_drop:.2%} ({self.white_prev_win_rate:.2%} -> {white_win_rate:.2%})")
            else:
                self.white_consecutive_drops = 0
        
        # 检查Black胜率是否下降
        if self.black_prev_win_rate is not None:
            black_drop = self.black_prev_win_rate - black_win_rate
            if black_drop > WIN_RATE_DROP_THRESHOLD:
                self.black_consecutive_drops += 1
                print(f"  [保护] Black胜率下降{black_drop:.2%} ({self.black_prev_win_rate:.2%} -> {black_win_rate:.2%})")
            else:
                self.black_consecutive_drops = 0
        
        # 触发回退
        if self.white_consecutive_drops >= CONSECUTIVE_DROPS_BEFORE_ROLLBACK:
            need_rollback_white = True
            self.white_rollback_count += 1
            adjusted_lr = LEARNING_RATE * (LEARNING_RATE_ADJUSTMENT ** self.white_rollback_count)
            print(f"  [保护] 触发White模型回退！学习率调整为{adjusted_lr:.6f}")
        
        if self.black_consecutive_drops >= CONSECUTIVE_DROPS_BEFORE_ROLLBACK:
            need_rollback_black = True
            self.black_rollback_count += 1
            adjusted_lr = LEARNING_RATE * (LEARNING_RATE_ADJUSTMENT ** self.black_rollback_count)
            print(f"  [保护] 触发Black模型回退！学习率调整为{adjusted_lr:.6f}")
        
        # 更新上一轮胜率
        self.white_prev_win_rate = white_win_rate
        self.black_prev_win_rate = black_win_rate
        
        return need_rollback_white, need_rollback_black, adjusted_lr
    
    def _rollback_model(self, agent, color_name, cycle_num):
        """回退到最近的有效模型"""
        if cycle_num <= 0:
            return False
        
        import glob
        import re
        
        # 查找所有可用的历史模型
        pattern = os.path.join(self.model_dir, f"{color_name}_cycle_*.pth")
        available_models = glob.glob(pattern)
        
        # 提取cycle编号
        models_with_cycles = []
        for model_path in available_models:
            match = re.search(rf"{color_name}_cycle_(\d+)\.pth", os.path.basename(model_path))
            if match:
                cycle_num_in_file = int(match.group(1))
                if cycle_num_in_file < cycle_num:  # 只考虑当前轮次之前的模型
                    models_with_cycles.append((cycle_num_in_file, model_path))
        
        if not models_with_cycles:
            print(f"  [保护] 找不到{color_name}可用的历史模型，无法回退")
            return False
        
        # 选择最近的一个模型
        models_with_cycles.sort(key=lambda x: x[0], reverse=True)
        target_cycle, model_path = models_with_cycles[0]
        
        try:
            agent.load(model_path)
            print(f"  [保护] {color_name}模型已回退到Cycle {target_cycle}")
            return True
        except Exception as e:
            print(f"  [保护] 回退{color_name}模型失败: {e}")
            return False
    
    def _adjust_learning_rate(self, agent, new_lr):
        """调整agent的学习率"""
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"  [保护] 学习率已调整为{new_lr:.6f}")
    
    def train_cycle(self, cycle_num, t=T):
        """训练一个cycle（支持自适应训练）"""
        print(f"\n{'='*50}")
        print(f"Cycle {cycle_num + 1}/{CYCLE}")
        print(f"{'='*50}\n")
        
        # 根据上一轮结果调整训练量
        white_games = t
        black_games = t
        if cycle_num > 0 and self.adaptive_training:
            # 获取上一轮的结果
            prev_cycle = self.stats['cycle'][-1]
            white_results = prev_cycle['white_training']['results']
            black_results = prev_cycle['black_training']['results']
            white_games, black_games = self._update_adaptive_training(white_results, black_results)
        
        cycle_stats = {
            'cycle': cycle_num,
            'white_training': {'results': []},
            'black_training': {'results': []}
        }
        
        # 阶段1: 固定Black，训练White
        if USE_MCTS and USE_PARALLEL_MCTS:
            print(f"训练 White: {white_games} 局 (使用MCTS评估胜率，并行batch={PARALLEL_BATCH_SIZE})")
            white_training_data = []
            
            # 创建并行MCTS
            white_mcts = ParallelMCTS(self.white_agent, GobangEnv, num_simulations=MCTS_NUM_SIMULATIONS, batch_size=PARALLEL_BATCH_SIZE)
            
            # 批量处理游戏
            for batch_start in tqdm(range(0, white_games, PARALLEL_BATCH_SIZE), desc="White batches"):
                batch_end = min(batch_start + PARALLEL_BATCH_SIZE, white_games)
                actual_batch_size = batch_end - batch_start
                
                # 并行运行多个游戏
                batch_results = self.play_games_with_mcts_batch(
                    self.black_agent, self.white_agent, white_mcts, actual_batch_size
                )
                
                # 处理结果
                for game_idx, (winner, black_data, white_data) in enumerate(batch_results):
                    global_game = batch_start + game_idx
                    self._log_game(cycle_num, global_game, winner, self.env.move_history)
                    cycle_stats['white_training']['results'].append(winner)
                    white_training_data.append(white_data)
                
                # 定期训练
                if len(white_training_data) >= TRAIN_FREQUENCY:
                    recent_data = self._aggregate_training_data_mcts(white_training_data[-TRAIN_FREQUENCY:])
                    loss = self.train_agent(self.white_agent, recent_data)
                    if loss:
                        self.stats['losses'].append(loss)
        elif USE_MCTS:
            # 使用MCTS但不并行
            print(f"训练 White: {white_games} 局 (使用MCTS评估胜率，串行)")
            white_training_data = []
            
            for game in tqdm(range(white_games), desc="White"):
                winner, black_data, white_data = \
                    self.play_game_with_mcts(self.black_agent, self.white_agent, num_simulations=MCTS_NUM_SIMULATIONS)
                
                self._log_game(cycle_num, game, winner, self.env.move_history)
                cycle_stats['white_training']['results'].append(winner)
                white_training_data.append(white_data)
                
                if (game + 1) % TRAIN_FREQUENCY == 0 and len(white_training_data) > 0:
                    recent_data = self._aggregate_training_data_mcts(white_training_data[-TRAIN_FREQUENCY:])
                    loss = self.train_agent(self.white_agent, recent_data)
                    if loss:
                        self.stats['losses'].append(loss)
        else:
            # 不使用MCTS，普通训练
            print(f"训练 White: {white_games} 局 (普通自对弈)")
            white_training_data = []
            
            for game in tqdm(range(white_games), desc="White"):
                epsilon = max(0.01, 1.0 - game / white_games * 0.9)
                winner, black_data, white_data = \
                    self.play_game(self.black_agent, self.white_agent, epsilon, use_rules=self.use_rules)
                
                self._log_game(cycle_num, game, winner, self.env.move_history)
                cycle_stats['white_training']['results'].append(winner)
                white_training_data.append(white_data)
                
                if (game + 1) % TRAIN_FREQUENCY == 0 and len(white_training_data) > 0:
                    recent_data = self._aggregate_training_data(white_training_data[-TRAIN_FREQUENCY:])
                    loss = self.train_agent(self.white_agent, recent_data)
                    if loss:
                        self.stats['losses'].append(loss)
        
        cycle_stats['white_training']['segments'] = \
            self.calculate_segment_win_rates(cycle_stats['white_training']['results'])
        
        # 阶段2: 固定White，训练Black
        if USE_MCTS and USE_PARALLEL_MCTS:
            print(f"\n训练 Black: {black_games} 局 (使用MCTS评估胜率，并行batch={PARALLEL_BATCH_SIZE})")
            black_training_data = []
            
            # 创建并行MCTS
            black_mcts = ParallelMCTS(self.black_agent, GobangEnv, num_simulations=MCTS_NUM_SIMULATIONS, batch_size=PARALLEL_BATCH_SIZE)
            
            # 批量处理游戏
            for batch_start in tqdm(range(0, black_games, PARALLEL_BATCH_SIZE), desc="Black batches"):
                batch_end = min(batch_start + PARALLEL_BATCH_SIZE, black_games)
                actual_batch_size = batch_end - batch_start
                
                # 并行运行多个游戏
                batch_results = self.play_games_with_mcts_batch(
                    self.black_agent, self.white_agent, black_mcts, actual_batch_size
                )
                
                # 处理结果
                for game_idx, (winner, black_data, white_data) in enumerate(batch_results):
                    global_game = batch_start + game_idx + white_games
                    self._log_game(cycle_num, global_game, winner, self.env.move_history)
                    cycle_stats['black_training']['results'].append(winner)
                    black_training_data.append(black_data)
                
                # 定期训练
                if len(black_training_data) >= TRAIN_FREQUENCY:
                    recent_data = self._aggregate_training_data_mcts(black_training_data[-TRAIN_FREQUENCY:])
                    loss = self.train_agent(self.black_agent, recent_data)
                    if loss:
                        self.stats['losses'].append(loss)
        elif USE_MCTS:
            # 使用MCTS但不并行
            print(f"\n训练 Black: {black_games} 局 (使用MCTS评估胜率，串行)")
            black_training_data = []
            
            for game in tqdm(range(black_games), desc="Black"):
                winner, black_data, white_data = \
                    self.play_game_with_mcts(self.black_agent, self.white_agent, num_simulations=MCTS_NUM_SIMULATIONS)
                
                self._log_game(cycle_num, game + white_games, winner, self.env.move_history)
                cycle_stats['black_training']['results'].append(winner)
                black_training_data.append(black_data)
                
                if (game + 1) % TRAIN_FREQUENCY == 0 and len(black_training_data) > 0:
                    recent_data = self._aggregate_training_data_mcts(black_training_data[-TRAIN_FREQUENCY:])
                    loss = self.train_agent(self.black_agent, recent_data)
                    if loss:
                        self.stats['losses'].append(loss)
        else:
            # 不使用MCTS，普通训练
            print(f"\n训练 Black: {black_games} 局 (普通自对弈)")
            black_training_data = []
            
            for game in tqdm(range(black_games), desc="Black"):
                epsilon = max(0.01, 1.0 - game / black_games * 0.9)
                winner, black_data, white_data = \
                    self.play_game(self.black_agent, self.white_agent, epsilon, use_rules=self.use_rules)
                
                self._log_game(cycle_num, game + white_games, winner, self.env.move_history)
                cycle_stats['black_training']['results'].append(winner)
                black_training_data.append(black_data)
                
                if (game + 1) % TRAIN_FREQUENCY == 0 and len(black_training_data) > 0:
                    recent_data = self._aggregate_training_data(black_training_data[-TRAIN_FREQUENCY:])
                    loss = self.train_agent(self.black_agent, recent_data)
                    if loss:
                        self.stats['losses'].append(loss)
        
        cycle_stats['black_training']['segments'] = \
            self.calculate_segment_win_rates(cycle_stats['black_training']['results'])
        
        # 更新统计
        self.stats['cycle'].append(cycle_stats)
        self.stats['game_results'].extend(cycle_stats['white_training']['results'])
        self.stats['game_results'].extend(cycle_stats['black_training']['results'])
        self.stats['training_phase'].extend(['W'] * white_games)
        self.stats['training_phase'].extend(['B'] * black_games)
        
        # 打印统计
        self._print_cycle_stats(cycle_stats)
        
        # 检查是否需要触发训练保护
        if cycle_num > 0 and self.training_protection:
            need_rollback_white, need_rollback_black, adjusted_lr = self._check_training_protection(
                cycle_stats['white_training']['results'],
                cycle_stats['black_training']['results']
            )
            
            # 回退White模型
            if need_rollback_white:
                if self._rollback_model(self.white_agent, 'white', cycle_num):
                    self._adjust_learning_rate(self.white_agent, adjusted_lr)
                    self.white_consecutive_drops = 0  # 重置计数器
            
            # 回退Black模型
            if need_rollback_black:
                if self._rollback_model(self.black_agent, 'black', cycle_num):
                    self._adjust_learning_rate(self.black_agent, adjusted_lr)
                    self.black_consecutive_drops = 0  # 重置计数器
        
        # 保存模型
        self.save_models(cycle_num)
        
        return cycle_stats
    
    def _aggregate_training_data(self, game_data_list):
        """聚合多个对局的训练数据"""
        all_states = []
        all_policies = []
        all_values = []
        
        for states, policies, final_value in game_data_list:
            if len(states) > 0:
                all_states.extend(states)
                all_policies.extend(policies)
                for i in range(len(states)):
                    discount = 0.95 ** (len(states) - i - 1)
                    all_values.append(final_value * discount)
        
        return (np.array(all_states), np.array(all_policies), np.array(all_values).reshape(-1, 1))
    
    def _print_cycle_stats(self, cycle_stats):
        """打印cycle统计"""
        print(f"\n--- Cycle {cycle_stats['cycle'] + 1} 统计 ---")
        
        w_results = cycle_stats['white_training']['results']
        w_black_wins = w_results.count(BLACK)
        w_white_wins = w_results.count(WHITE)
        w_draws = w_results.count(None)
        print(f"White训练: 黑胜={w_black_wins}, 白胜={w_white_wins}, 平局={w_draws}")
        
        b_results = cycle_stats['black_training']['results']
        b_black_wins = b_results.count(BLACK)
        b_white_wins = b_results.count(WHITE)
        b_draws = b_results.count(None)
        print(f"Black训练: 黑胜={b_black_wins}, 白胜={b_white_wins}, 平局={b_draws}")
        print()
    
    def _get_model_path(self, color, cycle_num):
        """统一获取模型路径"""
        return os.path.join(self.model_dir, f"{color}_cycle_{cycle_num:03d}.pth")
    
    def _clean_old_models(self, color, current_cycle_num):
        """删除指定颜色的旧cycle模型文件（保留最近N个用于训练保护回退）"""
        import glob
        import re
        
        pattern = os.path.join(self.model_dir, f"{color}_cycle_*.pth")
        old_models = glob.glob(pattern)
        
        # 提取cycle编号并排序
        models_with_cycles = []
        for model_path in old_models:
            match = re.search(rf"{color}_cycle_(\d+)\.pth", os.path.basename(model_path))
            if match:
                cycle_num = int(match.group(1))
                models_with_cycles.append((cycle_num, model_path))
        
        # 按cycle编号排序
        models_with_cycles.sort(key=lambda x: x[0], reverse=True)
        
        # 保留最近的MODEL_KEEP_LAST_N个模型
        models_to_keep = set()
        for i, (cycle_num, _) in enumerate(models_with_cycles):
            if i < MODEL_KEEP_LAST_N:
                models_to_keep.add(cycle_num)
        
        # 删除其他旧模型
        deleted_count = 0
        for cycle_num, model_path in models_with_cycles:
            if cycle_num not in models_to_keep:
                try:
                    os.remove(model_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"  [错误] 删除失败 {model_path}: {e}")
        
        if deleted_count > 0:
            print(f"  [清理] 删除{deleted_count}个旧模型，保留最近{MODEL_KEEP_LAST_N}个")
    
    def _clean_oldest_model(self, color):
        """删除指定颜色最古老（编辑时间最早）的模型，只保留最近1个"""
        import glob
        
        pattern = os.path.join(self.model_dir, f"{color}_cycle_*.pth")
        models = glob.glob(pattern)
        
        if len(models) <= 1:
            return  # 只有1个或没有，不需要删除
        
        # 按文件修改时间排序（最古老的最前面）
        models.sort(key=lambda x: os.path.getmtime(x))
        
        # 删除最古老的（保留最后一个，即最新的）
        oldest_model = models[0]
        try:
            os.remove(oldest_model)
            print(f"  [清理] 删除最古老模型: {os.path.basename(oldest_model)}")
        except Exception as e:
            print(f"  [错误] 删除失败 {oldest_model}: {e}")
    
    def save_models(self, cycle_num):
        """保存模型，并删除最古老（编辑时间最早）的模型"""
        # 保存Black模型
        black_path = self._get_model_path('black', cycle_num)
        self.black_agent.save(black_path)
        print(f"  [保存] Black模型: {os.path.basename(black_path)}")
        self._clean_oldest_model('black')
        
        # 保存White模型
        white_path = self._get_model_path('white', cycle_num)
        self.white_agent.save(white_path)
        print(f"  [保存] White模型: {os.path.basename(white_path)}")
        self._clean_oldest_model('white')
    
    def save_stats(self):
        """保存训练统计"""
        stats_file = os.path.join(self.log_dir, "training_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def train(self, num_cycles=CYCLE, t=T):
        print(f"配置: {num_cycles} cycles, 每cycle {t} 局")
        print(f"设备: {self.black_agent.device}")
        print(f"规则检查: {'启用' if self.use_rules else '禁用'} (USE_RULES={self.use_rules})")
        print(f"日志缓冲: {self.log_buffer_size} 局 (LOG_BUFFER_SIZE={LOG_BUFFER_SIZE})")
        print(f"训练频率: 每 {TRAIN_FREQUENCY} 局训练一次")
        print(f"自适应训练: {'启用' if self.adaptive_training else '禁用'} (ADAPTIVE_TRAINING={self.adaptive_training})")
        if self.adaptive_training:
            print(f"  - 胜率阈值: {WIN_RATE_THRESHOLD:.0%}")
            print(f"  - 增强因子: {TRAINING_BOOST_FACTOR}x")
            print(f"  - 最大增强: {MAX_BOOST_FACTOR}x")
            print(f"  - 触发条件: 连续{CONSECUTIVE_LOW_WIN_ROUNDS}轮低于阈值")
        print(f"训练保护: {'启用' if self.training_protection else '禁用'} (ENABLE_TRAINING_PROTECTION={self.training_protection})")
        if self.training_protection:
            print(f"  - 下降阈值: {WIN_RATE_DROP_THRESHOLD:.0%}")
            print(f"  - 触发条件: 连续{CONSECUTIVE_DROPS_BEFORE_ROLLBACK}轮下降")
            print(f"  - 学习率调整: {LEARNING_RATE_ADJUSTMENT}x")
            print(f"  - 保留模型: 最近{MODEL_KEEP_LAST_N}轮")
        
        try:
            for cycle in range(num_cycles):
                self.train_cycle(cycle, t)  # train_cycle 内部已调用 save_models
                self.save_stats()
        finally:
            # 确保最后刷新日志缓冲区
            self._flush_log_buffer()
            print("\n[清理] 已刷新日志缓冲区")
        
        print("\n训练完成！")
        print(f"模型保存在: {self.model_dir}")


def main():
    trainer = FastTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
