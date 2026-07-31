"""
五子棋AI快速训练脚本 - 不记录详细日志
"""
import os
import json
import numpy as np
import torch
from tqdm import tqdm

from config import T, CYCLE, SEGMENT_SIZE, MODEL_DIR, LOG_DIR, BOARD_SIZE, BLACK, WHITE,K,REWARD,E
from gobang_env import GobangEnv
from model import GobangAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class FastTrainer:
    """快速训练管理器（最小化日志）"""
    
    def __init__(self):
        self.env = GobangEnv()
        self.black_agent = GobangAgent(BLACK)
        self.white_agent = GobangAgent(WHITE)
        
        # 只创建模型目录
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 最小化统计
        self.stats = {
            'cycle': [],
            'game_results': [],
            'training_phase': [],
            'losses': []
        }
        
        # 游戏日志文件路径
        self.game_log_file = os.path.join(LOG_DIR, "game_logs.csv")
        # 初始化日志文件（如果不存在）
        if not os.path.exists(self.game_log_file):
            with open(self.game_log_file, 'w', encoding='utf-8') as f:
                f.write("game_id,winner,gamelength,moves\n")
        
        # 全局游戏计数器
        self.global_game_count = 0
        
        # 加载已有模型（如果存在）
        self._load_existing_models()
    
    def _find_latest_model(self, color):
        """查找指定颜色最新的模型文件"""
        import glob
        pattern = os.path.join(MODEL_DIR, f"{color}_cycle_*.pth")
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
        
        # 写入日志
        log_line = f"{game_id},winner{winner_str},gamelength{game_length},{moves_str}\n"
        
        with open(self.game_log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
    
    def play_game(self, black_agent, white_agent, epsilon=0.1,K=K,R=REWARD,E=E):
        """进行一局对局（不记录日志）"""
        state = self.env.reset()
        done = False
        
        # 存储训练数据
        black_states = []
        black_policies = []
        white_states = []
        white_policies = []
        
        while not done:
            current_player = self.env.current_player
            valid_moves = self.env.get_valid_moves()
            
            if not valid_moves:
                break
            
            # 选择动作
            if current_player == BLACK:
                action = black_agent.get_action(self.env, epsilon)
                black_states.append(black_agent.board_to_input(self.env.get_state(), BLACK))
            else:
                action = white_agent.get_action(self.env, epsilon)
                white_states.append(white_agent.board_to_input(self.env.get_state(), WHITE))
            
            # 执行动作
            state, reward, done, info = self.env.step(action)
        
        # 确定胜者
        winner = self.env.get_game_result()
        game_length = len(self.env.move_history)
        max_length = BOARD_SIZE * BOARD_SIZE  # 361
        
        # ===== 奖励塑形：早结束的对局给予更大奖励 =====
        # 基础奖励
        if winner == BLACK:
            base_black_value = R
            base_white_value = -R
        elif winner == WHITE:
            base_black_value = -R
            base_white_value = R
        else:
            base_black_value = -E
            base_white_value = -E
        
        # 快速获胜奖励（步数越少，奖励越大）
        # 最短获胜步数：9步（黑方）或10步（白方）
        # 使用指数衰减，快速获胜获得显著更高的奖励
        min_win_moves = 9  # 理论最短获胜步数
        if winner is not None:  # 非平局
            # 计算额外奖励：步数越少，奖励越大
            # 使用公式：bonus = K * exp(-(game_length - min_win_moves) / 50)
            quick_win_bonus = K * np.exp(-(game_length - min_win_moves) / 50.0)
            quick_win_bonus = max(0, quick_win_bonus)  # 确保非负
            
            if winner == BLACK:
                black_value = base_black_value + quick_win_bonus
                white_value = base_white_value - quick_win_bonus
            else:  # WHITE wins
                black_value = base_black_value - quick_win_bonus
                white_value = base_white_value + quick_win_bonus
        else:
            # 平局：给予小的负面奖励，鼓励快速结束
            draw_penalty = -0.1 * (game_length / max_length)
            black_value = base_black_value + draw_penalty
            white_value = base_white_value + draw_penalty
        
        # 生成策略目标
        black_policies = self._create_policy_targets(black_states, self.env.move_history, BLACK)
        white_policies = self._create_policy_targets(white_states, self.env.move_history, WHITE)
        
        return winner, \
               (black_states, black_policies, black_value), \
               (white_states, white_policies, white_value)
    
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
    
    def train_cycle(self, cycle_num, t=T):
        """训练一个cycle"""
        print(f"\n{'='*50}")
        print(f"Cycle {cycle_num + 1}/{CYCLE}")
        print(f"{'='*50}\n")
        
        cycle_stats = {
            'cycle': cycle_num,
            'white_training': {'results': []},
            'black_training': {'results': []}
        }
        
        # 阶段1: 固定Black，训练White
        print(f"训练 White: {t} 局")
        white_training_data = []
        
        for game in tqdm(range(t), desc="White"):
            epsilon = max(0.01, 1.0 - game / t * 0.9)
            
            winner, black_data, white_data = \
                self.play_game(self.black_agent, self.white_agent, epsilon)
            
            # 记录游戏日志
            self._log_game(cycle_num, game, winner, self.env.move_history)
            
            cycle_stats['white_training']['results'].append(winner)
            white_training_data.append(white_data)
            
            # 定期训练
            if (game + 1) % 10 == 0 and len(white_training_data) > 0:
                recent_data = self._aggregate_training_data(white_training_data[-10:])
                loss = self.train_agent(self.white_agent, recent_data)
                if loss:
                    self.stats['losses'].append(loss)
        
        cycle_stats['white_training']['segments'] = \
            self.calculate_segment_win_rates(cycle_stats['white_training']['results'])
        
        # 阶段2: 固定White，训练Black
        print(f"\n训练 Black: {t} 局")
        black_training_data = []
        
        for game in tqdm(range(t), desc="Black"):
            epsilon = max(0.01, 1.0 - game / t * 0.9)
            
            winner, black_data, white_data = \
                self.play_game(self.black_agent, self.white_agent, epsilon)
            
            # 记录游戏日志
            self._log_game(cycle_num, game + t, winner, self.env.move_history)
            
            cycle_stats['black_training']['results'].append(winner)
            black_training_data.append(black_data)
            
            if (game + 1) % 10 == 0 and len(black_training_data) > 0:
                recent_data = self._aggregate_training_data(black_training_data[-10:])
                loss = self.train_agent(self.black_agent, recent_data)
                if loss:
                    self.stats['losses'].append(loss)
        
        cycle_stats['black_training']['segments'] = \
            self.calculate_segment_win_rates(cycle_stats['black_training']['results'])
        
        # 更新统计
        self.stats['cycle'].append(cycle_stats)
        self.stats['game_results'].extend(cycle_stats['white_training']['results'])
        self.stats['game_results'].extend(cycle_stats['black_training']['results'])
        self.stats['training_phase'].extend(['W'] * t)
        self.stats['training_phase'].extend(['B'] * t)
        
        # 打印统计
        self._print_cycle_stats(cycle_stats)
        
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
        return os.path.join(MODEL_DIR, f"{color}_cycle_{cycle_num:03d}.pth")
    
    def _clean_old_models(self, color, current_cycle_num):
        """删除指定颜色的旧cycle模型文件（保留当前cycle）"""
        import glob
        pattern = os.path.join(MODEL_DIR, f"{color}_cycle_*.pth")
        old_models = glob.glob(pattern)
        for model_path in old_models:
            if f"{color}_cycle_{current_cycle_num:03d}.pth" in model_path:
                continue
            try:
                os.remove(model_path)
                print(f"  [清理] 删除旧模型: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"  [错误] 删除失败 {model_path}: {e}")
    
    def save_models(self, cycle_num):
        """保存模型（保存前删除同色的旧cycle模型，只保留当前）"""
        # 保存Black模型
        black_path = self._get_model_path('black', cycle_num)
        self.black_agent.save(black_path)
        print(f"  [保存] Black模型: {os.path.basename(black_path)}")
        self._clean_old_models('black', cycle_num)
        
        # 保存White模型
        white_path = self._get_model_path('white', cycle_num)
        self.white_agent.save(white_path)
        print(f"  [保存] White模型: {os.path.basename(white_path)}")
        self._clean_old_models('white', cycle_num)
    
    def save_stats(self):
        """保存训练统计"""
        stats_file = os.path.join(LOG_DIR, "training_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def train(self, num_cycles=CYCLE, t=T):
        """完整训练流程"""
        print("开始快速训练（不记录详细日志）")
        print(f"配置: {num_cycles} cycles, 每cycle {t} 局")
        print(f"设备: {self.black_agent.device}")
        
        for cycle in range(num_cycles):
            self.train_cycle(cycle, t)
            self.save_stats()
        
        print("\n训练完成！")
        print(f"模型保存在: {MODEL_DIR}")


def main():
    trainer = FastTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
