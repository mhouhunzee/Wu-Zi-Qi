"""
五子棋训练可视化程序
生成各种训练图表
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

from config import T, SEGMENT_SIZE, VISUALIZATION_DIR, BLACK, WHITE


class Visualizer:
    """可视化工具"""
    
    def __init__(self, stats_file=None):
        self.stats_file = stats_file or os.path.join("logs", "training_stats.json")
        self.stats = None
        self.output_dir = VISUALIZATION_DIR
        self.game_log_file = os.path.join("logs", "game_logs.csv")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_game_logs(self):
        """
        从CSV文件加载游戏日志
        返回: list of dict，每个dict包含game_id, winner, gamelength, moves
        """
        if not os.path.exists(self.game_log_file):
            print(f"游戏日志文件不存在: {self.game_log_file}")
            return []
        
        game_logs = []
        try:
            with open(self.game_log_file, 'r', encoding='utf-8') as f:
                # 跳过标题行
                header = f.readline().strip()
                if not header.startswith('game_id'):
                    # 如果没有标题行，重置文件指针
                    f.seek(0)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                    
                    # 解析 game_id (格式: game{cycle}_{game_num})
                    game_id = parts[0]
                    
                    # 解析 winner (格式: winnerB/winnerW/winnerD)
                    winner_part = parts[1]
                    winner = None
                    if winner_part.startswith('winner'):
                        winner_char = winner_part[6:]  # 提取 B/W/D
                        if winner_char == 'B':
                            winner = BLACK
                        elif winner_char == 'W':
                            winner = WHITE
                        else:
                            winner = None  # 平局
                    
                    # 解析 gamelength (格式: gamelength{num})
                    length_part = parts[2]
                    game_length = 0
                    if length_part.startswith('gamelength'):
                        try:
                            game_length = int(length_part[10:])
                        except:
                            game_length = 0
                    
                    # 解析 moves (从第4个元素开始)
                    moves = []
                    for move_str in parts[3:]:
                        if move_str and len(move_str) >= 6:
                            try:
                                # 解析 B(row,col) 或 W(row,col)
                                if move_str[0] in ['B', 'W'] and '(' in move_str and ')' in move_str:
                                    color = BLACK if move_str[0] == 'B' else WHITE
                                    coords = move_str[2:-1].split(',')
                                    if len(coords) == 2:
                                        row, col = int(coords[0]), int(coords[1])
                                        moves.append((color, row, col))
                            except:
                                continue
                    
                    game_logs.append({
                        'game_id': game_id,
                        'winner': winner,
                        'game_length': game_length,
                        'moves': moves
                    })
            
            print(f"已加载 {len(game_logs)} 条游戏日志")
            return game_logs
        except Exception as e:
            print(f"读取游戏日志出错: {e}")
            return []
    
    def load_stats(self):
        """加载训练统计"""
        with open(self.stats_file, 'r') as f:
            self.stats = json.load(f)
        print(f"已加载训练统计: {self.stats_file}")
    
    def plot_segmented_win_rates(self):
        """
        绘制分段胜率直方图（上下排列）
        上半：White胜率，下半：Black胜率
        """
        if not self.stats or 'cycle' not in self.stats:
            print("没有训练数据")
            return
        
        # 收集所有分段数据
        white_segments = []
        black_segments = []
        segment_labels = []
        
        for cycle_data in self.stats['cycle']:
            cycle_num = cycle_data['cycle']
            
            # White训练阶段
            for seg in cycle_data['white_training']['segments']:
                white_segments.append(seg['white_win_rate'])
                black_segments.append(seg['black_win_rate'])
                segment_labels.append(f"C{cycle_num+1}W")
            
            # Black训练阶段
            for seg in cycle_data['black_training']['segments']:
                white_segments.append(seg['white_win_rate'])
                black_segments.append(seg['black_win_rate'])
                segment_labels.append(f"C{cycle_num+1}B")
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        x = np.arange(len(white_segments))
        width = 0.8
        
        # 上半：White胜率
        colors1 = ['#90EE90' if 'W' in label else '#FFD700' for label in segment_labels]
        bars1 = ax1.bar(x, white_segments, width, color=colors1, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('White Win Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Segmented Win Rates (50 games per segment)\nTop: White, Bottom: Black', 
                      fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, white_segments)):
            if i % 2 == 0:  # 每隔一个显示，避免拥挤
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
        
        # 下半：Black胜率
        colors2 = ['#90EE90' if 'W' in label else '#FFD700' for label in segment_labels]
        bars2 = ax2.bar(x, black_segments, width, color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Black Win Rate', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Segments', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars2, black_segments)):
            if i % 2 == 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
        
        # 设置x轴标签
        step = max(1, len(segment_labels) // 20)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels([segment_labels[i] for i in range(0, len(segment_labels), step)], 
                           rotation=45, ha='right')
        
        # 图例
        green_patch = mpatches.Patch(color='#90EE90', label='Training White')
        gold_patch = mpatches.Patch(color='#FFD700', label='Training Black')
        fig.legend(handles=[green_patch, gold_patch], loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_segmented_win_rates.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 01_segmented_win_rates.png")
        plt.close()
    
    def plot_win_rate_trend(self):
        """
        绘制胜率变化折线图
        背景用颜色区分训练阶段
        """
        if not self.stats or 'game_results' not in self.stats:
            print("没有训练数据")
            return
        
        game_results = self.stats['game_results']
        training_phase = self.stats['training_phase']
        
        # 计算滑动窗口胜率
        window_size = 100
        white_win_rates = []
        black_win_rates = []
        game_indices = []
        
        for i in range(window_size, len(game_results) + 1, window_size // 2):
            window = game_results[i-window_size:i]
            black_wins = window.count(BLACK)
            white_wins = window.count(WHITE)
            total = len(window)
            
            black_win_rates.append(black_wins / total)
            white_win_rates.append(white_wins / total)
            game_indices.append(i - window_size // 2)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # 绘制背景色块
        phase_changes = [0]
        current_phase = training_phase[0]
        for i, phase in enumerate(training_phase):
            if phase != current_phase:
                phase_changes.append(i)
                current_phase = phase
        phase_changes.append(len(training_phase))
        
        for i in range(len(phase_changes) - 1):
            start = phase_changes[i]
            end = phase_changes[i + 1]
            color = '#90EE90' if training_phase[start] == 'W' else '#FFD700'
            ax.axvspan(start, end, alpha=0.2, color=color)
        
        # 绘制胜率曲线
        ax.plot(game_indices, white_win_rates, 'b-', linewidth=2, label='White Win Rate', marker='o', markersize=3)
        ax.plot(game_indices, black_win_rates, 'r-', linewidth=2, label='Black Win Rate', marker='s', markersize=3)
        
        ax.set_xlabel('Game Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
        ax.set_title('Win Rate Trend Over Training\n(Green = Training White, Orange = Training Black)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 图例
        green_patch = mpatches.Patch(color='#90EE90', alpha=0.3, label='Training White')
        gold_patch = mpatches.Patch(color='#FFD700', alpha=0.3, label='Training Black')
        ax.legend(handles=[ax.lines[0], ax.lines[1], green_patch, gold_patch], 
                 loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_win_rate_trend.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 02_win_rate_trend.png")
        plt.close()
    
    def plot_game_length_distribution(self):
        """绘制平均对局步数变化（从CSV日志读取）"""
        # 从CSV日志文件加载
        game_logs = self.load_game_logs()
        
        if not game_logs:
            print("没有找到对局长度数据，跳过此图表")
            return
        
        # 按cycle分组计算平均步数
        cycle_lengths = defaultdict(lambda: {'white_training': [], 'black_training': []})
        
        for log in game_logs:
            game_id = log['game_id']
            game_length = log['game_length']
            
            # 解析 game_id: game{cycle}_{game_num}
            try:
                parts = game_id.split('_')
                cycle = int(parts[0][4:])  # 提取 "game" 后的数字
                game_num = int(parts[1])
                
                # 判断训练阶段：每个cycle有2*T局，前T局是White训练，后T局是Black训练
                if game_num < T:
                    cycle_lengths[cycle]['white_training'].append(game_length)
                else:
                    cycle_lengths[cycle]['black_training'].append(game_length)
            except:
                continue
        
        if not cycle_lengths:
            print("无法解析对局长度数据")
            return
        
        # 计算每个cycle的平均步数
        cycles = sorted(cycle_lengths.keys())
        white_avg_lengths = []
        black_avg_lengths = []
        
        for c in cycles:
            white_lengths = cycle_lengths[c]['white_training']
            black_lengths = cycle_lengths[c]['black_training']
            
            white_avg = np.mean(white_lengths) if white_lengths else 0
            black_avg = np.mean(black_lengths) if black_lengths else 0
            
            white_avg_lengths.append(white_avg)
            black_avg_lengths.append(black_avg)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(cycles))
        width = 0.35
        
        ax.bar(x - width/2, white_avg_lengths, width, label='Training White', color='#90EE90', alpha=0.8)
        ax.bar(x + width/2, black_avg_lengths, width, label='Training Black', color='#FFD700', alpha=0.8)
        
        ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Game Length (moves)', fontsize=12, fontweight='bold')
        ax.set_title('Average Game Length per Cycle', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c+1}' for c in cycles])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_game_length.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 03_game_length.png")
        plt.close()
    
    def plot_opening_heatmap(self):
        """绘制开局多样性热力图（从CSV日志读取）"""
        # 从CSV日志文件加载
        game_logs = self.load_game_logs()
        
        if not game_logs:
            print("没有找到开局数据，跳过开局热力图")
            return
        
        # 统计前3步的开局位置
        opening_positions = defaultdict(int)
        
        for log in game_logs:
            moves = log['moves'][:3]  # 前3步
            for player, row, col in moves:
                opening_positions[(row, col)] += 1
        
        if not opening_positions:
            print("没有找到开局数据")
            return
        
        # 创建热力图
        from config import BOARD_SIZE
        heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for (row, col), count in opening_positions.items():
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                heatmap[row, col] = count
        
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(heatmap, cmap='YlOrRd', interpolation='nearest')
        
        ax.set_title('Opening Move Distribution Heatmap\n(First 3 moves of all games)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency', fontsize=11)
        
        # 添加数值标签（只显示高频位置）
        threshold = np.percentile(list(opening_positions.values()), 90)
        for (row, col), count in opening_positions.items():
            if count >= threshold:
                ax.text(col, row, str(count), ha='center', va='center', 
                       color='white' if count > threshold * 1.5 else 'black', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_opening_heatmap.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 04_opening_heatmap.png")
        plt.close()
    
    def plot_training_loss(self):
        """绘制训练损失曲线"""
        if not self.stats or 'losses' not in self.stats or not self.stats['losses']:
            print("没有损失数据")
            return
        
        losses = self.stats['losses']
        total_loss = [l['loss'] for l in losses]
        policy_loss = [l['policy_loss'] for l in losses]
        value_loss = [l['value_loss'] for l in losses]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(total_loss, 'b-', linewidth=1, alpha=0.7)
        ax1.set_ylabel('Total Loss', fontsize=11)
        ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(policy_loss, 'g-', linewidth=1, alpha=0.7)
        ax2.set_ylabel('Policy Loss', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(value_loss, 'r-', linewidth=1, alpha=0.7)
        ax3.set_ylabel('Value Loss', fontsize=11)
        ax3.set_xlabel('Training Step', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_training_loss.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 05_training_loss.png")
        plt.close()
    
    def plot_elo_rating(self):
        """绘制ELO评分变化"""
        if not self.stats or 'game_results' not in self.stats:
            print("没有训练数据")
            return
        
        game_results = self.stats['game_results']
        
        # 计算ELO评分
        def update_elo(rating_a, rating_b, result_a, k=32):
            """更新ELO评分"""
            expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
            new_rating_a = rating_a + k * (result_a - expected_a)
            return new_rating_a
        
        black_elo = [1500]
        white_elo = [1500]
        
        for result in game_results:
            current_black = black_elo[-1]
            current_white = white_elo[-1]
            
            if result == BLACK:  # 黑胜
                new_black = update_elo(current_black, current_white, 1)
                new_white = update_elo(current_white, current_black, 0)
            elif result == WHITE:  # 白胜
                new_black = update_elo(current_black, current_white, 0)
                new_white = update_elo(current_white, current_black, 1)
            else:  # 平局
                new_black = update_elo(current_black, current_white, 0.5)
                new_white = update_elo(current_white, current_black, 0.5)
            
            black_elo.append(new_black)
            white_elo.append(new_white)
        
        # 绘制ELO变化
        fig, ax = plt.subplots(figsize=(14, 7))
        
        games = range(len(black_elo))
        ax.plot(games, black_elo, 'r-', linewidth=2, label='Black ELO', alpha=0.8)
        ax.plot(games, white_elo, 'b-', linewidth=2, label='White ELO', alpha=0.8)
        
        ax.set_xlabel('Game Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
        ax.set_title('ELO Rating Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # 添加初始和最终评分标注
        ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Initial (1500)')
        ax.text(len(games) - 1, black_elo[-1], f'  {black_elo[-1]:.0f}', 
               va='center', fontsize=10, color='red')
        ax.text(len(games) - 1, white_elo[-1], f'  {white_elo[-1]:.0f}', 
               va='center', fontsize=10, color='blue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '06_elo_rating.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 06_elo_rating.png")
        plt.close()
    
    def plot_segmented_game_length(self):
        """
        绘制每个cycle每个SEGMENT_SIZE的平均对局长度变化
        从CSV日志文件读取实际对局长度
        """
        # 从CSV日志文件加载
        game_logs = self.load_game_logs()
        
        if not game_logs:
            print("没有游戏日志数据，跳过分段对局长度图表")
            return
        
        # 提取对局长度和训练阶段
        game_lengths = []
        training_phases = []
        
        for log in game_logs:
            game_id = log['game_id']
            game_length = log['game_length']
            
            # 解析 game_id: game{cycle}_{game_num}
            try:
                parts = game_id.split('_')
                cycle = int(parts[0][4:])  # 提取 "game" 后的数字
                game_num = int(parts[1])
                
                game_lengths.append(game_length)
                
                # 判断训练阶段：每个cycle有2*T局，前T局是White训练，后T局是Black训练
                if game_num < T:
                    training_phases.append('W')
                else:
                    training_phases.append('B')
            except:
                continue
        
        if not game_lengths:
            print("无法解析对局长度数据")
            return
        
        # 按SEGMENT_SIZE分段计算平均对局长度
        segments = []
        segment_labels = []
        segment_phases = []
        
        for i in range(0, len(game_lengths), SEGMENT_SIZE):
            segment = game_lengths[i:i+SEGMENT_SIZE]
            avg_length = np.mean(segment)
            segments.append(avg_length)
            
            # 确定该段所属的训练阶段
            phase_in_segment = training_phases[i:i+SEGMENT_SIZE]
            # 取该段中出现最多的phase作为该段的phase
            from collections import Counter
            most_common_phase = Counter(phase_in_segment).most_common(1)[0][0]
            segment_phases.append(most_common_phase)
            
            # 计算cycle编号 (每个cycle有2*T局，即2*100=200局)
            cycle_num = (i // (2 * T)) + 1
            segment_in_cycle = (i % (2 * T)) // SEGMENT_SIZE + 1
            segment_labels.append(f"C{cycle_num}S{segment_in_cycle}")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(segments))
        
        # 根据训练阶段设置颜色
        colors = ['#90EE90' if phase == 'W' else '#FFD700' for phase in segment_phases]
        
        # 绘制柱状图
        bars = ax.bar(x, segments, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加趋势线
        ax.plot(x, segments, 'b-', linewidth=2, alpha=0.6, marker='o', markersize=4)
        
        ax.set_xlabel('Training Segments', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Game Length (moves)', fontsize=12, fontweight='bold')
        ax.set_title(f'Average Game Length per Segment ({SEGMENT_SIZE} games per segment)\nLower = Faster Wins, Higher = Longer Games/Draws', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加参考线
        ax.axhline(y=130, color='red', linestyle='--', alpha=0.5, label='Black Win Avg (~130)')
        ax.axhline(y=140, color='blue', linestyle='--', alpha=0.5, label='White Win Avg (~140)')
        ax.axhline(y=361, color='gray', linestyle='--', alpha=0.3, label='Draw (361)')
        
        # 设置x轴标签
        step = max(1, len(segment_labels) // 30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([segment_labels[i] for i in range(0, len(segment_labels), step)], 
                           rotation=45, ha='right', fontsize=8)
        
        # 添加数值标签（每隔几个显示一次）
        for i, (bar, val) in enumerate(zip(bars, segments)):
            if i % 2 == 0:  # 每隔一个显示
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7)
        
        # 图例
        green_patch = mpatches.Patch(color='#90EE90', label='Training White')
        gold_patch = mpatches.Patch(color='#FFD700', label='Training Black')
        ax.legend(handles=[green_patch, gold_patch], loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '08_segmented_game_length.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 08_segmented_game_length.png")
        plt.close()
    
    def plot_cycle_summary(self):
        """绘制Cycle总结对比图"""
        if not self.stats or 'cycle' not in self.stats:
            print("没有训练数据")
            return
        
        cycles = self.stats['cycle']
        cycle_nums = [c['cycle'] + 1 for c in cycles]
        
        # 收集数据
        white_wins_in_white_phase = []
        black_wins_in_white_phase = []
        white_wins_in_black_phase = []
        black_wins_in_black_phase = []
        
        for c in cycles:
            w_results = c['white_training']['results']
            white_wins_in_white_phase.append(w_results.count(WHITE))
            black_wins_in_white_phase.append(w_results.count(BLACK))
            
            b_results = c['black_training']['results']
            white_wins_in_black_phase.append(b_results.count(WHITE))
            black_wins_in_black_phase.append(b_results.count(BLACK))
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        x = np.arange(len(cycle_nums))
        width = 0.35
        
        # 训练White阶段的胜负
        ax1.bar(x - width/2, black_wins_in_white_phase, width, label='Black Wins', color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width/2, white_wins_in_white_phase, width, label='White Wins', color='#4ECDC4', alpha=0.8)
        ax1.set_ylabel('Number of Wins', fontsize=11)
        ax1.set_title('Training White Phase: Win Distribution', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'C{c}' for c in cycle_nums])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 训练Black阶段的胜负
        ax2.bar(x - width/2, black_wins_in_black_phase, width, label='Black Wins', color='#FF6B6B', alpha=0.8)
        ax2.bar(x + width/2, white_wins_in_black_phase, width, label='White Wins', color='#4ECDC4', alpha=0.8)
        ax2.set_ylabel('Number of Wins', fontsize=11)
        ax2.set_xlabel('Cycle', fontsize=11)
        ax2.set_title('Training Black Phase: Win Distribution', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'C{c}' for c in cycle_nums])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '07_cycle_summary.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 07_cycle_summary.png")
        plt.close()
    
    def generate_all(self):
        """生成所有可视化图表"""
        print("="*50)
        print("开始生成可视化图表")
        print("="*50)
        
        self.load_stats()
        
        print("\n生成图表中...")
        self.plot_segmented_win_rates()
        self.plot_win_rate_trend()
        self.plot_game_length_distribution()
        self.plot_opening_heatmap()
        self.plot_training_loss()
        self.plot_elo_rating()
        self.plot_cycle_summary()
        self.plot_segmented_game_length()
        
        print("\n" + "="*50)
        print(f"所有图表已保存到: {self.output_dir}")
        print("="*50)


def main():
    visualizer = Visualizer()
    visualizer.generate_all()


if __name__ == "__main__":
    main()
