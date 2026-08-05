"""
五子棋环境 - 实现游戏规则和状态管理
"""
import numpy as np
from config import BOARD_SIZE, EMPTY, BLACK, WHITE


class GobangEnv:
    """五子棋游戏环境"""
    
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = BLACK  # 黑方先手
        self.move_history = []  # 记录落子历史
        self.done = False
        self.winner = None
        return self.get_state()
    
    def get_state(self):
        """获取当前游戏状态"""
        return self.board.copy()
    
    def get_valid_moves(self):
        """获取所有合法落子位置"""
        if self.done:
            return []
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == EMPTY:
                    valid_moves.append((row, col))
        return valid_moves
    
    def step(self, action):
        """
        执行一步动作
        action: (row, col) 落子位置
        返回: (next_state, reward, done, info)
        """
        row, col = action
        
        # 检查动作合法性
        if self.board[row, col] != EMPTY:
            raise ValueError(f"Invalid move: position ({row}, {col}) is already occupied")
        
        # 执行落子
        self.board[row, col] = self.current_player
        self.move_history.append((self.current_player, row, col))
        
        # 检查游戏是否结束
        if self.check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0 if self.current_player == BLACK else -1.0
        elif len(self.move_history) >= self.board_size * self.board_size:
            # 棋盘填满，平局
            self.done = True
            self.winner = None
            reward = 0.0
        else:
            # 游戏继续
            reward = 0.0
        
        # 切换玩家
        if not self.done:
            self.current_player = WHITE if self.current_player == BLACK else BLACK
        
        info = {
            'current_player': self.current_player,
            'winner': self.winner,
            'move_history': self.move_history.copy()
        }
        
        return self.get_state(), reward, self.done, info
    
    def check_win(self, row, col):
        """检查在(row, col)落子后是否获胜"""
        player = self.board[row, col]
        if player == EMPTY:
            return False
        
        # 四个方向：水平、垂直、对角线、反对角线
        directions = [
            [(0, 1), (0, -1)],   # 水平
            [(1, 0), (-1, 0)],   # 垂直
            [(1, 1), (-1, -1)],  # 对角线
            [(1, -1), (-1, 1)]   # 反对角线
        ]
        
        for direction in directions:
            count = 1  # 当前落子算1个
            
            # 向两个方向检查
            for dr, dc in direction:
                r, c = row + dr, col + dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.board[r, c] == player:
                        count += 1
                        r += dr
                        c += dc
                    else:
                        break
            
            if count >= 5:
                return True
        
        return False
    
    def get_game_result(self):
        """获取游戏结果"""
        if not self.done:
            return None
        return self.winner
    
    def render(self):
        """打印棋盘（用于调试）"""
        symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
        print('  ' + ' '.join(str(i % 10) for i in range(self.board_size)))
        for i, row in enumerate(self.board):
            print(f"{i % 10} " + ' '.join(symbols[cell] for cell in row))
        print()
    
    def clone(self):
        """克隆当前环境状态"""
        new_env = GobangEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.move_history = self.move_history.copy()
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env
    
    def count_attack_patterns(self, player):
        """
        统计指定玩家的进攻局面数量
        进攻定义：活三（open three）、跳活三或 四连（four-in-a-row，差一步五连）
        
        返回: (活三数量, 四连数量)
        """
        open_three_count = 0
        four_count = 0
        
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 对角线
            (1, -1)   # 反对角线
        ]
        
        # 检查每个方向上的所有连线
        checked_patterns = set()  # 避免重复计数
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] != player:
                    continue
                
                for dr, dc in directions:
                    # 确保只处理每个方向一次（以该方向上最左上/最左下的棋子为起点）
                    start_key = (row, col, dr, dc)
                    if start_key in checked_patterns:
                        continue
                    
                    # 检查是否是该方向上最开始的棋子
                    prev_r, prev_c = row - dr, col - dc
                    if 0 <= prev_r < self.board_size and 0 <= prev_c < self.board_size:
                        if self.board[prev_r, prev_c] == player:
                            continue  # 不是起点，跳过
                    
                    # 统计该方向上的连续棋子数和空位情况
                    pieces = []
                    r, c = row, col
                    while 0 <= r < self.board_size and 0 <= c < self.board_size:
                        pieces.append((r, c, self.board[r, c]))
                        r += dr
                        c += dc
                    
                    # 分析这个序列
                    ot, f = self._analyze_sequence(pieces, player, checked_patterns)
                    open_three_count += ot
                    four_count += f
        
        # 额外检测跳活三（需要检查所有可能的跳活三模式）
        jump_three_count = self._count_jump_threes(player)
        open_three_count += jump_three_count
        
        return open_three_count, four_count
    
    def _count_jump_threes(self, player):
        """
        专门检测跳活三数量
        跳活三模式：
        - XX_X: 两子连，一空，一子，两端为空
        - X_XX: 一子，一空，两子连，两端为空
        """
        detected_patterns = set()  # 用于去重
        
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 对角线
            (1, -1)   # 反对角线
        ]
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] != player:
                    continue
                
                for dr, dc in directions:
                    # 检查XX_X模式：从当前棋子开始，检查 (0,0), (0,1), (0,2), (0,3) 模式
                    # 即：player, player, empty, player
                    r1, c1 = row + dr, col + dc
                    r2, c2 = row + 2*dr, col + 2*dc
                    r3, c3 = row + 3*dr, col + 3*dc
                    
                    # 检查边界
                    if not (0 <= r1 < self.board_size and 0 <= c1 < self.board_size):
                        continue
                    if not (0 <= r2 < self.board_size and 0 <= c2 < self.board_size):
                        continue
                    if not (0 <= r3 < self.board_size and 0 <= c3 < self.board_size):
                        continue
                    
                    # XX_X模式：当前, player, empty, player
                    if (self.board[r1, c1] == player and 
                        self.board[r2, c2] == 0 and 
                        self.board[r3, c3] == player):
                        # 检查两端是否为空
                        prev_r, prev_c = row - dr, col - dc
                        next_r, next_c = row + 4*dr, col + 4*dc
                        
                        left_empty = (prev_r < 0 or prev_r >= self.board_size or 
                                     prev_c < 0 or prev_c >= self.board_size or 
                                     self.board[prev_r, prev_c] == 0)
                        right_empty = (next_r < 0 or next_r >= self.board_size or 
                                      next_c < 0 or next_c >= self.board_size or 
                                      self.board[next_r, next_c] == 0)
                        
                        if left_empty and right_empty:
                            # 使用排序后的位置作为唯一标识，避免重复计数
                            pattern_key = tuple(sorted([(row, col), (r1, c1), (r3, c3)]))
                            if pattern_key not in detected_patterns:
                                detected_patterns.add(pattern_key)
                    
                    # X_XX模式：当前, empty, player, player
                    if (self.board[r1, c1] == 0 and 
                        self.board[r2, c2] == player and 
                        self.board[r3, c3] == player):
                        # 检查两端是否为空
                        prev_r, prev_c = row - dr, col - dc
                        next_r, next_c = row + 4*dr, col + 4*dc
                        
                        left_empty = (prev_r < 0 or prev_r >= self.board_size or 
                                     prev_c < 0 or prev_c >= self.board_size or 
                                     self.board[prev_r, prev_c] == 0)
                        right_empty = (next_r < 0 or next_r >= self.board_size or 
                                      next_c < 0 or next_c >= self.board_size or 
                                      self.board[next_r, next_c] == 0)
                        
                        if left_empty and right_empty:
                            # 使用排序后的位置作为唯一标识，避免重复计数
                            pattern_key = tuple(sorted([(row, col), (r2, c2), (r3, c3)]))
                            if pattern_key not in detected_patterns:
                                detected_patterns.add(pattern_key)
        
        return len(detected_patterns)
    
    def _analyze_sequence(self, pieces, player, checked_patterns):
        """分析一条线上的棋子序列，检测活三、跳活三和四连
        返回: (open_three_count, four_count)
        """
        open_three_count = 0
        four_count = 0
        
        # 提取棋子位置序列（0=空，1=player，2=对手）
        sequence = [p[2] for p in pieces]
        n = len(sequence)
        
        # 检测四连: 连续4个player
        i = 0
        while i < n:
            if sequence[i] != player:
                i += 1
                continue
            
            # 找到连续的玩家棋子
            consecutive_count = 0
            start = i
            while i < n and sequence[i] == player:
                consecutive_count += 1
                i += 1
            
            if consecutive_count == 4:
                four_count += 1
        
        # 检测活三: 连续3个player，两边都是空的
        i = 0
        while i < n:
            if sequence[i] != player:
                i += 1
                continue
            
            # 找到连续的玩家棋子
            consecutive_count = 0
            start = i
            while i < n and sequence[i] == player:
                consecutive_count += 1
                i += 1
            end = i - 1
            
            if consecutive_count == 3:
                left_empty = (start == 0) or (start > 0 and sequence[start - 1] == 0)
                right_empty = (end == n - 1) or (end < n - 1 and sequence[end + 1] == 0)
                if left_empty and right_empty:
                    open_three_count += 1
        
        # 检测跳活三: 3个player中间隔1个空位，两边都是空的
        # 模式: 空-Player-Player-空-Player-空 (XX_X) 或 空-Player-空-Player-Player-空 (X_XX)
        # 也支持不从边界开始的情况: X-Player-Player-空-Player-空-X
        i = 0
        while i <= n - 5:  # 需要至少5个位置
            # 检查模式: ..., 0, player, player, 0, player, 0, ... (XX_X)
            if (i + 4 < n and
                sequence[i + 1] == player and 
                sequence[i + 2] == player and 
                sequence[i + 3] == 0 and 
                sequence[i + 4] == player):
                # 检查左边是否为空或边界
                left_ok = (i == 0) or (sequence[i] == 0)
                # 检查右边是否为空或边界
                right_ok = (i + 5 >= n) or (sequence[i + 5] == 0)
                if left_ok and right_ok:
                    open_three_count += 1
                    i += 5
                    continue
            
            # 检查模式: ..., 0, player, 0, player, player, 0, ... (X_XX)
            if (i + 4 < n and
                sequence[i + 1] == player and 
                sequence[i + 2] == 0 and 
                sequence[i + 3] == player and 
                sequence[i + 4] == player):
                # 检查左边是否为空或边界
                left_ok = (i == 0) or (sequence[i] == 0)
                # 检查右边是否为空或边界
                right_ok = (i + 5 >= n) or (sequence[i + 5] == 0)
                if left_ok and right_ok:
                    open_three_count += 1
                    i += 5
                    continue
            
            i += 1
        
        return open_three_count, four_count
    
    def is_attack_move(self, row, col, player):
        """
        判断在(row, col)落子是否构成进攻
        进攻：形成活三、跳活三或四连
        
        返回: (是否进攻, 进攻类型) 进攻类型: 'open_three', 'four', 'none'
        """
        if self.board[row, col] != 0:
            return False, 'none'
        
        # 临时落子
        original = self.board[row, col]
        self.board[row, col] = player
        
        directions = [
            [(0, 1), (0, -1)],   # 水平
            [(1, 0), (-1, 0)],   # 垂直
            [(1, 1), (-1, -1)],  # 对角线
            [(1, -1), (-1, 1)]   # 反对角线
        ]
        
        is_four = False
        is_open_three = False
        
        for direction in directions:
            # 获取该方向上的完整序列（包括空位）
            # 正向
            forward_pieces = []
            r, c = row + direction[0][0], col + direction[0][1]
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                forward_pieces.append(self.board[r, c])
                r += direction[0][0]
                c += direction[0][1]
            
            # 反向
            backward_pieces = []
            r, c = row + direction[1][0], col + direction[1][1]
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                backward_pieces.append(self.board[r, c])
                r += direction[1][0]
                c += direction[1][1]
            
            # 合并序列：反向（倒序）+ 当前落子 + 正向
            sequence = list(reversed(backward_pieces)) + [player] + forward_pieces
            current_idx = len(backward_pieces)  # 当前落子在序列中的位置
            
            # 统计连续棋子数
            # 向左（反向）
            left_count = 0
            left_empty = False
            i = current_idx - 1
            while i >= 0:
                if sequence[i] == player:
                    left_count += 1
                    i -= 1
                elif sequence[i] == 0:
                    left_empty = True
                    break
                else:
                    break
            
            # 向右（正向）
            right_count = 0
            right_empty = False
            i = current_idx + 1
            while i < len(sequence):
                if sequence[i] == player:
                    right_count += 1
                    i += 1
                elif sequence[i] == 0:
                    right_empty = True
                    break
                else:
                    break
            
            total_count = 1 + left_count + right_count
            
            # 检测四连
            if total_count >= 4:
                is_four = True
                break
            
            # 检测活三（3连，两边都是空的）
            if total_count == 3 and left_empty and right_empty:
                is_open_three = True
                continue
            
            # 检测跳活三
            # 模式1: 当前落子与左边形成2连，右边隔1空有1子，且远端为空
            # 模式2: 当前落子与右边形成2连，左边隔1空有1子，且远端为空
            if left_empty and right_empty:
                # 模式1: 检查 X_XX (左1子，当前，右1子，隔1空，再1子)
                if left_count >= 1 and right_count == 1:
                    # 检查右边是否隔1空有1子
                    check_pos = current_idx + 2
                    if check_pos < len(sequence) and sequence[check_pos] == 0:
                        if check_pos + 1 < len(sequence) and sequence[check_pos + 1] == player:
                            # 检查远端是否为空
                            if check_pos + 2 >= len(sequence) or sequence[check_pos + 2] == 0:
                                is_open_three = True
                                continue
                
                # 模式2: 检查 XX_X (左1子，隔1空，再1子，当前，右1子)
                if left_count == 1 and right_count >= 1:
                    # 检查左边是否隔1空有1子
                    check_pos = current_idx - 2
                    if check_pos >= 0 and sequence[check_pos] == 0:
                        if check_pos - 1 >= 0 and sequence[check_pos - 1] == player:
                            # 检查远端是否为空
                            if check_pos - 2 < 0 or sequence[check_pos - 2] == 0:
                                is_open_three = True
                                continue
                
                # 模式3: 当前落子在中间，形成 X_XX (左1子，隔1空，当前，右1子)
                if left_count == 0 and right_count == 1:
                    # 左边隔1空有1子
                    if current_idx - 2 >= 0 and sequence[current_idx - 2] == 0:
                        if current_idx - 3 >= 0 and sequence[current_idx - 3] == player:
                            # 远端为空
                            if current_idx - 4 < 0 or sequence[current_idx - 4] == 0:
                                is_open_three = True
                                continue
                
                # 模式4: 当前落子在中间，形成 XX_X (左1子，当前，隔1空，右1子)
                if left_count == 1 and right_count == 0:
                    # 右边隔1空有1子
                    if current_idx + 2 < len(sequence) and sequence[current_idx + 2] == 0:
                        if current_idx + 3 < len(sequence) and sequence[current_idx + 3] == player:
                            # 远端为空
                            if current_idx + 4 >= len(sequence) or sequence[current_idx + 4] == 0:
                                is_open_three = True
                                continue
        
        # 恢复棋盘
        self.board[row, col] = original
        
        if is_four:
            return True, 'four'
        elif is_open_three:
            return True, 'open_three'
        else:
            return False, 'none'
    
    def get_move_log(self):
        """获取格式化的对局日志"""
        log_entries = []
        for player, row, col in self.move_history:
            color = 'B' if player == BLACK else 'W'
            log_entries.append(f"{color}({row},{col})")
        return ','.join(log_entries)
