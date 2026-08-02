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
    
    def get_move_log(self):
        """获取格式化的对局日志"""
        log_entries = []
        for player, row, col in self.move_history:
            color = 'B' if player == BLACK else 'W'
            log_entries.append(f"{color}({row},{col})")
        return ','.join(log_entries)
