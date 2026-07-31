"""
五子棋AI模型 - 使用深度神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import BOARD_SIZE, BLACK, WHITE, CONV_FILTERS, HIDDEN_SIZE


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GobangNet(nn.Module):
    """五子棋神经网络"""
    def __init__(self, board_size=BOARD_SIZE):
        super(GobangNet, self).__init__()
        self.board_size = board_size

        # 输入通道:2个平面(黑棋位置、白棋位置)
        self.conv_input = nn.Conv2d(2, CONV_FILTERS[0], kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(CONV_FILTERS[0])

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(CONV_FILTERS[0]) for _ in range(4)
        ])

        # 策略头(输出落子概率)
        self.conv_policy = nn.Conv2d(CONV_FILTERS[0], 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # 价值头(输出胜率评估)
        self.conv_value = nn.Conv2d(CONV_FILTERS[0], 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(board_size * board_size, HIDDEN_SIZE)
        self.fc_value2 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        """
        前向传播
        x: [batch_size, 2, board_size, board_size]
        返回: (policy, value)
        policy: [batch_size, board_size * board_size] 落子概率
        value: [batch_size, 1] 胜率评估 [-1, 1]
        """
        # 共享层
        x = F.relu(self.bn_input(self.conv_input(x)))
        for res_block in self.res_blocks:
            x = res_block(x)

        # 策略头
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)
        policy = F.log_softmax(policy, dim=1)

        # 价值头
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))

        return policy, value


class GobangAgent:
    """五子棋AI代理"""
    def __init__(self, color, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.color = color  # BLACK or WHITE
        self.device = device
        self.model = GobangNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.training_history = []

    def board_to_input(self, board, current_player):
        """
        将棋盘转换为网络输入
        board: [board_size, board_size]
        返回: [2, board_size, board_size]
        """
        # 通道0:当前玩家的棋子位置
        # 通道1:对手的棋子位置
        input_tensor = np.zeros((2, self.model.board_size, self.model.board_size), dtype=np.float32)

        if current_player == BLACK:
            input_tensor[0] = (board == BLACK).astype(np.float32)
            input_tensor[1] = (board == WHITE).astype(np.float32)
        else:
            input_tensor[0] = (board == WHITE).astype(np.float32)
            input_tensor[1] = (board == BLACK).astype(np.float32)

        return input_tensor

    def _check_winning_move(self, board, player, valid_moves):
        """
        检查是否有立即获胜的位置（完成五连）
        返回: (row, col) 或 None
        """
        for row, col in valid_moves:
            if self._would_win(board, row, col, player):
                return (row, col)
        return None
    
    def _check_blocking_move(self, board, player, valid_moves):
        """
        检查是否需要阻止对手获胜（对手有四连）
        返回: (row, col) 或 None
        """
        opponent = WHITE if player == BLACK else BLACK
        for row, col in valid_moves:
            if self._would_win(board, row, col, opponent):
                return (row, col)
        return None
    
    def _would_win(self, board, row, col, player):
        """检查在(row, col)落子后是否获胜"""
        # 临时落子
        original = board[row, col]
        board[row, col] = player
        
        # 检查是否五连
        directions = [
            [(0, 1), (0, -1)],   # 水平
            [(1, 0), (-1, 0)],   # 垂直
            [(1, 1), (-1, -1)],  # 对角线
            [(1, -1), (-1, 1)]   # 反对角线
        ]
        
        won = False
        for direction in directions:
            count = 1
            for dr, dc in direction:
                r, c = row + dr, col + dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
            if count >= 5:
                won = True
                break
        
        # 恢复棋盘
        board[row, col] = original
        return won
    
    def _check_open_four(self, board, player, valid_moves):
        """
        检查是否有活四（两边都是空的四连）
        返回: (row, col) 或 None
        """
        best_move = None
        best_score = 0
        
        for row, col in valid_moves:
            score = self._count_open_four(board, row, col, player)
            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move if best_score > 0 else None
    
    def _count_open_four(self, board, row, col, player):
        """计算在(row, col)落子后形成的活四数量"""
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 对角线
            (1, -1)   # 反对角线
        ]
        
        total_score = 0
        original = board[row, col]
        board[row, col] = player
        
        for dr, dc in directions:
            # 计算该方向上的连续棋子数
            count = 1
            empty_ends = 0
            
            # 正向
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                else:
                    if board[r, c] == 0:  # 空位
                        empty_ends += 1
                    break
            
            # 反向
            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                else:
                    if board[r, c] == 0:  # 空位
                        empty_ends += 1
                    break
            
            # 活四: 4连且两边都是空的
            if count == 4 and empty_ends == 2:
                total_score += 100
            # 冲四: 4连但一边被堵
            elif count == 4 and empty_ends == 1:
                total_score += 50
        
        board[row, col] = original
        return total_score
    
    def get_action(self, env, epsilon=0.0):
        """
        根据当前状态选择动作
        env: GobangEnv 环境
        epsilon: 探索率
        返回: (row, col) 落子位置
        """
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return None
        
        # Epsilon-贪婪策略
        if np.random.random() < epsilon:
            return valid_moves[np.random.randint(len(valid_moves))]
        
        board = env.get_state()
        current_player = env.current_player
        
        # ===== 规则层 =====
        # 1. 检查是否有立即获胜的位置
        winning_move = self._check_winning_move(board, current_player, valid_moves)
        if winning_move:
            return winning_move
        
        # 2. 检查是否需要阻止对手获胜
        blocking_move = self._check_blocking_move(board, current_player, valid_moves)
        if blocking_move:
            return blocking_move
        
        # 3. 检查是否有活四/冲四
        open_four_move = self._check_open_four(board, current_player, valid_moves)
        if open_four_move:
            return open_four_move
        
        # 4. 检查是否需要阻止对手的活四
        opponent = WHITE if current_player == BLACK else BLACK
        opponent_open_four = self._check_open_four(board, opponent, valid_moves)
        if opponent_open_four:
            return opponent_open_four
        
        # ===== 神经网络 =====
        # 使用模型预测
        try:
            state = self.board_to_input(board, current_player)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy, value = self.model(state_tensor)
                policy = torch.exp(policy).cpu().numpy()[0]
            
            # 只考虑合法动作
            valid_moves_flat = [r * self.model.board_size + c for r, c in valid_moves]
            valid_policy = policy[valid_moves_flat]
            
            # 检查valid_policy是否有效
            if valid_policy.sum() == 0 or np.isnan(valid_policy).any():
                print(f"警告: 策略输出无效，使用随机选择")
                return valid_moves[np.random.randint(len(valid_moves))]
            
            valid_policy = valid_policy / valid_policy.sum()  # 重新归一化
            
            # 选择概率最高的动作
            best_idx = np.argmax(valid_policy)
            return valid_moves[best_idx]
        except Exception as e:
            print(f"get_action出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时随机选择
            return valid_moves[np.random.randint(len(valid_moves))]

    def get_win_probability(self, env):
        """
        获取当前局面的胜率评估
        返回: float [0, 1] 表示当前玩家的胜率
        """
        try:
            state = self.board_to_input(env.get_state(), env.current_player)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, value = self.model(state_tensor)
                value = value.cpu().numpy()[0][0]
            
            # value范围是[-1, 1]，转换为[0, 1]的胜率
            win_prob = (value + 1) / 2
            return float(win_prob)
        except Exception as e:
            print(f"get_win_probability出错: {e}")
            return 0.5

    def train_step(self, states, policies, values):
        """
        训练一步
        states: [batch_size, 2, board_size, board_size]
        policies: [batch_size, board_size * board_size] 目标策略
        values: [batch_size, 1] 目标价值
        """
        states = torch.FloatTensor(states).to(self.device)
        target_policies = torch.FloatTensor(policies).to(self.device)
        target_values = torch.FloatTensor(values).to(self.device)

        # 前向传播
        pred_policies, pred_values = self.model(states)

        # 计算损失
        policy_loss = -torch.mean(torch.sum(target_policies * pred_policies, dim=1))
        value_loss = F.mse_loss(pred_values, target_values)
        loss = policy_loss + value_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'color': self.color,
            'training_history': self.training_history
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.color = checkpoint['color']
        self.training_history = checkpoint.get('training_history', [])
