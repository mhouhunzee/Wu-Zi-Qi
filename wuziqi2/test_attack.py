"""
测试进攻检测功能
"""
import numpy as np
from gobang_env import GobangEnv
from config import BLACK, WHITE

def test_attack_detection():
    """测试进攻检测功能"""
    env = GobangEnv()
    
    # 测试1: 活三检测
    print("=" * 50)
    print("测试1: 活三检测")
    env.reset()
    # 创建一个活三: 黑棋在 (9,8), (9,9), (9,10)，两边都是空的
    env.board[9, 8] = BLACK
    env.board[9, 9] = BLACK
    env.board[9, 10] = BLACK
    env.current_player = BLACK
    
    # 在 (9,7) 落子应该形成活四
    is_attack, attack_type = env.is_attack_move(9, 7, BLACK)
    print(f"在(9,7)落子形成活四: {is_attack}, 类型: {attack_type}")
    
    # 在 (9,11) 落子应该形成活四
    is_attack, attack_type = env.is_attack_move(9, 11, BLACK)
    print(f"在(9,11)落子形成活四: {is_attack}, 类型: {attack_type}")
    
    # 测试2: 四连检测
    print("\n" + "=" * 50)
    print("测试2: 四连检测")
    env.reset()
    # 创建一个四连: 黑棋在 (9,7), (9,8), (9,9), (9,10)
    env.board[9, 7] = BLACK
    env.board[9, 8] = BLACK
    env.board[9, 9] = BLACK
    env.board[9, 10] = BLACK
    env.current_player = BLACK
    
    # 在 (9,6) 或 (9,11) 落子形成五连
    is_attack, attack_type = env.is_attack_move(9, 6, BLACK)
    print(f"在(9,6)落子形成五连: {is_attack}, 类型: {attack_type}")
    
    is_attack, attack_type = env.is_attack_move(9, 11, BLACK)
    print(f"在(9,11)落子形成五连: {is_attack}, 类型: {attack_type}")
    
    # 测试3: 非进攻落子
    print("\n" + "=" * 50)
    print("测试3: 非进攻落子")
    env.reset()
    env.board[9, 9] = BLACK
    env.current_player = BLACK
    
    is_attack, attack_type = env.is_attack_move(9, 10, BLACK)
    print(f"在(9,10)落子（单个棋子旁边）: {is_attack}, 类型: {attack_type}")
    
    is_attack, attack_type = env.is_attack_move(5, 5, BLACK)
    print(f"在(5,5)落子（空白处）: {is_attack}, 类型: {attack_type}")
    
    # 测试4: 活三计数
    print("\n" + "=" * 50)
    print("测试4: 活三和四连计数")
    env.reset()
    # 创建两个活三
    env.board[9, 8] = BLACK
    env.board[9, 9] = BLACK
    env.board[9, 10] = BLACK  # 水平活三
    env.board[8, 9] = BLACK
    env.board[10, 9] = BLACK  # 垂直活三
    
    open_three, four = env.count_attack_patterns(BLACK)
    print(f"活三数量: {open_three}, 四连数量: {four}")
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    test_attack_detection()
