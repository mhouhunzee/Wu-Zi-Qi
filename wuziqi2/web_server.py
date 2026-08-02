"""
五子棋Web服务器 - 提供人机对弈界面
"""
import os
import json
import random
import glob
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from config import BOARD_SIZE, BLACK, WHITE, MODEL_DIR
from gobang_env import GobangEnv
from model import GobangAgent


def find_latest_model(color):
    """查找指定颜色最新的模型文件"""
    pattern = os.path.join(MODEL_DIR, f"{color}_cycle_*.pth")
    models = glob.glob(pattern)
    if not models:
        return None
    models.sort()
    return models[-1]

app = Flask(__name__)
CORS(app)

# 全局游戏状态
game_env = None
ai_agent = None
player_color = None
ai_color = None


def load_agent(color):
    """加载指定颜色的AI模型"""
    agent = GobangAgent(color)
    color_name = 'white' if color == WHITE else 'black'
    model_path = find_latest_model(color_name)
    
    if model_path and os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"已加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}，将使用随机策略")
    else:
        print(f"警告: 未找到{color_name}模型文件，使用随机策略")
    
    return agent


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_game():
    """开始新游戏"""
    global game_env, ai_agent, player_color, ai_color
    
    try:
        data = request.json
        print(f"start_game 收到请求: {data}")
        
        player_choice = data.get('player_color', 'black')  # 'black' 或 'white'
        
        # 设置颜色
        if player_choice == 'black':
            player_color = BLACK
            ai_color = WHITE
            ai_agent = load_agent(WHITE)
        else:
            player_color = WHITE
            ai_color = BLACK
            ai_agent = load_agent(BLACK)
        
        # 初始化游戏环境
        game_env = GobangEnv()
        game_env.reset()
        
        # 如果AI先手，AI落子
        ai_move = None
        if ai_color == BLACK:
            print("AI先手，AI落子")
            ai_move = ai_agent.get_action(game_env, epsilon=0.0)
            print(f"AI落子: {ai_move}")
            if ai_move:
                game_env.step(ai_move)
                print(f"AI落子后当前玩家: {game_env.current_player}")
        
        response = {
            'board': game_env.board.tolist(),
            'current_player': int(game_env.current_player),
            'player_color': int(player_color),
            'ai_color': int(ai_color),
            'ai_move': (int(ai_move[0]), int(ai_move[1])) if ai_move else None,
            'game_over': bool(game_env.done),
            'winner': int(game_env.winner) if game_env.winner is not None else None,
            'ai_win_probability': float(ai_agent.get_win_probability(game_env)) if ai_agent else 0.5
        }
        
        print(f"start_game 返回: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"start_game 出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/move', methods=['POST'])
def make_move():
    """玩家落子"""
    global game_env, ai_agent
    
    try:
        if game_env is None or game_env.done:
            return jsonify({'error': '游戏未开始或已结束'}), 400
        
        data = request.json
        row = data.get('row')
        col = data.get('col')
        
        print(f"make_move: 玩家落子 ({row}, {col}), 当前玩家: {game_env.current_player}, player_color: {player_color}")
        
        # 检查是否轮到玩家
        if game_env.current_player != player_color:
            return jsonify({'error': '不是玩家的回合'}), 400
        
        # 检查落子是否合法
        valid_moves = game_env.get_valid_moves()
        if (row, col) not in valid_moves:
            return jsonify({'error': '非法落子位置'}), 400
        
        # 玩家落子
        game_env.step((row, col))
        
        # 检查游戏是否结束
        if game_env.done:
            return jsonify({
                'board': game_env.board.tolist(),
                'current_player': int(game_env.current_player),
                'player_color': int(player_color),
                'ai_color': int(ai_color),
                'game_over': True,
                'winner': int(game_env.winner) if game_env.winner is not None else None,
                'move_log': game_env.get_move_log()
            })
        
        # AI落子
        ai_move = None
        ai_win_prob = 0.5
        
        print(f"AI落子前 - 当前玩家: {game_env.current_player}, AI颜色: {ai_color}, 游戏结束: {game_env.done}")
        
        if ai_agent and not game_env.done:
            try:
                ai_move = ai_agent.get_action(game_env, epsilon=0.0)
                print(f"AI选择落子: {ai_move}")
                if ai_move:
                    game_env.step(ai_move)
                    ai_win_prob = ai_agent.get_win_probability(game_env)
                    print(f"AI落子成功: {ai_move}, 新状态 - 当前玩家: {game_env.current_player}, 游戏结束: {game_env.done}")
                else:
                    print(f"警告: AI返回None，无法落子")
            except Exception as e:
                print(f"AI落子出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 检查AI落子后游戏是否结束
        if game_env.done:
            print(f"游戏结束，获胜者: {game_env.winner}")
            return jsonify({
                'board': game_env.board.tolist(),
                'current_player': int(game_env.current_player),
                'player_color': int(player_color),
                'ai_color': int(ai_color),
                'player_move': (int(row), int(col)),
                'ai_move': (int(ai_move[0]), int(ai_move[1])) if ai_move else None,
                'game_over': True,
                'winner': int(game_env.winner) if game_env.winner is not None else None,
                'ai_win_probability': float(ai_win_prob),
                'move_log': game_env.get_move_log()
            })
        
        response = {
            'board': game_env.board.tolist(),
            'current_player': int(game_env.current_player),
            'player_color': int(player_color),
            'ai_color': int(ai_color),
            'player_move': (int(row), int(col)),
            'ai_move': (int(ai_move[0]), int(ai_move[1])) if ai_move else None,
            'game_over': bool(game_env.done),
            'winner': int(game_env.winner) if game_env.winner is not None else None,
            'ai_win_probability': float(ai_win_prob),
            'move_log': game_env.get_move_log() if game_env.done else None
        }
        
        print(f"返回数据: current_player={response['current_player']}, player_color={response['player_color']}")
        return jsonify(response)
    except Exception as e:
        print(f"make_move 出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/state', methods=['GET'])
def get_state():
    """获取当前游戏状态"""
    if game_env is None:
        return jsonify({
            'game_started': False,
            'message': '游戏未开始，请点击"开始游戏"'
        }), 200
    
    ai_win_prob = 0.5
    if ai_agent:
        ai_win_prob = ai_agent.get_win_probability(game_env)
    
    return jsonify({
        'board': game_env.board.tolist(),
        'current_player': int(game_env.current_player),
        'player_color': int(player_color),
        'ai_color': int(ai_color),
        'game_over': bool(game_env.done),
        'winner': int(game_env.winner) if game_env.winner is not None else None,
        'ai_win_probability': float(ai_win_prob)
    })


@app.route('/api/undo', methods=['POST'])
def undo_move():
    """悔棋（撤销最近两步：玩家和AI各一步）"""
    global game_env
    
    if game_env is None or len(game_env.move_history) < 2:
        return jsonify({'error': '无法悔棋'}), 400
    
    # 撤销两步
    game_env.move_history.pop()  # AI的落子
    game_env.move_history.pop()  # 玩家的落子
    
    # 重建棋盘
    game_env.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    game_env.current_player = BLACK
    game_env.done = False
    game_env.winner = None
    
    for player, row, col in game_env.move_history:
        game_env.board[row][col] = player
        game_env.current_player = WHITE if player == BLACK else BLACK
    
    # 检查是否有人获胜
    if game_env.move_history:
        last_player, last_row, last_col = game_env.move_history[-1]
        if game_env.check_win(last_row, last_col):
            game_env.done = True
            game_env.winner = last_player
    
    ai_win_prob = 0.5
    if ai_agent:
        ai_win_prob = ai_agent.get_win_probability(game_env)
    
    return jsonify({
        'board': game_env.board.tolist(),
        'current_player': int(game_env.current_player),
        'player_color': int(player_color),
        'ai_color': int(ai_color),
        'game_over': bool(game_env.done),
        'winner': int(game_env.winner) if game_env.winner is not None else None,
        'ai_win_probability': float(ai_win_prob)
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    models = {'black': [], 'white': []}
    
    if os.path.exists(MODEL_DIR):
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('.pth'):
                if 'black' in filename:
                    models['black'].append(filename)
                elif 'white' in filename:
                    models['white'].append(filename)
    
    return jsonify(models)


if __name__ == '__main__':
    print("启动五子棋Web服务器...")
    print("请在浏览器中访问: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
