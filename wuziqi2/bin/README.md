# 五子棋AI对抗训练项目 (Gobang AI)

## 项目概述

本项目是一个基于GAN（生成对抗网络）模式的五子棋AI系统，包含两个独立的AI内核（Black和White）。通过强化学习的交替对抗训练，使两个内核的五子棋能力不断提升，最终达到高水平对弈能力。

**项目特点：**
- 采用深度残差神经网络（ResNet）作为AI决策核心
- GAN式交替训练机制，两个AI互相促进提升
- 完整的可视化分析工具，监控训练过程
- 美观的Web交互界面，支持人机对弈

---

## 技术架构

### 核心组件

| 文件 | 功能描述 |
|------|----------|
| `config.py` | 全局配置参数（棋盘大小、训练参数等） |
| `gobang_env.py` | 五子棋游戏环境，实现游戏规则和状态管理 |
| `model.py` | AI神经网络模型（ResNet + 策略/价值双头输出） |
| `train.py` | 训练管理器，实现GAN式交替训练流程 |
| `visualizer.py` | 可视化工具，生成7种训练分析图表 |
| `web_server.py` | Flask Web服务器，提供人机对弈API |
| `templates/index.html` | 响应式Web界面，支持19×19棋盘交互 |

### 技术栈

- **深度学习**：PyTorch（ResNet架构）
- **后端**：Flask + Flask-CORS
- **前端**：原生HTML5/CSS3/JavaScript
- **可视化**：Matplotlib
- **数据处理**：NumPy

---

## 项目结构

```
wuziqi/
├── config.py              # 配置文件
├── gobang_env.py          # 游戏环境
├── model.py               # AI模型
├── train.py               # 训练脚本
├── visualizer.py          # 可视化工具
├── web_server.py          # Web服务器
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
│
├── templates/
│   └── index.html         # Web界面
│
├── models/                # 模型文件目录（自动创建）
│   ├── black_latest.pth   # Black内核最新模型
│   ├── white_latest.pth   # White内核最新模型
│   ├── black_cycle_XX.pth # 各Cycle的Black模型
│   └── white_cycle_XX.pth # 各Cycle的White模型
│
├── logs/                  # 对局日志目录（自动创建）
│   ├── training_stats.json    # 训练统计数据
│   └── cycle_XX_phase_game_XXXXX.txt  # 单局对局记录
│
└── visualizations/        # 可视化图表目录（自动创建）
    ├── 01_segmented_win_rates.png
    ├── 02_win_rate_trend.png
    ├── 03_game_length.png
    ├── 04_opening_heatmap.png
    ├── 05_training_loss.png
    ├── 06_elo_rating.png
    └── 07_cycle_summary.png
```

---

## 安装与配置

### 环境要求

- Python 3.8+
- CUDA（可选，用于GPU加速训练）

### 安装步骤

```bash
# 进入项目目录
cd D:\新建文件夹\wuziqi

# 安装依赖
pip install -r requirements.txt
```

### 依赖列表

```
torch>=2.0.0          # 深度学习框架
numpy>=1.24.0         # 数值计算
flask>=2.3.0          # Web服务器
flask-cors>=4.0.0     # 跨域支持
matplotlib>=3.7.0     # 数据可视化
tqdm>=4.65.0          # 进度条显示
```

---

## 使用指南

### 1. 训练AI模型

```bash
python train.py
```

**训练流程：**
1. 初始化Black和White两个AI代理
2. 每个Cycle包含两个阶段：
   - 阶段1：固定Black，训练White进行T局对弈
   - 阶段2：固定White，训练Black进行T局对弈
3. 重复CYCLE个Cycle
4. 自动保存模型和训练日志

**训练参数配置（config.py）：**

```python
T = 2000              # 每个内核每轮训练对局数
CYCLE = 40            # 总训练轮数
SEGMENT_SIZE = 50     # 胜率统计分段大小
BOARD_SIZE = 19       # 棋盘大小（19×19）
LEARNING_RATE = 0.001 # 学习率
```

**训练输出：**
- 模型文件：`models/black_latest.pth`, `models/white_latest.pth`
- 对局日志：`logs/cycle_XX_phase_game_XXXXX.txt`
- 训练统计：`logs/training_stats.json`

### 2. 生成可视化图表

```bash
python visualizer.py
```

生成的7种图表：

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 分段胜率直方图 | `01_segmented_win_rates.png` | 上下排列，上White下Black，每50局分段统计 |
| 胜率变化折线图 | `02_win_rate_trend.png` | 带训练阶段背景色（绿=训练White，橙=训练Black） |
| 平均对局步数 | `03_game_length.png` | 反映AI是否学会快速取胜 |
| 开局多样性热力图 | `04_opening_heatmap.png` | 显示AI探索的开局位置分布 |
| 训练损失曲线 | `05_training_loss.png` | 监控训练稳定性（总损失/策略损失/价值损失） |
| ELO评分变化 | `06_elo_rating.png` | 量化整体实力提升 |
| Cycle总结对比 | `07_cycle_summary.png` | 每个Cycle两个训练阶段的胜负分布 |

### 3. 启动Web对弈界面

```bash
python web_server.py
```

访问地址：`http://localhost:5000`

**功能说明：**
- 选择执黑或执白
- 实时显示AI胜率评估（模型自身的胜率估计）
- 支持悔棋功能
- 显示最后落子位置（红点标记）
- 游戏结束提示

---

## 游戏规则

### 基本规则

1. **棋盘**：19×19网格，共361个交叉点可落子
2. **先手**：黑方（Black）先行
3. **落子**：双方轮流在交叉点放置棋子
4. **获胜条件**：任意一方在横、竖、斜方向形成连续5个同色棋子
5. **平局**：棋盘填满无空位时判定为平局

### 判定逻辑

```python
# 四个方向检查
- 水平方向：左右检查
- 垂直方向：上下检查
- 对角线方向：左上-右下
- 反对角线方向：右上-左下

# 获胜条件
连续5个同色棋子 → 该方获胜
```

---

## 训练机制详解

### GAN式交替训练

```
Cycle 1:
  ├─ 阶段1: 固定Black → 训练White (T=2000局)
  └─ 阶段2: 固定White → 训练Black (T=2000局)

Cycle 2:
  ├─ 阶段1: 固定Black → 训练White (T=2000局)
  └─ 阶段2: 固定White → 训练Black (T=2000局)

...重复CYCLE次
```

### 强化学习算法

1. **策略网络（Policy Network）**：输出每个位置的落子概率
2. **价值网络（Value Network）**：评估当前局面的胜率 [-1, 1]
3. **损失函数**：
   - 策略损失：交叉熵损失（实际落子 vs 预测概率）
   - 价值损失：MSE损失（实际结果 vs 预测价值）
4. **探索策略**：Epsilon-贪婪（随训练逐渐降低探索率）

### 神经网络架构

```
输入：[2, 19, 19]  (当前玩家棋子平面 + 对手棋子平面)
  ↓
卷积层 + 4个残差块
  ↓
├─→ 策略头 → [361] 落子概率
└─→ 价值头 → [1] 胜率评估
```

---

## 日志格式

### 单局对局日志

文件：`logs/cycle_XX_phase_game_XXXXX.txt`

```
Winner: Black
B(9,9),W(9,10),B(10,9),W(10,10),B(11,9),W(11,10),B(12,9)
```

格式说明：
- `B(row,col)`：Black落子位置
- `W(row,col)`：White落子位置
- 坐标范围：0-18（对应19×19棋盘）

### 训练统计日志

文件：`logs/training_stats.json`

包含：
- 每个Cycle的训练结果
- 分段胜率统计
- 损失值记录
- 训练阶段标记

---

## 可视化图表说明

### 1. 分段胜率直方图

**特点：**
- 上下排列：上半White胜率，下半Black胜率
- 颜色标记：绿色=训练White阶段，橙色=训练Black阶段
- 分段统计：每50局为一个统计单元

**解读：**
- 被训练方的胜率应逐渐提升
- 固定方的胜率可能下降（对手变强）

### 2. 胜率变化折线图

**特点：**
- 横轴：对局序号
- 纵轴：胜率（滑动窗口平均）
- 背景色：区分训练阶段

### 3. 其他图表

- **平均对局步数**：步数减少说明AI学会快速取胜
- **开局热力图**：显示AI偏好的开局位置
- **ELO评分**：量化AI整体实力变化

---

## 开发信息

### 作者
开发日期：2026-07-29

### 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-07-29 | 初始版本，完整实现GAN训练、可视化和Web对弈 |

### 待优化项

- [ ] 实现更高效的MCTS（蒙特卡洛树搜索）
- [ ] 添加模型对战回放功能
- [ ] 支持加载指定Cycle的模型进行对弈
- [ ] 优化神经网络架构（尝试Transformer）
- [ ] 添加分布式训练支持

---

## 许可证

本项目仅供学习和研究使用。

---

## 联系方式

如有问题或建议，欢迎反馈。
