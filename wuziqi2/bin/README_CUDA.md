# CUDA 兼容性解决方案

## 问题描述

你的显卡 **NVIDIA GeForce RTX 5060 Laptop GPU** 使用 CUDA capability sm_120 架构。

当前安装的 PyTorch 2.6.0+cu124 不支持 sm_120，需要 CUDA 12.8 或更高版本。

## 解决方案

### 方案1：使用CPU版本（立即可用）

如果不需要GPU加速，可以直接使用CPU版本：

```bash
# 修改 model.py 中的设备设置
# 将 device='cuda' if torch.cuda.is_available() else 'cpu'
# 改为 device='cpu'
```

**缺点**：训练速度较慢

---

### 方案2：安装 PyTorch 2.7+ with CUDA 12.8（推荐）

RTX 50系列显卡需要 CUDA 12.8 或更高版本。

#### 步骤1：卸载现有PyTorch
```bash
pip uninstall torch torchvision torchaudio -y
```

#### 步骤2：安装CUDA 12.8版本
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 注意：下载约2.8GB，需要耐心等待

#### 步骤3：验证安装
```bash
python check_cuda.py
```

---

### 方案3：使用Nightly版本

如果稳定版仍不支持，可以尝试nightly版本：

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 当前状态

项目已配置为自动检测CUDA：
- 如果CUDA可用 → 使用GPU
- 如果CUDA不可用 → 自动回退到CPU

代码已恢复GPU支持，安装正确的PyTorch版本后即可使用GPU加速。

## 推荐操作

1. **短期**：使用CPU版本进行代码测试
2. **长期**：下载并安装 PyTorch cu128 版本启用GPU加速

## 参考资料

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA 兼容性](https://developer.nvidia.com/cuda-gpus)
