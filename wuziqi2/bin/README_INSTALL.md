# PyTorch 安装故障排除

## 当前错误

```
OSError: [WinError 5] 拒绝访问。
'D:\anaconda\Lib\site-packages\torch\lib\nvrtc64_120_0.dll'
```

**原因**：文件被占用或权限不足

---

## 解决方案

### 方法1：使用 `--user` 参数（推荐）

以普通用户身份安装到用户目录：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --user
```

---

### 方法2：以管理员身份运行

1. 右键点击 `install_torch_admin.bat`
2. 选择"以管理员身份运行"
3. 等待安装完成

---

### 方法3：手动安装步骤

#### 步骤1：关闭所有Python程序
- 关闭所有命令行窗口
- 关闭IDE（VS Code/PyCharm等）
- 关闭Jupyter Notebook

#### 步骤2：打开管理员命令提示符
```
Win + X → Windows终端(管理员) 或 命令提示符(管理员)
```

#### 步骤3：执行安装
```bash
cd D:\新建文件夹\wuziqi
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

### 方法4：使用conda安装（如果使用Anaconda）

```bash
conda uninstall pytorch torchvision torchaudio -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
```

---

## 验证安装

安装完成后运行：

```bash
python check_cuda.py
```

成功输出应显示：
```
PyTorch 版本: 2.7.0+cu128 或更高
CUDA 是否可用: True
当前设备: cuda
```

---

## 备选方案：继续使用CPU版本

如果GPU安装困难，可以暂时使用CPU版本：

```bash
# 安装CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 修改代码强制使用CPU（已自动处理）
# model.py 会自动检测CUDA是否可用
```

**注意**：CPU版本训练速度较慢，但功能完整。

---

## 推荐操作

1. **立即**：使用管理员命令提示符执行方法3
2. **如果失败**：使用方法1的 `--user` 参数
3. **如果仍失败**：暂时使用CPU版本，后续再解决GPU问题
