@echo off
chcp 65001
echo ==========================================
echo 安装支持 RTX 50系列的 PyTorch
echo ==========================================

echo [1/3] 卸载现有 PyTorch...
pip uninstall torch torchvision torchaudio -y

echo [2/3] 安装 PyTorch 2.7.0 (支持 CUDA 12.8)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo [3/3] 安装完成，验证中...
python check_cuda.py

echo ==========================================
echo 安装完成！
pause
