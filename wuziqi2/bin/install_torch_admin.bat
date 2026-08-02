@echo off
chcp 65001
echo ==========================================
echo 以管理员身份安装 PyTorch CUDA 12.8
echo ==========================================

echo [1/3] 关闭所有使用PyTorch的程序...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] 卸载现有 PyTorch...
pip uninstall torch torchvision torchaudio -y

echo [3/3] 安装 PyTorch 2.7+ with CUDA 12.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --user

echo ==========================================
echo 安装完成！
echo 请运行 check_cuda.py 验证安装
echo ==========================================
pause
