"""
检查CUDA是否可用
"""
import torch

print("=" * 50)
print("PyTorch CUDA 检查")
print("=" * 50)

print(f"\nPyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"\n当前设备: cuda")
else:
    print(f"\n当前设备: cpu")

print("=" * 50)
