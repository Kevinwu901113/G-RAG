import torch

def test_gpu():
    # 检查CUDA是否可用
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 获取当前GPU设备
        current_device = torch.cuda.current_device()
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"当前GPU设备: {current_device}")
        print(f"GPU数量: {gpu_count}")
        print(f"GPU名称: {gpu_name}")
        
        # 测试GPU计算
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("\nGPU计算测试成功！")
        print(f"计算结果: {z}")
    else:
        print("CUDA不可用，请检查您的GPU驱动和CUDA安装。")

if __name__ == "__main__":
    test_gpu() 