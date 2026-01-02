import torch

def verify_setup():
    print("--- 实验环境检查 ---")
    # 检查 PyTorch 是否能看到 GPU
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch 是否可用 GPU: {cuda_available}")
    
    if cuda_available:
        print(f"当前 GPU 设备: {torch.cuda.get_device_name(0)}")
        # 尝试创建一个张量并搬移到 GPU
        x = torch.rand(5, 3).cuda()
        print("成功在 GPU 上创建 Tensor!")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("警告: 无法检测到 GPU，请检查 PyTorch 安装版本是否匹配。")

if __name__ == "__main__":
    verify_setup()