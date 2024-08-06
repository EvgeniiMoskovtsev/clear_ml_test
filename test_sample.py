import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA доступна: {cuda_available}")

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print(f"Используемый GPU: {gpu_name}")
    print(f"Версия CUDA: {cuda_version}")
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    c = a + b
    print(f"Результат a + b на GPU: {c}")
else:
    print("CUDA не доступна")
