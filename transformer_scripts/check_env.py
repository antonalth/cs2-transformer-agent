import torch
import tensorrt
import sys

print('--- Verification ---')
print(f'Python Version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'TensorRT version: {tensorrt.__version__}')
print(f'CUDA available for PyTorch: {torch.cuda.is_available()}')
print('--------------------')

print('\\n--- Detailed GPU Information ---')
if not torch.cuda.is_available():
    print('CUDA is not available. No GPUs were found by PyTorch.')
else:
    device_count = torch.cuda.device_count()
    print(f'Found {device_count} CUDA-enabled GPU(s).')
    for i in range(device_count):
        print(f'\\n--- GPU {i} ---')
        print(f'  Name:          {torch.cuda.get_device_name(i)}')
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f'  Total Memory:  {total_mem:.2f} GB')
        major, minor = torch.cuda.get_device_capability(i)
        print(f'  Compute Major: {major}')
        print(f'  Compute Minor: {minor}')
print('--------------------------------')