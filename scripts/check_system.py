import os
import psutil
import time
import shutil

print('=== 系统资源检查 ===')
print(f'当前时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')

# 检查内存
mem = psutil.virtual_memory()
print(f'总内存: {mem.total / 1024**3:.1f} GB')
print(f'可用内存: {mem.available / 1024**3:.1f} GB')
print(f'内存使用率: {mem.percent}%')

# 检查CPU
print(f'\n=== CPU 信息 ===')
print(f'CPU核心数: {psutil.cpu_count()}')
print(f'CPU使用率: {psutil.cpu_percent(interval=1)}%')

# 检查GPU内存（如果可用）
try:
    import torch
    if torch.cuda.is_available():
        print(f'\n=== GPU 信息 ===')
        print(f'GPU设备数: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
            print(f'  内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
            print(f'  已用内存: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB')
            print(f'  缓存内存: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB')
except Exception as e:
    print(f'\nGPU检查错误: {e}')

# 检查磁盘空间
total, used, free = shutil.disk_usage('D:/')
print(f'\n=== D盘空间 ===')
print(f'总空间: {total / 1024**3:.1f} GB')
print(f'已用空间: {used / 1024**3:.1f} GB')
print(f'可用空间: {free / 1024**3:.1f} GB')

# 检查进程限制
print(f'\n=== 进程限制 ===')
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    print(f'CPU时间限制: soft={soft}, hard={hard}')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f'虚拟内存限制: soft={soft}, hard={hard}')
except:
    print('无法获取进程限制（Windows可能不支持）')

# 检查是否有其他Python进程
print(f'\n=== Python进程检查 ===')
python_processes = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if 'python' in proc.info['name'].lower():
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'optuna_gigtransformer' in cmdline:
                python_processes.append(proc)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if python_processes:
    print(f'找到 {len(python_processes)} 个相关Python进程:')
    for proc in python_processes:
        print(f'  PID: {proc.info["pid"]}, 命令: {proc.info["cmdline"]}')
else:
    print('没有找到相关Python进程')