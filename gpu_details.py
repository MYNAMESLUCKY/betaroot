# 🎮 Detailed GPU Analysis with PyTorch
import torch
import subprocess
import sys
import os
import json

print('🎮 DETAILED GPU ANALYSIS - PYTORCH')
print('=' * 50)

def get_gpu_details():
    """Get comprehensive GPU information"""
    
    print('\n🔥 PYTORCH GPU INFORMATION:')
    print('-' * 40)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print('✅ CUDA Available: YES')
        print(f'🔥 CUDA Version: {torch.version.cuda}')
        print(f'🧠 cuDNN Version: {torch.backends.cudnn.version()}')
        print(f'📊 GPU Count: {torch.cuda.device_count()}')
        
        # Get details for each GPU
        for i in range(torch.cuda.device_count()):
            print(f'\n🎯 GPU {i} Details:')
            print(f'   Name: {torch.cuda.get_device_name(i)}')
            
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            print(f'   Compute Capability: {props.major}.{props.minor}')
            print(f'   Total Memory: {props.total_memory / 1024**3:.2f} GB')
            print(f'   Multiprocessors: {props.multi_processor_count}')
            
            # Memory info
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f'   Memory Allocated: {memory_allocated:.2f} MB')
            print(f'   Memory Reserved: {memory_reserved:.2f} MB')
            
            # Current memory usage
            free_memory = (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3
            used_memory = torch.cuda.memory_allocated(i) / 1024**3
            print(f'   Memory Usage: {used_memory:.2f} GB / {props.total_memory / 1024**3:.2f} GB ({used_memory/(props.total_memory/1024**3)*100:.1f}%)')
            print(f'   Free Memory: {free_memory:.2f} GB')
    
    else:
        print('❌ CUDA Available: NO')
        print('💡 GPU acceleration not available')
    
    # Backend information
    print(f'\n🔧 BACKEND INFORMATION:')
    print('-' * 30)
    print(f'PyTorch Version: {torch.__version__}')
    print(f'PyTorch Built With: {torch.__config__.show() if hasattr(torch.__config__, 'show') else 'N/A'}')
    
    # cuDNN settings
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        print(f'cuDNN Enabled: YES')
        print(f'cuDNN Version: {torch.backends.cudnn.version()}')
        print(f'cuDNN Benchmark: {torch.backends.cudnn.benchmark}')
        print(f'cuDNN Deterministic: {torch.backends.cudnn.deterministic}')
    else:
        print('cuDNN Enabled: NO')

def get_nvidia_smi_details():
    """Get detailed NVIDIA GPU information using nvidia-smi"""
    
    print(f'\n🖥️  NVIDIA-SMI DETAILS:')
    print('-' * 30)
    
    try:
        # Run nvidia-smi command
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 8:
                        print(f'\n🎯 GPU {i}:')
                        print(f'   Name: {parts[0]}')
                        print(f'   Driver Version: {parts[1]}')
                        print(f'   Memory Total: {parts[2]} MB')
                        print(f'   Memory Used: {parts[3]} MB')
                        print(f'   Memory Free: {parts[4]} MB')
                        print(f'   GPU Utilization: {parts[5]}%')
                        print(f'   Temperature: {parts[6]}°C')
                        print(f'   Power Draw: {parts[7]} W')
        else:
            print('❌ Could not get nvidia-smi details')
    
    except Exception as e:
        print(f'❌ Error getting nvidia-smi details: {e}')

def get_gpu_benchmark():
    """Run a simple GPU benchmark"""
    
    print(f'\n⚡ GPU PERFORMANCE BENCHMARK:')
    print('-' * 35)
    
    if not torch.cuda.is_available():
        print('❌ No GPU available for benchmark')
        return
    
    try:
        import time
        import numpy as np
        
        device = torch.device('cuda')
        
        # Matrix multiplication benchmark
        print('🚀 Running Matrix Multiplication Benchmark...')
        
        sizes = [512, 1024, 2048, 4096]
        
        for size in sizes:
            # Create random matrices
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warm up
            for _ in range(10):
                _ = torch.mm(a, b)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(100):
                c = torch.mm(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            gflops = (2 * size**3) / (avg_time * 1e9)
            
            print(f'   {size}x{size}: {avg_time*1000:.2f}ms ({gflops:.1f} GFLOPS)')
        
        # Memory bandwidth test
        print('\n💾 Memory Bandwidth Test...')
        
        size = 1024 * 1024 * 100  # 100M elements
        a = torch.randn(size, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            b = a * 2.0
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        bandwidth = (size * 4 * 2) / (avg_time * 1e9)  # 4 bytes per float, read+write
        
        print(f'   Bandwidth: {bandwidth:.1f} GB/s')
        
    except Exception as e:
        print(f'❌ Benchmark failed: {e}')

def check_gpu_compatibility():
    """Check GPU compatibility with PyTorch features"""
    
    print(f'\n🔍 GPU COMPATIBILITY CHECK:')
    print('-' * 35)
    
    if not torch.cuda.is_available():
        print('❌ No GPU available')
        return
    
    device = torch.device('cuda')
    
    # Test different data types
    print('📊 Data Type Support:')
    
    data_types = [
        ('Float32', torch.float32),
        ('Float16', torch.float16),
        ('BFloat16', torch.bfloat16),
        ('Int32', torch.int32),
        ('Int16', torch.int16),
        ('Int8', torch.int8),
    ]
    
    for name, dtype in data_types:
        try:
            tensor = torch.randn(100, 100, dtype=dtype, device=device)
            _ = tensor @ tensor.T
            print(f'   ✅ {name}')
        except Exception as e:
            print(f'   ❌ {name}: {str(e)[:50]}...')
    
    # Check mixed precision
    print('\n🎯 Mixed Precision Support:')
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        model = torch.nn.Linear(100, 100).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()
        
        x = torch.randn(32, 100, device=device)
        
        with autocast():
            y = model(x)
            loss = y.sum()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print('   ✅ Mixed Precision: Supported')
    except Exception as e:
        print(f'   ❌ Mixed Precision: {str(e)[:50]}...')
    
    # Check advanced features
    print('\n🚀 Advanced Features:')
    
    features = [
        ('Tensor Cores', lambda: torch.cuda.get_device_capability()[0] >= 7),
        ('CUDA Graphs', lambda: hasattr(torch.cuda, 'graph') and torch.cuda.is_available()),
        ('Flash Attention', lambda: hasattr(torch.nn.functional, 'scaled_dot_product_attention')),
    ]
    
    for name, check in features:
        try:
            if check():
                print(f'   ✅ {name}')
            else:
                print(f'   ❌ {name}: Not supported')
        except Exception as e:
            print(f'   ❌ {name}: {str(e)[:50]}...')

def main():
    """Main function"""
    print('🎮 COMPREHENSIVE GPU ANALYSIS')
    print('=' * 40)
    print('🔥 PyTorch + NVIDIA RTX 2050')
    
    # Get all GPU details
    get_gpu_details()
    get_nvidia_smi_details()
    get_gpu_benchmark()
    check_gpu_compatibility()
    
    # Summary
    print(f'\n🎉 GPU ANALYSIS SUMMARY:')
    print('=' * 30)
    
    if torch.cuda.is_available():
        print(f'✅ GPU Available: YES')
        print(f'🎯 GPU Device: {torch.cuda.get_device_name(0)}')
        print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        print(f'🔥 CUDA Version: {torch.version.cuda}')
        print(f'🚀 Acceleration: READY')
        print(f'⚡ Performance: OPTIMIZED')
    else:
        print(f'❌ GPU Available: NO')
        print(f'🐌 Acceleration: DISABLED')

if __name__ == '__main__':
    main()
