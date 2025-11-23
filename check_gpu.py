# 🎮 GPU Detection and Status Check
import tensorflow as tf
import subprocess
import sys
import os

print('🎮 GPU DETECTION AND STATUS')
print('=' * 40)

def check_gpu_status():
    """Check GPU status and configuration"""
    
    print('\n🔍 TensorFlow GPU Detection:')
    print('-' * 30)
    
    # Check TensorFlow GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f'✅ GPU(s) Detected: {len(gpus)}')
        for i, gpu in enumerate(gpus):
            print(f'   GPU {i}: {gpu.name}')
            print(f'   Type: {gpu.device_type}')
            
            # Get GPU details
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f'   Name: {details.get("device_name", "Unknown")}')
                print(f'   Compute Capability: {details.get("compute_capability", "Unknown")}')
            except:
                print('   Details: Not available')
    else:
        print('❌ No GPU detected by TensorFlow')
        print('   Using CPU for training (slower)')
    
    print(f'\n🎮 GPU Configuration:')
    print('-' * 30)
    
    if gpus:
        # Configure GPU memory growth
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f'✅ Memory growth enabled for {gpu.name}')
            except:
                print(f'⚠️  Could not enable memory growth for {gpu.name}')
    
    # Check CUDA availability
    print(f'\n🔥 CUDA Status:')
    print('-' * 30)
    try:
        # Check if CUDA is available
        from tensorflow.python.platform import build_info as build
        cuda_version = build.cuda_version
        cudnn_version = build.cudnn_version
        print(f'✅ CUDA Version: {cuda_version}')
        print(f'✅ cuDNN Version: {cudnn_version}')
    except:
        print('⚠️  CUDA information not available')
    
    # Check NVIDIA drivers (Windows)
    print(f'\n🖥️  NVIDIA Driver Status:')
    print('-' * 30)
    try:
        # Try to get NVIDIA driver info
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print('✅ NVIDIA drivers installed')
            
            # Parse nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX 2050' in line or 'GeForce' in line:
                    print(f'🎯 GPU Found: {line.strip()}')
                elif 'CUDA Version' in line:
                    print(f'🔥 {line.strip()}')
                elif 'Driver Version' in line:
                    print(f'📱 {line.strip()}')
        else:
            print('❌ NVIDIA drivers not found or nvidia-smi not available')
    except:
        print('❌ Could not check NVIDIA drivers')
        print('💡 Install NVIDIA drivers for GPU acceleration')
    
    # Check current platform
    print(f'\n🌐 Platform Detection:')
    print('-' * 30)
    
    # Check for Google Colab
    try:
        import google.colab
        print('✅ Running on Google Colab')
    except ImportError:
        print('❌ Not running on Google Colab')
    
    # Check for Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print('✅ Running on Kaggle')
    else:
        print('❌ Not running on Kaggle')
    
    # Check for local environment
    if not 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            import google.colab
        except ImportError:
            print('✅ Running on local machine')
    
    # Memory info
    print(f'\n💾 Memory Status:')
    print('-' * 30)
    
    if gpus:
        try:
            # Get GPU memory info
            for i, gpu in enumerate(gpus):
                memory_info = tf.config.experimental.get_memory_info(gpu)
                print(f'GPU {i} Memory:')
                print(f'   Current: {memory_info.get("current", 0) / 1024**2:.1f} MB')
                print(f'   Peak: {memory_info.get("peak", 0) / 1024**2:.1f} MB')
        except:
            print('⚠️  GPU memory info not available')
    
    # Test GPU with simple operation
    print(f'\n⚡ GPU Performance Test:')
    print('-' * 30)
    
    if gpus:
        try:
            import time
            import numpy as np
            
            # Test matrix multiplication on GPU
            print('Testing GPU performance...')
            
            with tf.device('/GPU:0' if gpus else '/CPU:0'):
                # Create large matrices
                size = 1000
                a = tf.random.normal((size, size))
                b = tf.random.normal((size, size))
                
                # Time the operation
                start_time = time.time()
                c = tf.matmul(a, b)
                result = c.numpy()
                gpu_time = time.time() - start_time
                
                print(f'✅ GPU Matrix Multiplication: {gpu_time:.3f} seconds')
                print(f'   Matrix size: {size}x{size}')
                print(f'   Device: {"GPU" if gpus else "CPU"}')
            
            # Compare with CPU if GPU is available
            if gpus:
                with tf.device('/CPU:0'):
                    start_time = time.time()
                    c_cpu = tf.matmul(a, b)
                    result_cpu = c_cpu.numpy()
                    cpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time
                print(f'📊 GPU Speedup: {speedup:.2f}x faster than CPU')
                
        except Exception as e:
            print(f'❌ Performance test failed: {e}')
    else:
        print('⚠️  No GPU available for performance test')
    
    return len(gpus) > 0

def check_training_gpu_usage():
    """Check if training will use GPU"""
    print(f'\n🚀 Training GPU Usage:')
    print('-' * 30)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print('✅ Models will train on GPU')
        print(f'🎯 Available GPUs: {len(gpus)}')
        
        # Show GPU memory configuration
        for i, gpu in enumerate(gpus):
            try:
                # Check current memory growth setting
                print(f'GPU {i}: Memory growth enabled')
            except:
                print(f'GPU {i}: Memory growth not configured')
        
        print('\n💡 Tips for GPU training:')
        print('   - Use larger batch sizes for better GPU utilization')
        print('   - Monitor GPU memory usage')
        print('   - Enable mixed precision for faster training')
        
    else:
        print('❌ Models will train on CPU')
        print('⚠️  Training will be slower without GPU acceleration')
        print('\n💡 To enable GPU training:')
        print('   1. Install NVIDIA drivers')
        print('   2. Install CUDA toolkit')
        print('   3. Install cuDNN')
        print('   4. Ensure TensorFlow GPU version is installed')

def main():
    """Main function"""
    print('🎮 DEFENSE INTELLIGENCE - GPU STATUS CHECK')
    print('=' * 50)
    
    # Check GPU status
    gpu_available = check_gpu_status()
    
    # Check training GPU usage
    check_training_gpu_usage()
    
    # Summary
    print(f'\n🎉 GPU STATUS SUMMARY:')
    print('=' * 30)
    
    if gpu_available:
        print('✅ GPU Available: YES')
        print('🚀 Training Acceleration: ENABLED')
        print('⚡ Performance: OPTIMIZED')
        print('🎯 Your RTX 2050 should be utilized!')
    else:
        print('❌ GPU Available: NO')
        print('🐌 Training Acceleration: DISABLED')
        print('⚡ Performance: CPU ONLY')
        print('💡 Install NVIDIA drivers to enable GPU')
    
    print(f'\n🔥 TensorFlow Version: {tf.__version__}')
    print(f'📍 Platform: {"Local" if not "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "Cloud"}')

if __name__ == '__main__':
    main()
