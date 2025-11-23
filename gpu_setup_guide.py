# 🎮 GPU Setup Guide for RTX 2050
print('🎮 GPU SETUP GUIDE FOR RTX 2050')
print('=' * 50)

print("""
🔍 CURRENT STATUS ANALYSIS:
✅ NVIDIA RTX 2050 detected by system
✅ NVIDIA drivers installed (Version 577.03)
✅ CUDA 12.9 available
❌ TensorFlow NOT using GPU (CPU only training)

🚨 ISSUE IDENTIFIED:
You have Python 3.13, but TensorFlow GPU versions require Python 3.9-3.11

💡 SOLUTION OPTIONS:

OPTION 1: Use Python 3.11 (RECOMMENDED)
1. Install Python 3.11 from python.org
2. Create virtual environment with Python 3.11
3. Install TensorFlow GPU version

OPTION 2: Use PyTorch (Alternative GPU Framework)
1. Install PyTorch with CUDA support
2. Modify the defense system to use PyTorch

OPTION 3: Continue with CPU (Current setup)
1. Keep current Python 3.13 + TensorFlow 2.20
2. Accept slower CPU training

🔧 DETAILED SETUP FOR OPTION 1 (RECOMMENDED):

Step 1: Install Python 3.11
- Download from https://www.python.org/downloads/release/python-3119/
- Install and add to PATH

Step 2: Create Virtual Environment
python -m venv defense_gpu_env
defense_gpu_env\\Scripts\\activate

Step 3: Install TensorFlow GPU
pip install tensorflow==2.13.0

Step 4: Verify GPU
python -c "import tensorflow as tf; print('GPU:', len(tf.config.list_physical_devices('GPU')) > 0)"

🔧 DETAILED SETUP FOR OPTION 2 (PyTorch):

Step 1: Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Step 2: Verify GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

⚡ PERFORMANCE COMPARISON:
- CPU Training: ~25 seconds for signal model
- GPU Training: ~3-5 seconds for signal model (5-8x faster)

🎯 RECOMMENDATION:
Use Option 1 (Python 3.11 + TensorFlow GPU) for best performance with your RTX 2050
""")

# Check current setup
import tensorflow as tf
import sys

print(f'\n📊 CURRENT SETUP DETAILS:')
print(f'Python Version: {sys.version}')
print(f'TensorFlow Version: {tf.__version__}')
print(f'GPU Available: {len(tf.config.list_physical_devices("GPU")) > 0}')

# Check PyTorch option
try:
    import torch
    print(f'PyTorch Available: {torch.__version__}')
    print(f'PyTorch GPU Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'PyTorch GPU Device: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch Not Installed')

print(f'\n🎯 NEXT STEPS:')
print(f'1. Choose setup option (1, 2, or 3)')
print(f'2. Follow the installation steps')
print(f'3. Run GPU check again')
print(f'4. Enjoy 5-8x faster training with RTX 2050!')
