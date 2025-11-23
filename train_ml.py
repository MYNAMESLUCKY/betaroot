# Quick Training Script for Defense Intelligence ML
import subprocess
import sys
import os

print('🚀 Defense Intelligence ML - Quick Training')
print('=' * 50)

# Check if Python and required packages are available
def check_requirements():
    print('🔍 Checking requirements...')
    
    try:
        import tensorflow as tf
        print(f'✅ TensorFlow {tf.__version()}')
    except ImportError:
        print('❌ TensorFlow not installed')
        return False
    
    try:
        import numpy as np
        print(f'✅ NumPy {np.__version__}')
    except ImportError:
        print('❌ NumPy not installed')
        return False
    
    try:
        import cv2
        print(f'✅ OpenCV {cv2.__version__}')
    except ImportError:
        print('❌ OpenCV not installed')
        return False
    
    try:
        import sklearn
        print(f'✅ Scikit-learn {sklearn.__version__}')
    except ImportError:
        print('❌ Scikit-learn not installed')
        return False
    
    # Check GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f'✅ GPU detected: {len(gpus)} device(s)')
        else:
            print('⚠️  No GPU detected - using CPU (slower training)')
    except:
        print('⚠️  Could not check GPU availability')
    
    return True

def install_requirements():
    print('\n📦 Installing requirements...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print('✅ Requirements installed successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ Failed to install requirements: {e}')
        return False

def run_training():
    print('\n🎯 Starting ML training...')
    try:
        subprocess.check_call([sys.executable, 'defense_intelligence_ml.py'])
        print('✅ Training completed successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ Training failed: {e}')
        return False

def main():
    print('🛡️ Defense Intelligence ML Quick Start')
    print(f'📍 Working directory: {os.getcwd()}')
    
    # Check requirements
    if not check_requirements():
        print('\n📦 Installing missing packages...')
        if not install_requirements():
            print('❌ Failed to install requirements')
            return False
        
        # Check again after installation
        if not check_requirements():
            print('❌ Requirements still not met')
            return False
    
    # Run training
    print('\n🚀 Ready to start training!')
    input('Press Enter to begin training (or Ctrl+C to cancel)...')
    
    success = run_training()
    
    if success:
        print('\n🎉 Training completed successfully!')
        print('📁 Check the "models/" directory for trained models')
        print('📊 Training report saved to "models/training_report.json"')
    else:
        print('\n❌ Training failed. Check the error messages above.')
    
    return success

if __name__ == '__main__':
    main()
