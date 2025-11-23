# 🚀 Enhanced Defense Intelligence Setup Script
import subprocess
import sys
import os
import json

print('🚀 ENHANCED DEFENSE INTELLIGENCE SETUP')
print('=' * 50)
print('🎮 Multi-platform GPU + Database Integration')

def install_requirements():
    """Install enhanced requirements"""
    print('\n📦 Installing enhanced requirements...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'enhanced_requirements.txt'])
        print('✅ Enhanced requirements installed successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ Failed to install requirements: {e}')
        return False

def setup_kaggle_api():
    """Setup Kaggle API for dataset downloads"""
    print('\n📥 Setting up Kaggle API...')
    
    # Check if kaggle.json exists
    kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.path.exists(kaggle_path):
        print('✅ Kaggle API already configured')
        return True
    
    print('📝 To setup Kaggle API:')
    print('1. Go to https://www.kaggle.com/account')
    print('2. Click "Create New API Token"')
    print('3. Download kaggle.json')
    print('4. Place kaggle.json in ~/.kaggle/ directory')
    
    # Create .kaggle directory
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    
    return False

def setup_mongodb_atlas():
    """Setup MongoDB Atlas configuration"""
    print('\n💾 Setting up MongoDB Atlas...')
    
    config_file = 'config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if config.get('mongodb', {}).get('atlas_uri'):
            print('✅ MongoDB Atlas already configured')
            return True
    
    print('📝 To setup MongoDB Atlas:')
    print('1. Go to https://www.mongodb.com/cloud/atlas')
    print('2. Create a free cluster')
    print('3. Get your connection string')
    print('4. Update config.json with your credentials')
    
    return False

def setup_mysql():
    """Setup MySQL database"""
    print('\n🐬 Setting up MySQL...')
    
    try:
        import mysql.connector
        # Try to connect to local MySQL
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )
        conn.close()
        print('✅ MySQL is available locally')
        return True
    except:
        print('📝 To setup MySQL:')
        print('1. Install MySQL Server')
        print('2. Create database: defense_db')
        print('3. Create user: defense_user')
        print('4. Update config.json with your credentials')
        return False

def check_gpu_availability():
    """Check GPU availability on different platforms"""
    print('\n🎮 Checking GPU availability...')
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f'✅ Local GPU detected: {len(gpus)} device(s)')
            for i, gpu in enumerate(gpus):
                print(f'   GPU {i}: {gpu.name}')
        else:
            print('⚠️  No local GPU detected')
        
        # Check for Colab
        try:
            import google.colab
            print('✅ Running on Google Colab')
            if gpus:
                print('✅ Colab GPU available')
        except ImportError:
            pass
        
        # Check for Kaggle
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            print('✅ Running on Kaggle')
            if gpus:
                print('✅ Kaggle GPU available')
        
        return len(gpus) > 0
        
    except ImportError:
        print('❌ TensorFlow not available')
        return False

def create_directories():
    """Create necessary directories"""
    print('\n📁 Creating directories...')
    
    directories = [
        'models',
        'data',
        'data/satellite',
        'data/signals',
        'data/threats',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f'✅ Created: {directory}')

def download_sample_datasets():
    """Download sample datasets for testing"""
    print('\n📥 Downloading sample datasets...')
    
    try:
        import kaggle
        
        # Download a small sample dataset
        datasets = [
            'andrewmvd/satellite-image-classification',
            'ucimae/signal-processing-datasets'
        ]
        
        for dataset in datasets:
            try:
                print(f'📥 Downloading {dataset}...')
                kaggle.api.dataset_download_files(dataset, path='data', unzip=True)
                print(f'✅ Downloaded {dataset}')
            except Exception as e:
                print(f'❌ Failed to download {dataset}: {e}')
        
        return True
        
    except ImportError:
        print('⚠️  Kaggle API not available. Install with: pip install kaggle')
        return False

def main():
    """Main setup function"""
    print('🛡️ Enhanced Defense Intelligence Setup')
    print('🎮 RTX 2050 + Kaggle + Google Colab + MongoDB Atlas')
    
    # Step 1: Install requirements
    if not install_requirements():
        print('❌ Failed to install requirements')
        return False
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Step 4: Setup Kaggle API
    kaggle_ready = setup_kaggle_api()
    
    # Step 5: Setup MongoDB Atlas
    mongodb_ready = setup_mongodb_atlas()
    
    # Step 6: Setup MySQL
    mysql_ready = setup_mysql()
    
    # Step 7: Download sample datasets (if Kaggle is ready)
    if kaggle_ready:
        download_sample_datasets()
    
    # Summary
    print('\n🎉 SETUP COMPLETE!')
    print('=' * 30)
    print(f'🎮 GPU Available: {"✅" if gpu_available else "❌"}')
    print(f'📥 Kaggle API: {"✅" if kaggle_ready else "⚠️  Setup required"}')
    print(f'💾 MongoDB Atlas: {"✅" if mongodb_ready else "⚠️  Setup required"}')
    print(f'🐬 MySQL: {"✅" if mysql_ready else "⚠️  Setup required"}')
    
    print('\n🚀 Next Steps:')
    print('1. Configure Kaggle API (if not done)')
    print('2. Setup MongoDB Atlas (if not done)')
    print('3. Setup MySQL (optional)')
    print('4. Run: python enhanced_defense_ml.py')
    
    if gpu_available:
        print('🎮 GPU training ready!')
    else:
        print('⚠️  Using CPU training (slower)')
    
    return True

if __name__ == '__main__':
    main()
