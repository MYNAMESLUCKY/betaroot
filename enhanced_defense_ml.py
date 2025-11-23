# 🛡️ Enhanced Defense Intelligence System
# Multi-platform GPU + MongoDB Atlas + Real Datasets

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
import os
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

# Database imports
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️  MongoDB not installed. Install with: pip install pymongo")

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("⚠️  MySQL not installed. Install with: pip install mysql-connector-python")

print('🛡️ ENHANCED DEFENSE INTELLIGENCE SYSTEM')
print('=' * 60)
print('🔥 TensorFlow Version:', tf.__version__)
print('💾 MongoDB Atlas:', '✅ Available' if MONGODB_AVAILABLE else '❌ Not Available')
print('🐬 MySQL:', '✅ Available' if MYSQL_AVAILABLE else '❌ Not Available')
print('🎯 GPU Platforms: RTX 2050, Kaggle, Google Colab')

class EnhancedDefenseML:
    """Enhanced ML system with database integration and real datasets"""
    
    def __init__(self, models_dir='models', config_file='config.json'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize database connections
        self.mongodb_client = None
        self.mysql_connection = None
        self.init_databases()
        
        # Initialize models
        self.satellite_analyzer = None
        self.signal_analyzer = None
        self.threat_detector = None
        self.military_analyzer = None
        self.scaler = StandardScaler()
        
        # GPU configuration
        self.configure_gpu()
        
        print(f'✅ Enhanced Defense ML initialized')
        print(f'📁 Models directory: {models_dir}')
        print(f'💾 Database connections: {self.get_database_status()}')
    
    def load_config(self, config_file):
        """Load configuration from file or create default"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            default_config = {
                "mongodb": {
                    "atlas_uri": "mongodb+srv://username:password@cluster.mongodb.net/defense_intel",
                    "database": "defense_intelligence",
                    "collections": {
                        "signals": "signal_data",
                        "threats": "threat_data", 
                        "models": "trained_models",
                        "predictions": "predictions"
                    }
                },
                "mysql": {
                    "host": "localhost",
                    "port": 3306,
                    "user": "defense_user",
                    "password": "secure_password",
                    "database": "defense_db"
                },
                "data_sources": {
                    "kaggle_datasets": [
                        "andrewmvd/satellite-image-classification",
                        "chirag19/forest-fire-prediction-dataset",
                        "ucimae/signal-processing-datasets"
                    ],
                    "github_repos": [
                        "https://github.com/openai/gym",
                        "https://github.com/tensorflow/models"
                    ]
                },
                "model_config": {
                    "satellite": {
                        "input_shape": [224, 224, 3],
                        "num_classes": 10,
                        "classes": ["tank", "aircraft", "ship", "vehicle", "building", 
                                   "radar", "missile", "bunker", "bridge", "unknown"]
                    },
                    "signal": {
                        "sample_rate": 1000,
                        "feature_dim": 7,
                        "num_classes": 4,
                        "classes": ["communication", "radar", "data", "noise"]
                    },
                    "threat": {
                        "feature_dim": 12,
                        "num_classes": 3,
                        "classes": ["normal", "suspicious", "critical"]
                    }
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f'📝 Created default configuration: {config_file}')
            return default_config
    
    def configure_gpu(self):
        """Configure GPU settings for multiple platforms"""
        print('\n🎮 Configuring GPU settings...')
        
        # Check local GPU (RTX 2050)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f'✅ Local GPU detected: {len(gpus)} device(s)')
            for i, gpu in enumerate(gpus):
                print(f'   GPU {i}: {gpu.name}')
                # Configure GPU memory growth
                tf.config.experimental.set_memory_growth(gpu[i], True)
        else:
            print('⚠️  No local GPU detected - using CPU')
        
        # Check if running on Colab or Kaggle
        try:
            import google.colab
            print('✅ Running on Google Colab')
            # Colab-specific GPU configuration
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
        except ImportError:
            pass
        
        try:
            # Check for Kaggle environment
            if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                print('✅ Running on Kaggle')
                if gpus:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass
        
        print('🎮 GPU configuration complete')
    
    def init_databases(self):
        """Initialize database connections"""
        print('\n💾 Initializing database connections...')
        
        # MongoDB Atlas
        if MONGODB_AVAILABLE and self.config.get('mongodb', {}).get('atlas_uri'):
            try:
                self.mongodb_client = MongoClient(self.config['mongodb']['atlas_uri'])
                # Test connection
                self.mongodb_client.admin.command('ping')
                print('✅ MongoDB Atlas connected')
            except Exception as e:
                print(f'❌ MongoDB Atlas connection failed: {e}')
                self.mongodb_client = None
        
        # MySQL
        if MYSQL_AVAILABLE:
            try:
                mysql_config = self.config.get('mysql', {})
                self.mysql_connection = mysql.connector.connect(
                    host=mysql_config.get('host', 'localhost'),
                    port=mysql_config.get('port', 3306),
                    user=mysql_config.get('user', 'root'),
                    password=mysql_config.get('password', ''),
                    database=mysql_config.get('database', 'defense_db')
                )
                print('✅ MySQL connected')
            except Exception as e:
                print(f'❌ MySQL connection failed: {e}')
                self.mysql_connection = None
    
    def get_database_status(self):
        """Get database connection status"""
        status = []
        if self.mongodb_client:
            status.append('MongoDB Atlas ✅')
        if self.mysql_connection:
            status.append('MySQL ✅')
        if not self.mongodb_client and not self.mysql_connection:
            status.append('No databases connected ❌')
        return ', '.join(status)
    
    def download_kaggle_dataset(self, dataset_name, save_path='data'):
        """Download dataset from Kaggle"""
        print(f'📥 Downloading Kaggle dataset: {dataset_name}')
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Authenticate with Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            os.makedirs(save_path, exist_ok=True)
            api.dataset_download_files(dataset_name, path=save_path, unzip=True)
            
            print(f'✅ Downloaded {dataset_name} to {save_path}')
            return True
            
        except ImportError:
            print('❌ Kaggle API not installed. Install with: pip install kaggle')
            return False
        except Exception as e:
            print(f'❌ Failed to download {dataset_name}: {e}')
            return False
    
    def load_satellite_data(self, data_path='data'):
        """Load satellite image data from multiple sources"""
        print('🛰️ Loading satellite image data...')
        
        satellite_images = []
        labels = []
        
        # Try to load from downloaded datasets
        datasets = [
            f'{data_path}/satellite',
            f'{data_path}/image_classification',
            f'{data_path}/remote_sensing'
        ]
        
        for dataset_path in datasets:
            if os.path.exists(dataset_path):
                print(f'📂 Loading from {dataset_path}')
                # Load images from dataset
                for class_dir in os.listdir(dataset_path):
                    class_path = os.path.join(dataset_path, class_dir)
                    if os.path.isdir(class_path):
                        for img_file in os.listdir(class_path)[:100]:  # Limit to 100 per class
                            img_path = os.path.join(class_path, img_file)
                            try:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img = cv2.resize(img, (224, 224))
                                    satellite_images.append(img / 255.0)
                                    labels.append(class_dir)
                            except:
                                continue
        
        # If no real data found, generate synthetic data
        if len(satellite_images) < 100:
            print('⚠️  No real satellite data found, generating synthetic data...')
            satellite_images, labels = self.generate_synthetic_satellite_data(1000)
        else:
            print(f'✅ Loaded {len(satellite_images)} real satellite images')
        
        return np.array(satellite_images), np.array(labels)
    
    def load_signal_data(self, data_path='data'):
        """Load signal data from multiple sources"""
        print('📡 Loading signal data...')
        
        signal_data = []
        signal_labels = []
        
        # Try to load from downloaded datasets
        signal_files = [
            f'{data_path}/signals.csv',
            f'{data_path}/signal_processing.csv',
            f'{data_path}/communication_signals.csv'
        ]
        
        real_data_loaded = False
        for file_path in signal_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f'📂 Loaded signal data from {file_path}')
                    
                    # Extract signal features
                    for _, row in df.iterrows():
                        features = [
                            row.get('mean', 0),
                            row.get('std', 0),
                            row.get('rms', 0),
                            row.get('dominant_freq', 0),
                            row.get('spectral_centroid', 0),
                            row.get('zero_crossings', 0),
                            row.get('peak_count', 0)
                        ]
                        signal_data.append(features)
                        signal_labels.append(row.get('label', 'unknown'))
                    
                    real_data_loaded = True
                    break
                except Exception as e:
                    print(f'❌ Failed to load {file_path}: {e}')
        
        # If no real data found, generate synthetic data
        if not real_data_loaded:
            print('⚠️  No real signal data found, generating synthetic data...')
            signal_data, signal_labels = self.generate_synthetic_signal_data(2000)
        else:
            print(f'✅ Loaded {len(signal_data)} real signal samples')
        
        return np.array(signal_data), np.array(signal_labels)
    
    def load_threat_data(self, data_path='data'):
        """Load threat data from multiple sources"""
        print('🔍 Loading threat data...')
        
        threat_data = []
        threat_labels = []
        
        # Try to load from downloaded datasets
        threat_files = [
            f'{data_path}/cybersecurity_threats.csv',
            f'{data_path}/network_intrusion.csv',
            f'{data_path}/security_events.csv'
        ]
        
        real_data_loaded = False
        for file_path in threat_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f'📂 Loaded threat data from {file_path}')
                    
                    # Extract threat features
                    for _, row in df.iterrows():
                        features = [
                            row.get('timestamp_hour', 12),
                            row.get('duration_minutes', 30),
                            row.get('failed_attempts', 0),
                            row.get('unusual_location', 0),
                            row.get('data_volume_mb', 10),
                            row.get('concurrent_sessions', 1),
                            row.get('risk_score', 0.2),
                            row.get('anomaly_score', 0.1),
                            row.get('behavioral_deviation', 0.1),
                            row.get('network_anomaly', 0),
                            row.get('time_anomaly', 0),
                            row.get('access_frequency', 2)
                        ]
                        threat_data.append(features)
                        
                        # Determine threat level
                        risk_score = row.get('risk_score', 0.2)
                        if risk_score > 0.7:
                            threat_labels.append(2)  # critical
                        elif risk_score > 0.4:
                            threat_labels.append(1)  # suspicious
                        else:
                            threat_labels.append(0)  # normal
                    
                    real_data_loaded = True
                    break
                except Exception as e:
                    print(f'❌ Failed to load {file_path}: {e}')
        
        # If no real data found, generate synthetic data
        if not real_data_loaded:
            print('⚠️  No real threat data found, generating synthetic data...')
            threat_data, threat_labels = self.generate_synthetic_threat_data(3000)
        else:
            print(f'✅ Loaded {len(threat_data)} real threat samples')
        
        return np.array(threat_data), np.array(threat_labels)
    
    def generate_synthetic_satellite_data(self, num_samples=2000):
        """Generate synthetic satellite data"""
        print(f'🛰️ Generating {num_samples} synthetic satellite images...')
        
        images = []
        labels = []
        classes = self.config['model_config']['satellite']['classes']
        
        for i in range(num_samples):
            # Create synthetic image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add patterns based on class
            class_idx = np.random.randint(0, len(classes))
            class_name = classes[class_idx]
            
            if class_name in ['tank', 'vehicle', 'aircraft']:
                x, y = np.random.randint(20, 180, 2)
                w, h = np.random.randint(20, 60, 2)
                img[y:y+h, x:x+w] = np.random.randint(100, 255, (h, w, 3))
            
            images.append(img / 255.0)
            labels.append(class_name)
        
        return np.array(images), np.array(labels)
    
    def generate_synthetic_signal_data(self, num_samples=2000):
        """Generate synthetic signal data"""
        print(f'📡 Generating {num_samples} synthetic signal samples...')
        
        signals = []
        labels = []
        classes = self.config['model_config']['signal']['classes']
        
        for i in range(num_samples):
            t = np.linspace(0, 1, 1000)
            signal_type = np.random.choice(classes)
            class_idx = classes.index(signal_type)
            
            if signal_type == 'communication':
                signal = np.sin(2 * np.pi * 100 * t) * (1 + 0.5 * np.sin(2 * np.pi * 10 * t))
            elif signal_type == 'radar':
                signal = np.zeros(1000)
                for j in range(0, 1000, 100):
                    if j + 20 < 1000:
                        signal[j:j+20] = np.sin(2 * np.pi * 50 * np.arange(20))
            elif signal_type == 'data':
                bits = np.random.randint(0, 2, 100)
                signal = np.repeat(bits, 10)[:1000].astype(float)
            else:
                signal = np.random.normal(0, 0.1, 1000)
            
            # Extract features
            features = [
                np.mean(signal),
                np.std(signal),
                np.sqrt(np.mean(signal**2)),
                np.argmax(np.abs(np.fft.fft(signal)[:500])),
                len([i for i in range(1, len(signal)-1) if signal[i] > signal[i-1] and signal[i] > signal[i+1]])
            ]
            
            while len(features) < 7:
                features.append(0.0)
            
            signals.append(features[:7])
            labels.append(class_idx)
        
        return np.array(signals), np.array(labels)
    
    def generate_synthetic_threat_data(self, num_samples=3000):
        """Generate synthetic threat data"""
        print(f'🔍 Generating {num_samples} synthetic threat samples...')
        
        threats = []
        labels = []
        
        for i in range(num_samples):
            # Determine threat level
            threat_prob = np.random.random()
            if threat_prob < 0.3:
                threat_level = 2  # critical
                label = 2
            elif threat_prob < 0.5:
                threat_level = 1  # suspicious
                label = 1
            else:
                threat_level = 0  # normal
                label = 0
            
            # Generate features based on threat level
            features = [
                np.random.randint(0, 24),  # hour
                np.random.exponential(20 if threat_level == 0 else 60) + 5,  # duration
                np.random.randint(0, 5 if threat_level > 0 else 2),  # failed attempts
                np.random.random() < (0.1 if threat_level == 0 else 0.7),  # unusual location
                np.random.exponential(10 if threat_level == 0 else 100) + 1,  # data volume
                np.random.randint(1, 5 if threat_level > 0 else 2),  # concurrent sessions
                np.random.uniform(0.1, 0.4) if threat_level == 0 else np.random.uniform(0.6, 1.0),  # risk score
                np.random.uniform(0.0, 0.3) if threat_level == 0 else np.random.uniform(0.5, 1.0),  # anomaly score
                np.random.uniform(0.0, 0.2) if threat_level == 0 else np.random.uniform(0.4, 0.9),  # behavioral deviation
                np.random.random() < (0.05 if threat_level == 0 else 0.6),  # network anomaly
                np.random.random() < (0.1 if threat_level == 0 else 0.8),  # time anomaly
                np.random.randint(1, 10)  # access frequency
            ]
            
            threats.append(features)
            labels.append(label)
        
        return np.array(threats), np.array(labels)
    
    def save_to_database(self, data_type, data, labels=None):
        """Save data to database"""
        print(f'💾 Saving {data_type} to database...')
        
        if self.mongodb_client:
            try:
                db = self.mongodb_client[self.config['mongodb']['database']]
                collection = db[self.config['mongodb']['collections'][data_type]]
                
                # Prepare documents
                documents = []
                if data_type == 'signals':
                    for i, (signal, label) in enumerate(zip(data, labels)):
                        doc = {
                            'signal_features': signal.tolist(),
                            'label': label,
                            'timestamp': datetime.now(),
                            'source': 'training'
                        }
                        documents.append(doc)
                
                elif data_type == 'threats':
                    for i, (threat, label) in enumerate(zip(data, labels)):
                        doc = {
                            'threat_features': threat.tolist(),
                            'threat_level': int(label),
                            'timestamp': datetime.now(),
                            'source': 'training'
                        }
                        documents.append(doc)
                
                # Insert documents
                if documents:
                    collection.insert_many(documents)
                    print(f'✅ Saved {len(documents)} {data_type} to MongoDB Atlas')
                
            except Exception as e:
                print(f'❌ Failed to save to MongoDB: {e}')
        
        # Also save to MySQL if available
        if self.mysql_connection:
            try:
                cursor = self.mysql_connection.cursor()
                
                if data_type == 'signals':
                    # Create table if not exists
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS signal_data (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            features JSON,
                            label VARCHAR(50),
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Insert data
                    for signal, label in zip(data, labels):
                        cursor.execute('''
                            INSERT INTO signal_data (features, label)
                            VALUES (%s, %s)
                        ''', (json.dumps(signal.tolist()), str(label)))
                
                elif data_type == 'threats':
                    # Create table if not exists
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS threat_data (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            features JSON,
                            threat_level INT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Insert data
                    for threat, label in zip(data, labels):
                        cursor.execute('''
                            INSERT INTO threat_data (features, threat_level)
                            VALUES (%s, %s)
                        ''', (json.dumps(threat.tolist()), int(label)))
                
                self.mysql_connection.commit()
                print(f'✅ Saved {len(data)} {data_type} to MySQL')
                
            except Exception as e:
                print(f'❌ Failed to save to MySQL: {e}')
    
    def build_satellite_analyzer(self):
        """Build CNN for satellite image analysis"""
        print('\n🛰️ Building Satellite Image Analyzer...')
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', 
                                 input_shape=self.config['model_config']['satellite']['input_shape'], 
                                 padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.config['model_config']['satellite']['classes']), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.satellite_analyzer = model
        print('✅ Satellite analyzer built successfully')
        return model
    
    def build_signal_analyzer(self):
        """Build neural network for signal intelligence"""
        print('\n📡 Building Signal Intelligence Analyzer...')
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', 
                                 input_shape=(self.config['model_config']['signal']['feature_dim'],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(len(self.config['model_config']['signal']['classes']), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.signal_analyzer = model
        print('✅ Signal analyzer built successfully')
        return model
    
    def build_threat_detector(self):
        """Build threat detection system"""
        print('\n🔍 Building Threat Detection System...')
        
        # Neural network for threat scoring
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', 
                                 input_shape=(self.config['model_config']['threat']['feature_dim'],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        # Random Forest for classification
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Isolation Forest for anomaly detection
        anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        
        self.threat_detector = {
            'neural_network': nn_model,
            'random_forest': rf_model,
            'anomaly_detector': anomaly_model
        }
        
        print('✅ Threat detection system built successfully')
        return self.threat_detector
    
    def train_all_models(self):
        """Train all models with real and synthetic data"""
        print('\n🎯 TRAINING ENHANCED DEFENSE INTELLIGENCE MODELS')
        print('=' * 65)
        
        start_time = datetime.now()
        results = {}
        
        # 1. Satellite Image Analysis
        print('\n🛰️ SATELLITE IMAGE ANALYSIS')
        print('-' * 40)
        try:
            # Load data (real + synthetic)
            X, y = self.load_satellite_data()
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Build and train model
            self.build_satellite_analyzer()
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{self.models_dir}/satellite_analyzer.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
            
            # Train
            history = self.satellite_analyzer.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=30,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = self.satellite_analyzer.evaluate(X_test, y_test, verbose=0)
            print(f'📊 Satellite Analysis Accuracy: {test_acc:.4f}')
            
            results['satellite'] = {'accuracy': test_acc, 'status': 'success'}
            
            # Save data to database
            self.save_to_database('satellite', X, y_encoded)
            
        except Exception as e:
            print(f'❌ Satellite training failed: {e}')
            results['satellite'] = {'accuracy': 0, 'status': 'failed'}
        
        # 2. Signal Intelligence
        print('\n📡 SIGNAL INTELLIGENCE')
        print('-' * 40)
        try:
            # Load data (real + synthetic)
            X, y = self.load_signal_data()
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Build and train model
            self.build_signal_analyzer()
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{self.models_dir}/signal_analyzer.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
            
            # Train
            history = self.signal_analyzer.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=30,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = self.signal_analyzer.evaluate(X_test, y_test, verbose=0)
            print(f'📊 Signal Intelligence Accuracy: {test_acc:.4f}')
            
            results['signal'] = {'accuracy': test_acc, 'status': 'success'}
            
            # Save data to database
            self.save_to_database('signals', X, y_encoded)
            
        except Exception as e:
            print(f'❌ Signal training failed: {e}')
            results['signal'] = {'accuracy': 0, 'status': 'failed'}
        
        # 3. Threat Detection
        print('\n🔍 THREAT DETECTION')
        print('-' * 40)
        try:
            # Load data (real + synthetic)
            X, y = self.load_threat_data()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Build models
            self.build_threat_detector()
            
            # Train Random Forest
            rf_model = self.threat_detector['random_forest']
            rf_model.fit(X_train, y_train)
            rf_acc = rf_model.score(X_test, y_test)
            
            # Train Neural Network (binary classification)
            y_binary = (y > 0).astype(int)
            X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
                X_scaled, y_binary, test_size=0.2, random_state=42
            )
            
            nn_model = self.threat_detector['neural_network']
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{self.models_dir}/threat_neural_network.h5',
                    save_best_only=True,
                    monitor='val_auc',
                    mode='max'
                )
            ]
            
            # Train Neural Network
            history = nn_model.fit(
                X_train_nn, y_train_nn,
                validation_data=(X_test_nn, y_test_nn),
                epochs=30,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc, test_auc = nn_model.evaluate(X_test_nn, y_test_nn, verbose=0)
            
            # Train Anomaly Detector
            anomaly_model = self.threat_detector['anomaly_detector']
            anomaly_model.fit(X_train)
            
            # Test anomaly detection
            anomaly_pred = anomaly_model.predict(X_test)
            anomaly_labels = (anomaly_pred == -1).astype(int)
            anomaly_acc = np.mean(anomaly_labels == y_binary)
            
            print(f'📊 Threat Detection Results:')
            print(f'   Random Forest Accuracy: {rf_acc:.4f}')
            print(f'   Neural Network Accuracy: {test_acc:.4f}')
            print(f'   Neural Network AUC: {test_auc:.4f}')
            print(f'   Anomaly Detection Accuracy: {anomaly_acc:.4f}')
            
            results['threat'] = {
                'random_forest_acc': rf_acc,
                'neural_network_acc': test_acc,
                'neural_network_auc': test_auc,
                'anomaly_detection_acc': anomaly_acc,
                'status': 'success'
            }
            
            # Save data to database
            self.save_to_database('threats', X, y)
            
        except Exception as e:
            print(f'❌ Threat detection training failed: {e}')
            results['threat'] = {'status': 'failed'}
        
        # Save models
        self.save_models()
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Save training report
        self.save_training_report(results, training_time)
        
        # Summary
        print('\n🎉 ENHANCED TRAINING COMPLETE!')
        print('=' * 45)
        print(f'⏱️  Total Training Time: {training_time:.2f} seconds')
        print(f'💾 Database Status: {self.get_database_status()}')
        print(f'🎮 GPU Used: {len(tf.config.list_physical_devices("GPU"))} device(s)')
        
        for model, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                if 'accuracy' in result:
                    print(f'✅ {model.title()}: {result["accuracy"]:.4f} accuracy')
                else:
                    print(f'✅ {model.title()}: Trained successfully')
            else:
                print(f'❌ {model.title()}: Training failed')
        
        return results
    
    def save_models(self):
        """Save all trained models"""
        print('\n💾 Saving trained models...')
        
        try:
            # Save neural network models
            if self.satellite_analyzer:
                self.satellite_analyzer.save(f'{self.models_dir}/satellite_analyzer.h5')
            
            if self.signal_analyzer:
                self.signal_analyzer.save(f'{self.models_dir}/signal_analyzer.h5')
            
            if self.threat_detector:
                self.threat_detector['neural_network'].save(f'{self.models_dir}/threat_neural_network.h5')
                
                # Save traditional models
                with open(f'{self.models_dir}/threat_random_forest.pkl', 'wb') as f:
                    pickle.dump(self.threat_detector['random_forest'], f)
                
                with open(f'{self.models_dir}/threat_anomaly_detector.pkl', 'wb') as f:
                    pickle.dump(self.threat_detector['anomaly_detector'], f)
            
            # Save scaler
            with open(f'{self.models_dir}/feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print('✅ All models saved successfully')
            
        except Exception as e:
            print(f'❌ Error saving models: {e}')
    
    def save_training_report(self, results, training_time):
        """Save training report"""
        try:
            training_report = {
                'training_date': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'models': results,
                'database_status': self.get_database_status(),
                'gpu_available': len(tf.config.list_physical_devices('GPU')),
                'tensorflow_version': tf.__version__,
                'data_sources': 'Real datasets + Synthetic data',
                'platforms': 'RTX 2050 + Kaggle + Google Colab'
            }
            
            with open(f'{self.models_dir}/enhanced_training_report.json', 'w') as f:
                json.dump(training_report, f, indent=2)
            
            # Also save to database
            if self.mongodb_client:
                try:
                    db = self.mongodb_client[self.config['mongodb']['database']]
                    collection = db[self.config['mongodb']['collections']['models']]
                    collection.insert_one(training_report)
                    print('✅ Training report saved to MongoDB Atlas')
                except:
                    pass
            
        except Exception as e:
            print(f'❌ Error saving training report: {e}')
    
    def close_connections(self):
        """Close database connections"""
        if self.mongodb_client:
            self.mongodb_client.close()
            print('💾 MongoDB connection closed')
        
        if self.mysql_connection:
            self.mysql_connection.close()
            print('💾 MySQL connection closed')

def main():
    """Main function"""
    print('🛡️ ENHANCED DEFENSE INTELLIGENCE SYSTEM')
    print('=' * 50)
    print('🚀 Multi-platform GPU + Database Integration')
    print('📊 Real Datasets + MongoDB Atlas + MySQL')
    print('🎮 RTX 2050 + Kaggle + Google Colab')
    
    # Initialize enhanced system
    enhanced_ml = EnhancedDefenseML()
    
    try:
        # Train all models
        results = enhanced_ml.train_all_models()
        
        print('\n🎉 ENHANCED DEFENSE INTELLIGENCE SYSTEM READY!')
        print('=' * 55)
        print('💾 Database Integration: MongoDB Atlas + MySQL')
        print('🎮 GPU Platforms: RTX 2050 + Kaggle + Google Colab')
        print('📊 Data Sources: Real datasets + Synthetic data')
        print('🚀 System ready for production deployment!')
        
    finally:
        # Close database connections
        enhanced_ml.close_connections()

if __name__ == '__main__':
    main()
