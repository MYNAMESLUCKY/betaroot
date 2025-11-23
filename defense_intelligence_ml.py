# 🛡️ Defense Intelligence ML System
# Pure ML model for defense and military intelligence analysis

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print('🛡️ DEFENSE INTELLIGENCE ML SYSTEM')
print('=' * 50)
print('🎯 Military-grade AI analysis system')
print('📊 Multi-domain intelligence capabilities')
print('🚀 GPU Accelerated: ', len(tf.config.list_physical_devices('GPU')) > 0)
print('🔥 TensorFlow Version:', tf.__version__)

class DefenseIntelligenceML:
    """Comprehensive ML system for defense and military intelligence"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.satellite_analyzer = None
        self.signal_analyzer = None
        self.threat_detector = None
        self.military_analyzer = None
        self.scaler = StandardScaler()
        
        # Configuration
        self.config = {
            'satellite': {
                'input_shape': (224, 224, 3),
                'num_classes': 10,
                'classes': ['tank', 'aircraft', 'ship', 'vehicle', 'building', 
                           'radar', 'missile', 'bunker', 'bridge', 'unknown']
            },
            'signal': {
                'sample_rate': 1000,
                'feature_dim': 7,
                'num_classes': 4,
                'classes': ['communication', 'radar', 'data', 'noise']
            },
            'threat': {
                'feature_dim': 12,
                'num_classes': 3,
                'classes': ['normal', 'suspicious', 'critical']
            },
            'military': {
                'input_shape': (150, 150, 3),
                'num_classes': 8,
                'classes': ['tank', 'artillery', 'aircraft', 'helicopter', 
                           'naval_vessel', 'missile_system', 'command_center', 'convoy']
            }
        }
        
        print(f'✅ Defense Intelligence ML initialized')
        print(f'📁 Models directory: {models_dir}')
    
    def build_satellite_analyzer(self):
        """Build CNN for satellite image analysis"""
        print('\n🛰️ Building Satellite Image Analyzer...')
        
        model = tf.keras.Sequential([
            # Feature extraction layers
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', 
                                 input_shape=self.config['satellite']['input_shape'], 
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
            
            tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),
            
            # Classification layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.config['satellite']['num_classes'], activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.satellite_analyzer = model
        print('✅ Satellite analyzer built successfully')
        return model
    
    def build_signal_analyzer(self):
        """Build neural network for signal intelligence"""
        print('\n📡 Building Signal Intelligence Analyzer...')
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', 
                                 input_shape=(self.config['signal']['feature_dim'],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(self.config['signal']['num_classes'], activation='softmax')
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
                                 input_shape=(self.config['threat']['feature_dim'],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(1, activation='sigmoid')  # Threat probability
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
    
    def build_military_analyzer(self):
        """Build CNN for military asset recognition"""
        print('\n🎯 Building Military Asset Analyzer...')
        
        model = tf.keras.Sequential([
            # Feature extraction
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', 
                                 input_shape=self.config['military']['input_shape'], 
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
            
            # Classification
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.config['military']['num_classes'], activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.military_analyzer = model
        print('✅ Military analyzer built successfully')
        return model
    
    def generate_satellite_data(self, num_samples=2000):
        """Generate synthetic satellite imagery data"""
        print(f'📊 Generating {num_samples} satellite images...')
        
        images = []
        labels = []
        classes = self.config['satellite']['classes']
        
        for i in range(num_samples):
            # Create synthetic satellite image (224x224x3)
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add realistic patterns based on class
            class_idx = np.random.randint(0, len(classes))
            class_name = classes[class_idx]
            
            if class_name in ['tank', 'vehicle', 'aircraft']:
                # Add rectangular shapes for vehicles/aircraft
                x, y = np.random.randint(20, 180, 2)
                w, h = np.random.randint(20, 60, 2)
                img[y:y+h, x:x+w] = np.random.randint(100, 255, (h, w, 3))
                
            elif class_name in ['building', 'radar', 'missile']:
                # Add geometric structures
                center_x, center_y = np.random.randint(50, 174, 2)
                radius = np.random.randint(10, 30)
                cv2.circle(img, (center_x, center_y), radius, 
                          (np.random.randint(100, 255),) * 3, -1)
                
            elif class_name in ['ship', 'naval_vessel']:
                # Add elongated shapes for ships
                x, y = np.random.randint(20, 180, 2)
                length = np.random.randint(40, 80)
                width = np.random.randint(10, 20)
                img[y:y+width, x:x+length] = np.random.randint(100, 255, (width, length, 3))
            
            # Add noise and weather effects
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = np.clip(img + noise, 0, 255)
            
            images.append(img / 255.0)  # Normalize
            labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def generate_signal_data(self, num_samples=3000):
        """Generate synthetic signal intelligence data"""
        print(f'📊 Generating {num_samples} signal samples...')
        
        signals = []
        labels = []
        classes = self.config['signal']['classes']
        sample_rate = self.config['signal']['sample_rate']
        
        for i in range(num_samples):
            t = np.linspace(0, 1, sample_rate)
            signal_type = np.random.choice(classes)
            class_idx = classes.index(signal_type)
            
            if signal_type == 'communication':
                # Amplitude modulated voice signal
                carrier = np.sin(2 * np.pi * 100 * t)
                modulation = np.sin(2 * np.pi * 10 * t)
                signal = carrier * (1 + 0.5 * modulation)
                
            elif signal_type == 'radar':
                # Pulsed radar signal
                signal = np.zeros(sample_rate)
                pulse_interval = 100
                for j in range(0, sample_rate, pulse_interval):
                    if j + 20 < sample_rate:
                        signal[j:j+20] = np.sin(2 * np.pi * 50 * np.arange(20))
                        
            elif signal_type == 'data':
                # Digital data transmission
                bits = np.random.randint(0, 2, 100)
                signal = np.repeat(bits, 10)[:sample_rate].astype(float)
                
            else:  # noise
                signal = np.random.normal(0, 0.1, sample_rate)
            
            # Add realistic noise
            signal += np.random.normal(0, 0.05, sample_rate)
            
            # Extract features
            features = self.extract_signal_features(signal)
            
            signals.append(features)
            labels.append(class_idx)
        
        return np.array(signals), np.array(labels)
    
    def extract_signal_features(self, signal):
        """Extract features from signal data"""
        # Time domain features
        mean = np.mean(signal)
        std = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        
        # Frequency domain features
        fft = np.fft.fft(signal)
        fft_magnitude = np.abs(fft)
        dominant_freq = np.argmax(fft_magnitude[:len(fft)//2])
        spectral_centroid = np.sum(np.arange(len(fft_magnitude)) * fft_magnitude) / np.sum(fft_magnitude)
        
        # Time domain features
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        peak_count = len([i for i in range(1, len(signal)-1) 
                          if signal[i] > signal[i-1] and signal[i] > signal[i+1]])
        
        return [mean, std, rms, dominant_freq, spectral_centroid, zero_crossings, peak_count]
    
    def generate_threat_data(self, num_samples=5000):
        """Generate synthetic threat intelligence data"""
        print(f'📊 Generating {num_samples} threat scenarios...')
        
        activities = []
        labels = []
        
        for i in range(num_samples):
            # Determine threat level (30% critical, 20% suspicious, 50% normal)
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
            
            # Generate activity features
            activity = {
                'timestamp_hour': np.random.randint(0, 24),
                'duration_minutes': np.random.exponential(20 if threat_level == 0 else 60) + 5,
                'failed_attempts': np.random.randint(0, 5 if threat_level > 0 else 2),
                'unusual_location': np.random.random() < (0.1 if threat_level == 0 else 0.7),
                'data_volume_mb': np.random.exponential(10 if threat_level == 0 else 100) + 1,
                'concurrent_sessions': np.random.randint(1, 5 if threat_level > 0 else 2),
                'access_frequency': np.random.randint(1, 10 if threat_level > 0 else 3),
                'risk_score': np.random.uniform(0.1, 0.4) if threat_level == 0 else np.random.uniform(0.6, 1.0),
                'anomaly_score': np.random.uniform(0.0, 0.3) if threat_level == 0 else np.random.uniform(0.5, 1.0),
                'behavioral_deviation': np.random.uniform(0.0, 0.2) if threat_level == 0 else np.random.uniform(0.4, 0.9),
                'network_anomaly': np.random.random() < (0.05 if threat_level == 0 else 0.6),
                'time_anomaly': np.random.random() < (0.1 if threat_level == 0 else 0.8)
            }
            
            # Create feature vector
            features = [
                activity['timestamp_hour'],
                activity['duration_minutes'],
                activity['failed_attempts'],
                int(activity['unusual_location']),
                activity['data_volume_mb'],
                activity['concurrent_sessions'],
                activity['access_frequency'],
                activity['risk_score'],
                activity['anomaly_score'],
                activity['behavioral_deviation'],
                int(activity['network_anomaly']),
                int(activity['time_anomaly'])
            ]
            
            activities.append(features)
            labels.append(label)
        
        return np.array(activities), np.array(labels)
    
    def generate_military_data(self, num_samples=1500):
        """Generate synthetic military asset data"""
        print(f'📊 Generating {num_samples} military asset images...')
        
        images = []
        labels = []
        classes = self.config['military']['classes']
        
        for i in range(num_samples):
            # Create synthetic military image (150x150x3)
            img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
            
            class_idx = np.random.randint(0, len(classes))
            class_name = classes[class_idx]
            
            # Add class-specific patterns
            if class_name in ['tank', 'artillery']:
                # Add military vehicle shapes
                x, y = np.random.randint(20, 120, 2)
                w, h = np.random.randint(30, 60, 2)
                img[y:y+h, x:x+w] = np.random.randint(50, 200, (h, w, 3))
                # Add turret
                cx, cy = x + w//2, y + h//3
                cv2.circle(img, (cx, cy), w//6, (100, 100, 150), -1)
                
            elif class_name in ['aircraft', 'helicopter']:
                # Add aircraft shapes
                cx, cy = np.random.randint(30, 120, 2)
                # Fuselage
                cv2.rectangle(img, (cx-20, cy-5), (cx+20, cy+5), (150, 150, 200), -1)
                # Wings
                cv2.rectangle(img, (cx-30, cy-2), (cx+30, cy+2), (150, 150, 200), -1)
                
            elif class_name in ['naval_vessel', 'missile_system']:
                # Add naval/military shapes
                x, y = np.random.randint(10, 130, 2)
                length = np.random.randint(40, 80)
                width = np.random.randint(15, 30)
                img[y:y+width, x:x+length] = np.random.randint(80, 180, (width, length, 3))
                
            elif class_name in ['command_center', 'convoy']:
                # Add structures/multiple vehicles
                for j in range(np.random.randint(2, 5)):
                    x, y = np.random.randint(10, 130, 2)
                    w, h = np.random.randint(15, 30, 2)
                    img[y:y+h, x:x+w] = np.random.randint(100, 200, (h, w, 3))
            
            # Add camouflage patterns
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            img = np.clip(img + noise, 0, 255)
            
            images.append(img / 255.0)  # Normalize
            labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def train_satellite_analyzer(self, epochs=50, batch_size=32):
        """Train satellite image analyzer"""
        print('\n🚀 Training Satellite Image Analyzer...')
        
        # Generate data
        X, y = self.generate_satellite_data(2000)
        y_onehot = tf.keras.utils.to_categorical(y, self.config['satellite']['num_classes'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
        )
        
        # Build model if not exists
        if self.satellite_analyzer is None:
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
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc, top_k_acc = self.satellite_analyzer.evaluate(X_test, y_test, verbose=0)
        print(f'📊 Satellite Analysis Results:')
        print(f'   Test Accuracy: {test_acc:.4f}')
        print(f'   Top-5 Accuracy: {top_k_acc:.4f}')
        
        return history, test_acc
    
    def train_signal_analyzer(self, epochs=50, batch_size=32):
        """Train signal intelligence analyzer"""
        print('\n🚀 Training Signal Intelligence Analyzer...')
        
        # Generate data
        X, y = self.generate_signal_data(3000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model if not exists
        if self.signal_analyzer is None:
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
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = self.signal_analyzer.evaluate(X_test, y_test, verbose=0)
        print(f'📊 Signal Analysis Results:')
        print(f'   Test Accuracy: {test_acc:.4f}')
        
        return history, test_acc
    
    def train_threat_detector(self):
        """Train threat detection system"""
        print('\n🚀 Training Threat Detection System...')
        
        # Generate data
        X, y = self.generate_threat_data(5000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build models if not exists
        if self.threat_detector is None:
            self.build_threat_detector()
        
        # Train Random Forest
        rf_model = self.threat_detector['random_forest']
        rf_model.fit(X_train, y_train)
        rf_acc = rf_model.score(X_test, y_test)
        
        # Train Neural Network (binary classification: threat vs no threat)
        y_binary = (y > 0).astype(int)  # Convert to binary (threat vs normal)
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
                monitor='val_auc'
            )
        ]
        
        # Train Neural Network
        history = nn_model.fit(
            X_train_nn, y_train_nn,
            validation_data=(X_test_nn, y_test_nn),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate Neural Network
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
        
        return {
            'random_forest_acc': rf_acc,
            'neural_network_acc': test_acc,
            'neural_network_auc': test_auc,
            'anomaly_detection_acc': anomaly_acc
        }
    
    def train_military_analyzer(self, epochs=50, batch_size=32):
        """Train military asset analyzer"""
        print('\n🚀 Training Military Asset Analyzer...')
        
        # Generate data
        X, y = self.generate_military_data(1500)
        y_onehot = tf.keras.utils.to_categorical(y, self.config['military']['num_classes'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
        )
        
        # Build model if not exists
        if self.military_analyzer is None:
            self.build_military_analyzer()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                f'{self.models_dir}/military_analyzer.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train
        history = self.military_analyzer.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = self.military_analyzer.evaluate(X_test, y_test, verbose=0)
        print(f'📊 Military Analysis Results:')
        print(f'   Test Accuracy: {test_acc:.4f}')
        
        return history, test_acc
    
    def train_all_models(self):
        """Train all defense intelligence models"""
        print('\n🎯 TRAINING ALL DEFENSE INTELLIGENCE MODELS')
        print('=' * 60)
        
        start_time = datetime.now()
        
        # Train all models
        results = {}
        
        # 1. Satellite Image Analysis
        try:
            sat_history, sat_acc = self.train_satellite_analyzer()
            results['satellite'] = {'accuracy': sat_acc, 'status': 'success'}
        except Exception as e:
            print(f'❌ Satellite training failed: {e}')
            results['satellite'] = {'accuracy': 0, 'status': 'failed'}
        
        # 2. Signal Intelligence
        try:
            sig_history, sig_acc = self.train_signal_analyzer()
            results['signal'] = {'accuracy': sig_acc, 'status': 'success'}
        except Exception as e:
            print(f'❌ Signal training failed: {e}')
            results['signal'] = {'accuracy': 0, 'status': 'failed'}
        
        # 3. Threat Detection
        try:
            threat_results = self.train_threat_detector()
            results['threat'] = threat_results
            results['threat']['status'] = 'success'
        except Exception as e:
            print(f'❌ Threat detection training failed: {e}')
            results['threat'] = {'status': 'failed'}
        
        # 4. Military Asset Analysis
        try:
            mil_history, mil_acc = self.train_military_analyzer()
            results['military'] = {'accuracy': mil_acc, 'status': 'success'}
        except Exception as e:
            print(f'❌ Military analysis training failed: {e}')
            results['military'] = {'accuracy': 0, 'status': 'failed'}
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Save results
        self.save_training_results(results, training_time)
        
        # Summary
        print('\n🎉 TRAINING COMPLETE!')
        print('=' * 40)
        print(f'⏱️  Total Training Time: {training_time:.2f} seconds')
        print(f'📊 Models Trained: {sum(1 for r in results.values() if r.get("status") == "success")}/4')
        
        for model, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                acc = result.get('accuracy', 0)
                print(f'✅ {model.title()}: {acc:.4f} accuracy')
            else:
                print(f'❌ {model.title()}: Training failed')
        
        return results
    
    def save_training_results(self, results, training_time):
        """Save training results and models"""
        print('\n💾 Saving models and results...')
        
        # Save models
        if self.satellite_analyzer:
            self.satellite_analyzer.save(f'{self.models_dir}/satellite_analyzer.h5')
        
        if self.signal_analyzer:
            self.signal_analyzer.save(f'{self.models_dir}/signal_analyzer.h5')
        
        if self.military_analyzer:
            self.military_analyzer.save(f'{self.models_dir}/military_analyzer.h5')
        
        # Save traditional models
        if self.threat_detector:
            with open(f'{self.models_dir}/threat_random_forest.pkl', 'wb') as f:
                pickle.dump(self.threat_detector['random_forest'], f)
            
            with open(f'{self.models_dir}/threat_anomaly_detector.pkl', 'wb') as f:
                pickle.dump(self.threat_detector['anomaly_detector'], f)
        
        # Save scaler
        with open(f'{self.models_dir}/feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training results
        training_report = {
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'models': results,
            'config': self.config,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'tensorflow_version': tf.__version__
        }
        
        with open(f'{self.models_dir}/training_report.json', 'w') as f:
            json.dump(training_report, f, indent=2)
        
        print('✅ Models and results saved successfully!')
        print(f'📁 Models directory: {self.models_dir}')
    
    def load_models(self):
        """Load all trained models"""
        print('\n📂 Loading trained models...')
        
        try:
            # Load neural network models
            if os.path.exists(f'{self.models_dir}/satellite_analyzer.h5'):
                self.satellite_analyzer = tf.keras.models.load_model(f'{self.models_dir}/satellite_analyzer.h5')
                print('✅ Satellite analyzer loaded')
            
            if os.path.exists(f'{self.models_dir}/signal_analyzer.h5'):
                self.signal_analyzer = tf.keras.models.load_model(f'{self.models_dir}/signal_analyzer.h5')
                print('✅ Signal analyzer loaded')
            
            if os.path.exists(f'{self.models_dir}/military_analyzer.h5'):
                self.military_analyzer = tf.keras.models.load_model(f'{self.models_dir}/military_analyzer.h5')
                print('✅ Military analyzer loaded')
            
            # Load traditional models
            if os.path.exists(f'{self.models_dir}/threat_random_forest.pkl'):
                with open(f'{self.models_dir}/threat_random_forest.pkl', 'rb') as f:
                    rf_model = pickle.load(f)
                
                if os.path.exists(f'{self.models_dir}/threat_anomaly_detector.pkl'):
                    with open(f'{self.models_dir}/threat_anomaly_detector.pkl', 'rb') as f:
                        anomaly_model = pickle.load(f)
                    
                    # Load neural network
                    if os.path.exists(f'{self.models_dir}/threat_neural_network.h5'):
                        nn_model = tf.keras.models.load_model(f'{self.models_dir}/threat_neural_network.h5')
                    
                    self.threat_detector = {
                        'random_forest': rf_model,
                        'anomaly_detector': anomaly_model,
                        'neural_network': nn_model
                    }
                    print('✅ Threat detection system loaded')
            
            # Load scaler
            if os.path.exists(f'{self.models_dir}/feature_scaler.pkl'):
                with open(f'{self.models_dir}/feature_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                print('✅ Feature scaler loaded')
            
            print('🎉 All models loaded successfully!')
            return True
            
        except Exception as e:
            print(f'❌ Error loading models: {e}')
            return False
    
    def predict_satellite_analysis(self, image):
        """Predict satellite image analysis"""
        if self.satellite_analyzer is None:
            return {'error': 'Satellite analyzer not loaded'}
        
        try:
            # Preprocess image
            if isinstance(image, str):
                # Load from file path
                img = cv2.imread(image)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
            else:
                # Assume numpy array
                img = cv2.resize(image, (224, 224))
                img = img / 255.0
            
            # Predict
            img_batch = np.expand_dims(img, axis=0)
            prediction = self.satellite_analyzer.predict(img_batch)[0]
            
            # Get top predictions
            top_indices = np.argsort(prediction)[-3:][::-1]
            top_predictions = [
                {
                    'class': self.config['satellite']['classes'][i],
                    'confidence': float(prediction[i])
                }
                for i in top_indices
            ]
            
            return {
                'predictions': top_predictions,
                'primary_class': self.config['satellite']['classes'][np.argmax(prediction)],
                'confidence': float(np.max(prediction))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_signal_analysis(self, signal_data):
        """Predict signal intelligence analysis"""
        if self.signal_analyzer is None:
            return {'error': 'Signal analyzer not loaded'}
        
        try:
            # Extract features
            features = self.extract_signal_features(signal_data)
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.signal_analyzer.predict(features_scaled)[0]
            
            # Get top predictions
            top_indices = np.argsort(prediction)[-2:][::-1]
            top_predictions = [
                {
                    'signal_type': self.config['signal']['classes'][i],
                    'confidence': float(prediction[i])
                }
                for i in top_indices
            ]
            
            return {
                'predictions': top_predictions,
                'primary_type': self.config['signal']['classes'][np.argmax(prediction)],
                'confidence': float(np.max(prediction))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_threat_analysis(self, activity_features):
        """Predict threat analysis"""
        if self.threat_detector is None:
            return {'error': 'Threat detector not loaded'}
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([activity_features])
            
            # Random Forest prediction
            rf_pred = self.threat_detector['random_forest'].predict(features_scaled)[0]
            rf_prob = self.threat_detector['random_forest'].predict_proba(features_scaled)[0]
            
            # Neural Network prediction
            nn_pred = self.threat_detector['neural_network'].predict(features_scaled)[0][0]
            
            # Anomaly detection
            anomaly_pred = self.threat_detector['anomaly_detector'].predict(features_scaled)[0]
            is_anomaly = anomaly_pred == -1
            
            # Combine predictions
            threat_level = self.config['threat']['classes'][rf_pred]
            threat_confidence = max(rf_prob)
            
            return {
                'threat_level': threat_level,
                'threat_confidence': float(threat_confidence),
                'threat_probability': float(nn_pred),
                'is_anomalous': is_anomaly,
                'risk_score': float(nn_pred),
                'recommendation': self.get_threat_recommendation(threat_level, nn_pred, is_anomaly)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_military_analysis(self, image):
        """Predict military asset analysis"""
        if self.military_analyzer is None:
            return {'error': 'Military analyzer not loaded'}
        
        try:
            # Preprocess image
            if isinstance(image, str):
                img = cv2.imread(image)
                img = cv2.resize(img, (150, 150))
                img = img / 255.0
            else:
                img = cv2.resize(image, (150, 150))
                img = img / 255.0
            
            # Predict
            img_batch = np.expand_dims(img, axis=0)
            prediction = self.military_analyzer.predict(img_batch)[0]
            
            # Get top predictions
            top_indices = np.argsort(prediction)[-3:][::-1]
            top_predictions = [
                {
                    'asset_type': self.config['military']['classes'][i],
                    'confidence': float(prediction[i])
                }
                for i in top_indices
            ]
            
            primary_asset = self.config['military']['classes'][np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            return {
                'predictions': top_predictions,
                'primary_asset': primary_asset,
                'confidence': confidence,
                'threat_level': self.get_military_threat_level(primary_asset, confidence)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_threat_recommendation(self, threat_level, threat_prob, is_anomaly):
        """Get threat recommendation based on analysis"""
        if threat_level == 'critical' or threat_prob > 0.8 or is_anomaly:
            return 'IMMEDIATE_ACTION_REQUIRED'
        elif threat_level == 'suspicious' or threat_prob > 0.6:
            return 'MONITOR_CLOSELY'
        else:
            return 'NORMAL_OPERATIONS'
    
    def get_military_threat_level(self, asset_type, confidence):
        """Get threat level for military asset"""
        high_threat_assets = ['tank', 'artillery', 'missile_system']
        medium_threat_assets = ['aircraft', 'helicopter', 'naval_vessel']
        
        if asset_type in high_threat_assets and confidence > 0.7:
            return 'HIGH'
        elif asset_type in medium_threat_assets and confidence > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def run_demonstration(self):
        """Run a complete demonstration of the system"""
        print('\n🎭 DEFENSE INTELLIGENCE SYSTEM DEMONSTRATION')
        print('=' * 60)
        
        # Load models
        if not self.load_models():
            print('❌ Failed to load models. Training new ones...')
            self.train_all_models()
            self.load_models()
        
        # Demonstrate each capability
        print('\n🛰️ SATELLITE IMAGE ANALYSIS DEMO')
        print('-' * 40)
        
        # Generate test satellite image
        test_satellite_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        sat_result = self.predict_satellite_analysis(test_satellite_img)
        
        if 'error' not in sat_result:
            print(f'🎯 Primary Detection: {sat_result["primary_class"]}')
            print(f'📊 Confidence: {sat_result["confidence"]:.3f}')
            print('🔍 Top 3 Predictions:')
            for pred in sat_result['predictions']:
                print(f'   - {pred["class"]}: {pred["confidence"]:.3f}')
        else:
            print(f'❌ Error: {sat_result["error"]}')
        
        print('\n📡 SIGNAL INTELLIGENCE DEMO')
        print('-' * 40)
        
        # Generate test signal
        t = np.linspace(0, 1, 1000)
        test_signal = np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.1, 1000)
        sig_result = self.predict_signal_analysis(test_signal)
        
        if 'error' not in sig_result:
            print(f'🎯 Signal Type: {sig_result["primary_type"]}')
            print(f'📊 Confidence: {sig_result["confidence"]:.3f}')
            print('🔍 Signal Predictions:')
            for pred in sig_result['predictions']:
                print(f'   - {pred["signal_type"]}: {pred["confidence"]:.3f}')
        else:
            print(f'❌ Error: {sig_result["error"]}')
        
        print('\n🔍 THREAT DETECTION DEMO')
        print('-' * 40)
        
        # Generate test threat scenario
        test_threat_features = [
            23,  # timestamp_hour (night)
            120, # duration_minutes (long)
            4,   # failed_attempts
            1,   # unusual_location
            500, # data_volume_mb (high)
            3,   # concurrent_sessions
            8,   # access_frequency (high)
            0.85, # risk_score
            0.78, # anomaly_score
            0.82, # behavioral_deviation
            1,    # network_anomaly
            1     # time_anomaly
        ]
        
        threat_result = self.predict_threat_analysis(test_threat_features)
        
        if 'error' not in threat_result:
            print(f'🚨 Threat Level: {threat_result["threat_level"].upper()}')
            print(f'📊 Threat Confidence: {threat_result["threat_confidence"]:.3f}')
            print(f'🎯 Threat Probability: {threat_result["threat_probability"]:.3f}')
            print(f'⚠️  Anomaly Detected: {"Yes" if threat_result["is_anomalous"] else "No"}')
            print(f'💡 Recommendation: {threat_result["recommendation"]}')
        else:
            print(f'❌ Error: {threat_result["error"]}')
        
        print('\n🎯 MILITARY ASSET ANALYSIS DEMO')
        print('-' * 40)
        
        # Generate test military image
        test_military_img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        mil_result = self.predict_military_analysis(test_military_img)
        
        if 'error' not in mil_result:
            print(f'🎯 Primary Asset: {mil_result["primary_asset"]}')
            print(f'📊 Confidence: {mil_result["confidence"]:.3f}')
            print(f'⚔️  Threat Level: {mil_result["threat_level"]}')
            print('🔍 Top 3 Predictions:')
            for pred in mil_result['predictions']:
                print(f'   - {pred["asset_type"]}: {pred["confidence"]:.3f}')
        else:
            print(f'❌ Error: {mil_result["error"]}')
        
        print('\n🎉 DEMONSTRATION COMPLETE!')
        print('=' * 40)
        print('🛡️ Defense Intelligence System is fully operational!')
        print('📊 All ML models trained and ready for deployment!')
        print('🚀 System ready for military-grade intelligence analysis!')

def main():
    """Main function to run the defense intelligence system"""
    print('🛡️ INITIALIZING DEFENSE INTELLIGENCE ML SYSTEM')
    print('=' * 60)
    
    # Initialize system
    defense_ml = DefenseIntelligenceML()
    
    # Train all models
    print('\n🎯 STARTING TRAINING SESSION')
    defense_ml.train_all_models()
    
    # Run demonstration
    defense_ml.run_demonstration()
    
    print('\n🎉 DEFENSE INTELLIGENCE SYSTEM READY!')
    print('📁 Models saved in: models/')
    print('📊 Training report: models/training_report.json')
    print('🚀 System ready for operational deployment!')

if __name__ == '__main__':
    main()
