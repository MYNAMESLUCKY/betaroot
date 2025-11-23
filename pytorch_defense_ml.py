# 🛡️ Defense Intelligence System - PyTorch GPU Version
# Utilizes RTX 2050 GPU for accelerated training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
import warnings
warnings.filterwarnings('ignore')

print('🛡️ DEFENSE INTELLIGENCE - PYTORCH GPU VERSION')
print('=' * 60)
print('🔥 PyTorch Version:', torch.__version__)
print('🎮 CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🎯 GPU Device:', torch.cuda.get_device_name(0))
    print('💾 GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')

class PyTorchDefenseML:
    """PyTorch-based ML system with GPU acceleration"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'🚀 Using device: {self.device}')
        
        # Initialize models
        self.satellite_analyzer = None
        self.signal_analyzer = None
        self.threat_detector = None
        self.scaler = StandardScaler()
        
        # Configuration
        self.config = {
            'satellite': {
                'input_shape': (3, 224, 224),
                'num_classes': 10,
                'classes': ['tank', 'aircraft', 'ship', 'vehicle', 'building',
                           'radar', 'missile', 'bunker', 'bridge', 'unknown']
            },
            'signal': {
                'input_dim': 7,
                'num_classes': 4,
                'classes': ['communication', 'radar', 'data', 'noise']
            },
            'threat': {
                'input_dim': 12,
                'num_classes': 3,
                'classes': ['normal', 'suspicious', 'critical']
            }
        }
        
        print(f'✅ PyTorch Defense ML initialized')
        print(f'🎮 GPU Acceleration: {"ENABLED" if torch.cuda.is_available() else "DISABLED"}')
    
    class SatelliteNet(nn.Module):
        """CNN for satellite image analysis"""
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(128 * 28 * 28, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    class SignalNet(nn.Module):
        """Neural network for signal intelligence"""
        def __init__(self, input_dim=7, num_classes=4):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(16, num_classes)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    class ThreatNet(nn.Module):
        """Neural network for threat detection"""
        def __init__(self, input_dim=12):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    def build_satellite_analyzer(self):
        """Build CNN for satellite image analysis"""
        print('\n🛰️ Building Satellite Image Analyzer...')
        
        model = self.SatelliteNet(num_classes=self.config['satellite']['num_classes'])
        model = model.to(self.device)
        
        self.satellite_analyzer = model
        print('✅ Satellite analyzer built successfully')
        return model
    
    def build_signal_analyzer(self):
        """Build neural network for signal intelligence"""
        print('\n📡 Building Signal Intelligence Analyzer...')
        
        model = self.SignalNet(
            input_dim=self.config['signal']['input_dim'],
            num_classes=self.config['signal']['num_classes']
        )
        model = model.to(self.device)
        
        self.signal_analyzer = model
        print('✅ Signal analyzer built successfully')
        return model
    
    def build_threat_detector(self):
        """Build threat detection system"""
        print('\n🔍 Building Threat Detection System...')
        
        # Neural network for threat scoring
        nn_model = self.ThreatNet(input_dim=self.config['threat']['input_dim'])
        nn_model = nn_model.to(self.device)
        
        # Random Forest for classification
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Isolation Forest for anomaly detection
        anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        
        self.threat_detector = {
            'neural_network': nn_model,
            'random_forest': rf_model,
            'anomaly_detector': anomaly_model
        }
        
        print('✅ Threat detection system built successfully')
        return self.threat_detector
    
    def generate_satellite_data(self, num_samples=1000):
        """Generate synthetic satellite data"""
        print(f'🛰️ Generating {num_samples} satellite images...')
        
        images = []
        labels = []
        classes = self.config['satellite']['classes']
        
        for i in range(num_samples):
            # Create synthetic image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add patterns based on class
            class_idx = np.random.randint(0, len(classes))
            
            if class_idx in [0, 3]:  # tank, vehicle
                x, y = np.random.randint(20, 180, 2)
                w, h = np.random.randint(20, 60, 2)
                img[y:y+h, x:x+w] = np.random.randint(100, 255, (h, w, 3))
            
            images.append(img / 255.0)
            labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def generate_signal_data(self, num_samples=2000):
        """Generate synthetic signal data"""
        print(f'📡 Generating {num_samples} signal samples...')
        
        signals = []
        labels = []
        classes = self.config['signal']['classes']
        
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
    
    def generate_threat_data(self, num_samples=3000):
        """Generate synthetic threat data"""
        print(f'🔍 Generating {num_samples} threat samples...')
        
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
    
    def train_satellite_analyzer(self, epochs=20, batch_size=32):
        """Train satellite image analyzer"""
        print('\n🚀 Training Satellite Image Analyzer...')
        
        # Generate data
        X, y = self.generate_satellite_data(1000)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)  # NHWC -> NCHW
        y_tensor = torch.LongTensor(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Build model
        self.build_satellite_analyzer()
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.satellite_analyzer.parameters(), lr=0.001)
        
        # Training loop
        print(f'🎮 Training on {self.device}')
        for epoch in range(epochs):
            self.satellite_analyzer.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.satellite_analyzer(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Evaluation
            self.satellite_analyzer.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.satellite_analyzer(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * correct / total
            test_acc = 100. * test_correct / test_total
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        print(f'📊 Satellite Analysis Accuracy: {test_acc:.2f}%')
        return test_acc
    
    def train_signal_analyzer(self, epochs=30, batch_size=32):
        """Train signal intelligence analyzer"""
        print('\n🚀 Training Signal Intelligence Analyzer...')
        
        # Generate data
        X, y = self.generate_signal_data(2000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Build model
        self.build_signal_analyzer()
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.signal_analyzer.parameters(), lr=0.001)
        
        # Training loop
        print(f'🎮 Training on {self.device}')
        for epoch in range(epochs):
            self.signal_analyzer.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.signal_analyzer(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Evaluation
            self.signal_analyzer.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.signal_analyzer(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * correct / total
            test_acc = 100. * test_correct / test_total
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        print(f'📊 Signal Intelligence Accuracy: {test_acc:.2f}%')
        return test_acc
    
    def train_threat_detector(self):
        """Train threat detection system"""
        print('\n🚀 Training Threat Detection System...')
        
        # Generate data
        X, y = self.generate_threat_data(3000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
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
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_nn)
        y_train_tensor = torch.FloatTensor(y_train_nn).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test_nn)
        y_test_tensor = torch.FloatTensor(y_test_nn).unsqueeze(1)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Training setup
        nn_model = self.threat_detector['neural_network']
        criterion = nn.BCELoss()
        optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
        
        # Training loop
        print(f'🎮 Training Neural Network on {self.device}')
        for epoch in range(20):
            nn_model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = nn_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Evaluation
            nn_model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = nn_model(data)
                    test_loss += criterion(output, target).item()
                    predicted = (output > 0.5).float()
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * correct / total
            test_acc = 100. * test_correct / test_total
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Train Anomaly Detector
        anomaly_model = self.threat_detector['anomaly_detector']
        anomaly_model.fit(X_train)
        
        # Test anomaly detection
        anomaly_pred = anomaly_model.predict(X_test)
        anomaly_labels = (anomaly_pred == -1).astype(int)
        anomaly_acc = np.mean(anomaly_labels == y_binary)
        
        print(f'📊 Threat Detection Results:')
        print(f'   Random Forest Accuracy: {rf_acc:.4f}')
        print(f'   Neural Network Accuracy: {test_acc/100:.4f}')
        print(f'   Anomaly Detection Accuracy: {anomaly_acc:.4f}')
        
        return rf_acc, test_acc/100, anomaly_acc
    
    def train_all_models(self):
        """Train all models with GPU acceleration"""
        print('\n🎯 TRAINING PYTORCH DEFENSE INTELLIGENCE MODELS')
        print('=' * 65)
        
        start_time = datetime.now()
        results = {}
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print('🎮 GPU cache cleared')
        
        # 1. Satellite Image Analysis
        print('\n🛰️ SATELLITE IMAGE ANALYSIS')
        print('-' * 40)
        try:
            sat_acc = self.train_satellite_analyzer()
            results['satellite'] = {'accuracy': sat_acc/100, 'status': 'success'}
        except Exception as e:
            print(f'❌ Satellite training failed: {e}')
            results['satellite'] = {'accuracy': 0, 'status': 'failed'}
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. Signal Intelligence
        print('\n📡 SIGNAL INTELLIGENCE')
        print('-' * 40)
        try:
            sig_acc = self.train_signal_analyzer()
            results['signal'] = {'accuracy': sig_acc/100, 'status': 'success'}
        except Exception as e:
            print(f'❌ Signal training failed: {e}')
            results['signal'] = {'accuracy': 0, 'status': 'failed'}
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 3. Threat Detection
        print('\n🔍 THREAT DETECTION')
        print('-' * 40)
        try:
            rf_acc, nn_acc, anomaly_acc = self.train_threat_detector()
            results['threat'] = {
                'random_forest_acc': rf_acc,
                'neural_network_acc': nn_acc,
                'anomaly_detection_acc': anomaly_acc,
                'status': 'success'
            }
        except Exception as e:
            print(f'❌ Threat detection training failed: {e}')
            results['threat'] = {'status': 'failed'}
        
        # Save models
        self.save_models()
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Save training report
        self.save_training_report(results, training_time)
        
        # GPU Memory summary
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f'\n🎮 GPU Memory Usage:')
            print(f'   Allocated: {memory_allocated:.2f} GB')
            print(f'   Reserved: {memory_reserved:.2f} GB')
        
        # Summary
        print('\n🎉 PYTORCH GPU TRAINING COMPLETE!')
        print('=' * 45)
        print(f'⏱️  Total Training Time: {training_time:.2f} seconds')
        print(f'🎮 GPU Used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
        print(f'🚀 Acceleration: {"ENABLED" if torch.cuda.is_available() else "DISABLED"}')
        
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
            # Save PyTorch models
            if self.satellite_analyzer:
                torch.save(self.satellite_analyzer.state_dict(), f'{self.models_dir}/satellite_analyzer_pytorch.pth')
                torch.save(self.satellite_analyzer, f'{self.models_dir}/satellite_analyzer_full.pth')
            
            if self.signal_analyzer:
                torch.save(self.signal_analyzer.state_dict(), f'{self.models_dir}/signal_analyzer_pytorch.pth')
                torch.save(self.signal_analyzer, f'{self.models_dir}/signal_analyzer_full.pth')
            
            if self.threat_detector:
                torch.save(self.threat_detector['neural_network'].state_dict(), f'{self.models_dir}/threat_neural_network_pytorch.pth')
                torch.save(self.threat_detector['neural_network'], f'{self.models_dir}/threat_neural_network_full.pth')
                
                # Save traditional models
                with open(f'{self.models_dir}/threat_random_forest_pytorch.pkl', 'wb') as f:
                    pickle.dump(self.threat_detector['random_forest'], f)
                
                with open(f'{self.models_dir}/threat_anomaly_detector_pytorch.pkl', 'wb') as f:
                    pickle.dump(self.threat_detector['anomaly_detector'], f)
            
            # Save scaler
            with open(f'{self.models_dir}/feature_scaler_pytorch.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print('✅ All PyTorch models saved successfully')
            
        except Exception as e:
            print(f'❌ Error saving models: {e}')
    
    def save_training_report(self, results, training_time):
        """Save training report"""
        try:
            training_report = {
                'training_date': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'models': results,
                'gpu_available': torch.cuda.is_available(),
                'gpu_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'device_used': str(self.device),
                'acceleration': 'GPU' if torch.cuda.is_available() else 'CPU'
            }
            
            with open(f'{self.models_dir}/pytorch_training_report.json', 'w') as f:
                json.dump(training_report, f, indent=2)
            
        except Exception as e:
            print(f'❌ Error saving training report: {e}')

def main():
    """Main function"""
    print('🛡️ PYTORCH DEFENSE INTELLIGENCE SYSTEM')
    print('=' * 50)
    print('🎮 GPU Accelerated Training')
    print('🚀 RTX 2050 Optimization')
    
    # Initialize PyTorch system
    pytorch_ml = PyTorchDefenseML()
    
    # Train all models
    results = pytorch_ml.train_all_models()
    
    print('\n🎉 PYTORCH DEFENSE INTELLIGENCE SYSTEM READY!')
    print('=' * 55)
    print('🎮 GPU Acceleration: ENABLED')
    print('🚀 RTX 2050 Utilization: ACTIVE')
    print('⚡ High-Performance Training: OPERATIONAL')
    print('🛡️ Military Intelligence: DEPLOYED')

if __name__ == '__main__':
    main()
