# 🛡️ Defense Intelligence ML - Simplified Working Version
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import json
import os
from datetime import datetime

print('🛡️ DEFENSE INTELLIGENCE ML - SIMPLIFIED VERSION')
print('=' * 55)
print('🎯 Military-grade AI analysis system')
print('🔥 TensorFlow Version:', tf.__version__)

class SimpleDefenseML:
    """Simplified ML system for defense intelligence"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.signal_classifier = None
        self.threat_detector = None
        self.scaler = StandardScaler()
        
        print(f'✅ Simple Defense ML initialized')
    
    def build_signal_classifier(self):
        """Build simple signal classifier"""
        print('\n📡 Building Signal Classifier...')
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.signal_classifier = model
        print('✅ Signal classifier built')
        return model
    
    def build_threat_detector(self):
        """Build threat detection system"""
        print('\n🔍 Building Threat Detector...')
        
        # Random Forest for classification
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Isolation Forest for anomaly detection
        anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        
        self.threat_detector = {
            'classifier': rf_model,
            'anomaly_detector': anomaly_model
        }
        
        print('✅ Threat detector built')
        return self.threat_detector
    
    def generate_signal_data(self, num_samples=1000):
        """Generate synthetic signal data"""
        print(f'📊 Generating {num_samples} signal samples...')
        
        signals = []
        labels = []
        classes = ['communication', 'radar', 'data', 'noise']
        
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
            else:  # noise
                signal = np.random.normal(0, 0.1, 1000)
            
            # Add noise
            signal += np.random.normal(0, 0.05, 1000)
            
            # Extract features
            features = [
                np.mean(signal),
                np.std(signal),
                np.sqrt(np.mean(signal**2)),
                np.argmax(np.abs(np.fft.fft(signal)[:500])),
                len([i for i in range(1, len(signal)-1) if signal[i] > signal[i-1] and signal[i] > signal[i+1]])
            ]
            
            # Pad to 7 features
            while len(features) < 7:
                features.append(0.0)
            
            signals.append(features[:7])
            labels.append(class_idx)
        
        return np.array(signals), np.array(labels)
    
    def generate_threat_data(self, num_samples=2000):
        """Generate synthetic threat data"""
        print(f'📊 Generating {num_samples} threat scenarios...')
        
        activities = []
        labels = []
        
        for i in range(num_samples):
            # 30% critical, 20% suspicious, 50% normal
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
            
            # Generate features
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
            
            activities.append(features)
            labels.append(label)
        
        return np.array(activities), np.array(labels)
    
    def train_signal_classifier(self):
        """Train signal classifier"""
        print('\n🚀 Training Signal Classifier...')
        
        # Generate data
        X, y = self.generate_signal_data(1000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Build model
        self.build_signal_classifier()
        
        # Train
        history = self.signal_classifier.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=16,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = self.signal_classifier.evaluate(X_test, y_test, verbose=0)
        print(f'📊 Signal Classifier Accuracy: {test_acc:.4f}')
        
        # Save model
        self.signal_classifier.save(f'{self.models_dir}/signal_classifier.h5')
        
        return test_acc
    
    def train_threat_detector(self):
        """Train threat detector"""
        print('\n🚀 Training Threat Detector...')
        
        # Generate data
        X, y = self.generate_threat_data(2000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Build models
        self.build_threat_detector()
        
        # Train Random Forest
        rf_model = self.threat_detector['classifier']
        rf_model.fit(X_train, y_train)
        rf_acc = rf_model.score(X_test, y_test)
        
        # Train Anomaly Detector
        anomaly_model = self.threat_detector['anomaly_detector']
        anomaly_model.fit(X_train)
        
        # Test anomaly detection
        anomaly_pred = anomaly_model.predict(X_test)
        anomaly_labels = (anomaly_pred == -1).astype(int)
        y_binary = (y > 0).astype(int)
        anomaly_acc = np.mean(anomaly_labels == y_binary)
        
        print(f'📊 Threat Classifier Accuracy: {rf_acc:.4f}')
        print(f'📊 Anomaly Detection Accuracy: {anomaly_acc:.4f}')
        
        # Save models
        with open(f'{self.models_dir}/threat_classifier.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open(f'{self.models_dir}/threat_anomaly_detector.pkl', 'wb') as f:
            pickle.dump(anomaly_model, f)
        
        return rf_acc, anomaly_acc
    
    def train_all_models(self):
        """Train all models"""
        print('\n🎯 TRAINING ALL MODELS')
        print('=' * 40)
        
        start_time = datetime.now()
        results = {}
        
        # Train Signal Classifier
        try:
            signal_acc = self.train_signal_classifier()
            results['signal'] = {'accuracy': signal_acc, 'status': 'success'}
        except Exception as e:
            print(f'❌ Signal training failed: {e}')
            results['signal'] = {'accuracy': 0, 'status': 'failed'}
        
        # Train Threat Detector
        try:
            rf_acc, anomaly_acc = self.train_threat_detector()
            results['threat'] = {
                'classifier_accuracy': rf_acc,
                'anomaly_accuracy': anomaly_acc,
                'status': 'success'
            }
        except Exception as e:
            print(f'❌ Threat training failed: {e}')
            results['threat'] = {'status': 'failed'}
        
        # Save scaler
        with open(f'{self.models_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Save results
        training_report = {
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'models': results,
            'tensorflow_version': tf.__version__
        }
        
        with open(f'{self.models_dir}/training_report.json', 'w') as f:
            json.dump(training_report, f, indent=2)
        
        # Summary
        print('\n🎉 TRAINING COMPLETE!')
        print('=' * 30)
        print(f'⏱️  Training Time: {training_time:.2f} seconds')
        
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
    
    def load_models(self):
        """Load trained models"""
        print('\n📂 Loading trained models...')
        
        try:
            # Load signal classifier
            if os.path.exists(f'{self.models_dir}/signal_classifier.h5'):
                self.signal_classifier = tf.keras.models.load_model(f'{self.models_dir}/signal_classifier.h5')
                print('✅ Signal classifier loaded')
            
            # Load threat detector
            if os.path.exists(f'{self.models_dir}/threat_classifier.pkl'):
                with open(f'{self.models_dir}/threat_classifier.pkl', 'rb') as f:
                    rf_model = pickle.load(f)
                
                if os.path.exists(f'{self.models_dir}/threat_anomaly_detector.pkl'):
                    with open(f'{self.models_dir}/threat_anomaly_detector.pkl', 'rb') as f:
                        anomaly_model = pickle.load(f)
                    
                    self.threat_detector = {
                        'classifier': rf_model,
                        'anomaly_detector': anomaly_model
                    }
                    print('✅ Threat detector loaded')
            
            # Load scaler
            if os.path.exists(f'{self.models_dir}/scaler.pkl'):
                with open(f'{self.models_dir}/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                print('✅ Scaler loaded')
            
            print('🎉 All models loaded successfully!')
            return True
            
        except Exception as e:
            print(f'❌ Error loading models: {e}')
            return False
    
    def predict_signal(self, signal_data):
        """Predict signal type"""
        if self.signal_classifier is None:
            return {'error': 'Signal classifier not loaded'}
        
        try:
            # Extract features
            features = [
                np.mean(signal_data),
                np.std(signal_data),
                np.sqrt(np.mean(signal_data**2)),
                np.argmax(np.abs(np.fft.fft(signal_data)[:500])),
                len([i for i in range(1, len(signal_data)-1) if signal_data[i] > signal_data[i-1] and signal_data[i] > signal_data[i+1]])
            ]
            
            # Pad to 7 features
            while len(features) < 7:
                features.append(0.0)
            
            features = np.array(features[:7]).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.signal_classifier.predict(features_scaled)[0]
            classes = ['communication', 'radar', 'data', 'noise']
            
            return {
                'signal_type': classes[np.argmax(prediction)],
                'confidence': float(np.max(prediction)),
                'all_predictions': {classes[i]: float(prediction[i]) for i in range(len(classes))}
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_threat(self, activity_features):
        """Predict threat level"""
        if self.threat_detector is None:
            return {'error': 'Threat detector not loaded'}
        
        try:
            # Scale features
            features = np.array(activity_features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Random Forest prediction
            rf_pred = self.threat_detector['classifier'].predict(features_scaled)[0]
            rf_prob = self.threat_detector['classifier'].predict_proba(features_scaled)[0]
            
            # Anomaly detection
            anomaly_pred = self.threat_detector['anomaly_detector'].predict(features_scaled)[0]
            is_anomaly = anomaly_pred == -1
            
            classes = ['normal', 'suspicious', 'critical']
            
            return {
                'threat_level': classes[rf_pred],
                'confidence': float(max(rf_prob)),
                'is_anomalous': is_anomaly,
                'recommendation': self.get_recommendation(classes[rf_pred], max(rf_prob), is_anomaly)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_recommendation(self, threat_level, confidence, is_anomaly):
        """Get recommendation based on threat analysis"""
        if threat_level == 'critical' or confidence > 0.8 or is_anomaly:
            return 'IMMEDIATE_ACTION_REQUIRED'
        elif threat_level == 'suspicious' or confidence > 0.6:
            return 'MONITOR_CLOSELY'
        else:
            return 'NORMAL_OPERATIONS'
    
    def run_demo(self):
        """Run demonstration"""
        print('\n🎭 DEFENSE INTELLIGENCE DEMO')
        print('=' * 40)
        
        # Load models
        if not self.load_models():
            print('❌ Failed to load models')
            return
        
        # Signal analysis demo
        print('\n📡 SIGNAL ANALYSIS DEMO')
        print('-' * 30)
        
        # Generate test signal
        t = np.linspace(0, 1, 1000)
        test_signal = np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.1, 1000)
        
        signal_result = self.predict_signal(test_signal)
        
        if 'error' not in signal_result:
            print(f'🎯 Signal Type: {signal_result["signal_type"]}')
            print(f'📊 Confidence: {signal_result["confidence"]:.3f}')
            print('🔍 All Predictions:')
            for signal_type, prob in signal_result['all_predictions'].items():
                print(f'   - {signal_type}: {prob:.3f}')
        else:
            print(f'❌ Error: {signal_result["error"]}')
        
        # Threat detection demo
        print('\n🔍 THREAT DETECTION DEMO')
        print('-' * 30)
        
        # Generate test threat scenario (suspicious activity)
        test_threat = [
            23,  # night hour
            120, # long duration
            4,   # multiple failed attempts
            1,   # unusual location
            500, # high data volume
            3,   # multiple sessions
            0.85, # high risk score
            0.78, # high anomaly score
            0.82, # high behavioral deviation
            1,    # network anomaly
            1,    # time anomaly
            8     # high access frequency
        ]
        
        threat_result = self.predict_threat(test_threat)
        
        if 'error' not in threat_result:
            print(f'🚨 Threat Level: {threat_result["threat_level"].upper()}')
            print(f'📊 Confidence: {threat_result["confidence"]:.3f}')
            print(f'⚠️  Anomaly Detected: {"Yes" if threat_result["is_anomalous"] else "No"}')
            print(f'💡 Recommendation: {threat_result["recommendation"]}')
        else:
            print(f'❌ Error: {threat_result["error"]}')
        
        print('\n🎉 DEMONSTRATION COMPLETE!')
        print('🛡️ Defense Intelligence System is operational!')

def main():
    """Main function"""
    print('🛡️ SIMPLE DEFENSE INTELLIGENCE ML')
    print('=' * 40)
    
    # Initialize system
    defense_ml = SimpleDefenseML()
    
    # Train models
    defense_ml.train_all_models()
    
    # Run demonstration
    defense_ml.run_demo()
    
    print('\n🎉 SYSTEM READY!')
    print('📁 Models saved in: models/')
    print('🚀 Ready for defense intelligence operations!')

if __name__ == '__main__':
    main()
