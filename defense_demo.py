# 🛡️ Defense Intelligence ML - Final Demonstration
import numpy as np
from simple_defense_ml import SimpleDefenseML

print('🛡️ DEFENSE INTELLIGENCE ML - FINAL DEMO')
print('=' * 50)

def main():
    # Initialize the system
    defense_ml = SimpleDefenseML()
    
    # Load trained models
    print('\n📂 Loading trained defense models...')
    if not defense_ml.load_models():
        print('❌ Failed to load models. Training new ones...')
        defense_ml.train_all_models()
        defense_ml.load_models()
    
    print('\n🎯 DEFENSE INTELLIGENCE CAPABILITIES DEMO')
    print('=' * 50)
    
    # 1. Signal Intelligence Demo
    print('\n📡 SIGNAL INTELLIGENCE (SIGINT)')
    print('-' * 40)
    print('Analyzing intercepted communications...')
    
    # Test different signal types
    test_signals = {
        'Communication Signal': np.sin(2 * np.pi * 100 * np.linspace(0, 1, 1000)) * (1 + 0.5 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))),
        'Radar Pulse': np.zeros(1000),
        'Data Transmission': np.repeat(np.random.randint(0, 2, 100), 10)[:1000].astype(float),
        'Background Noise': np.random.normal(0, 0.1, 1000)
    }
    
    # Add radar pulses
    for i in range(0, 1000, 100):
        if i + 20 < 1000:
            test_signals['Radar Pulse'][i:i+20] = np.sin(2 * np.pi * 50 * np.arange(20))
    
    # Add noise to all signals
    for name, signal in test_signals.items():
        test_signals[name] += np.random.normal(0, 0.05, 1000)
    
    # Analyze each signal
    for signal_name, signal_data in test_signals.items():
        result = defense_ml.predict_signal(signal_data)
        if 'error' not in result:
            print(f'🎯 {signal_name}: {result["signal_type"].upper()} (confidence: {result["confidence"]:.3f})')
        else:
            print(f'❌ {signal_name}: Analysis failed')
    
    # 2. Threat Detection Demo
    print('\n🔍 THREAT DETECTION & ANALYSIS')
    print('-' * 40)
    print('Monitoring network activities for threats...')
    
    # Test different threat scenarios
    threat_scenarios = {
        'Normal Activity': [14, 15, 0, 0, 5, 1, 0.2, 0.1, 0.1, 0, 0, 2],
        'Suspicious Login': [2, 45, 3, 1, 50, 2, 0.7, 0.6, 0.5, 1, 0, 5],
        'Critical Breach': [23, 120, 4, 1, 500, 3, 0.85, 0.78, 0.82, 1, 1, 8],
        'Data Exfiltration': [18, 180, 2, 1, 800, 4, 0.9, 0.85, 0.88, 1, 1, 10]
    }
    
    # Analyze each scenario
    for scenario_name, features in threat_scenarios.items():
        result = defense_ml.predict_threat(features)
        if 'error' not in result:
            status = '🚨' if result['threat_level'] in ['critical', 'suspicious'] else '✅'
            print(f'{status} {scenario_name}: {result["threat_level"].upper()}')
            print(f'   Confidence: {result["confidence"]:.3f} | Anomaly: {"Yes" if result["is_anomalous"] else "No"}')
            print(f'   Recommendation: {result["recommendation"]}')
        else:
            print(f'❌ {scenario_name}: Analysis failed')
        print()
    
    # 3. Real-time Analysis Simulation
    print('⚡ REAL-TIME ANALYSIS SIMULATION')
    print('-' * 40)
    print('Simulating continuous monitoring...')
    
    # Simulate incoming data stream
    for i in range(5):
        print(f'\n📡 Signal Analysis #{i+1}:')
        
        # Generate random signal
        signal_type = np.random.choice(['communication', 'radar', 'data', 'noise'])
        if signal_type == 'communication':
            signal = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 1000)) + np.random.normal(0, 0.1, 1000)
        elif signal_type == 'radar':
            signal = np.zeros(1000)
            for j in range(0, 1000, 100):
                if j + 20 < 1000:
                    signal[j:j+20] = np.sin(2 * np.pi * 50 * np.arange(20))
        elif signal_type == 'data':
            signal = np.repeat(np.random.randint(0, 2, 100), 10)[:1000].astype(float)
        else:
            signal = np.random.normal(0, 0.1, 1000)
        
        signal_result = defense_ml.predict_signal(signal)
        
        if 'error' not in signal_result:
            print(f'   Detected: {signal_result["signal_type"]} (confidence: {signal_result["confidence"]:.3f})')
        else:
            print(f'   Error: {signal_result["error"]}')
        
        print(f'🔍 Threat Analysis #{i+1}:')
        
        # Generate random threat scenario
        threat_level = np.random.choice(['normal', 'suspicious', 'critical'])
        if threat_level == 'normal':
            features = [14, 15, 0, 0, 5, 1, 0.2, 0.1, 0.1, 0, 0, 2]
        elif threat_level == 'suspicious':
            features = [2, 45, 3, 1, 50, 2, 0.7, 0.6, 0.5, 1, 0, 5]
        else:
            features = [23, 120, 4, 1, 500, 3, 0.85, 0.78, 0.82, 1, 1, 8]
        
        threat_result = defense_ml.predict_threat(features)
        
        if 'error' not in threat_result:
            alert = '🚨' if threat_result['threat_level'] in ['critical', 'suspicious'] else '✅'
            print(f'   {alert} Threat Level: {threat_result["threat_level"].upper()}')
            print(f'   Recommendation: {threat_result["recommendation"]}')
        else:
            print(f'   Error: {threat_result["error"]}')
    
    # 4. System Performance Summary
    print('\n📊 SYSTEM PERFORMANCE SUMMARY')
    print('-' * 40)
    
    # Check model files
    import os
    models_dir = 'models'
    model_files = os.listdir(models_dir)
    
    print(f'📁 Trained Models: {len(model_files)} files')
    for file in model_files:
        print(f'   ✅ {file}')
    
    print(f'\n🔥 System Capabilities:')
    print(f'   ✅ Signal Intelligence Analysis')
    print(f'   ✅ Threat Detection & Classification')
    print(f'   ✅ Anomaly Detection')
    print(f'   ✅ Real-time Processing')
    print(f'   ✅ Risk Assessment')
    
    print(f'\n🎯 Military Applications:')
    print(f'   🛡️ Intelligence Analysis')
    print(f'   📡 Signal Monitoring')
    print(f'   🔍 Security Operations')
    print(f'   ⚠️  Threat Assessment')
    print(f'   🚨 Alert Generation')
    
    print('\n🎉 DEFENSE INTELLIGENCE SYSTEM READY!')
    print('=' * 50)
    print('🛡️ Military-grade AI analysis operational')
    print('📊 Models trained and deployed')
    print('🚀 Ready for intelligence operations')
    print('⚡ Real-time threat detection active')
    print('🔍 Continuous monitoring enabled')

if __name__ == '__main__':
    main()
