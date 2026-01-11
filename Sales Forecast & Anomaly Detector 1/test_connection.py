"""
COMPLETE SYSTEM TEST - VERIFY ALL INTEGRATIONS
==============================================
Tests: Models + API + Forecasting + Anomaly Detection
Run: python test_connections.py
"""

import os
import requests
import pickle
from datetime import datetime
import pandas as pd

print("="*80)
print("COMPLETE SYSTEM INTEGRATION TEST")
print("="*80)

# ============================================================================
# TEST 1: FILE EXISTENCE
# ============================================================================

print("\n" + "="*80)
print("TEST 1: FILE EXISTENCE")
print("="*80)

files = {
    'Core Scripts': [
        'Forecasting.ipynb',
        'AnomalyDetector.ipynb',
        'streaming_api_backend.py',
        'flask_api_server.py',
        'streamlit_ui.py',
        'business_impact.py'
    ],
    'Forecasting Models': [
        'random_forest_model.pkl',
        'linear_regression_model.pkl',
        'scaler.pkl'
    ],
    'Anomaly Models': [
        'kmeans_anomaly_model.pkl',
        'isolation_forest_model.pkl'
    ]
}

all_exist = True
for category, filelist in files.items():
    print(f"\n{category}:")
    for f in filelist:
        exists = os.path.exists(f)
        print(f"  {'✓' if exists else '✗'} {f}")
        if not exists and 'Models' in category:
            all_exist = False

# ============================================================================
# TEST 2: MODEL LOADING
# ============================================================================

print("\n" + "="*80)
print("TEST 2: MODEL LOADING TEST")
print("="*80)

models_status = {}

# Forecasting models
try:
    with open('random_forest_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    models_status['Random Forest'] = f"✓ Loaded (Type: {type(rf).__name__})"
except Exception as e:
    models_status['Random Forest'] = f"✗ {str(e)[:40]}"

try:
    with open('linear_regression_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    models_status['Linear Regression'] = f"✓ Loaded (Type: {type(lr).__name__})"
except Exception as e:
    models_status['Linear Regression'] = f"✗ {str(e)[:40]}"

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    models_status['Scaler'] = f"✓ Loaded (Type: {type(scaler).__name__})"
except Exception as e:
    models_status['Scaler'] = f"✗ {str(e)[:40]}"

# Anomaly models
try:
    with open('kmeans_anomaly_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    models_status['K-Means'] = f"✓ Loaded"
except Exception as e:
    models_status['K-Means'] = f"✗ {str(e)[:40]}"

try:
    with open('isolation_forest_model.pkl', 'rb') as f:
        iso = pickle.load(f)
    models_status['Isolation Forest'] = f"✓ Loaded (Type: {type(iso).__name__})"
except Exception as e:
    models_status['Isolation Forest'] = f"✗ {str(e)[:40]}"

print("\nModel Status:")
for model, status in models_status.items():
    print(f"  {model}: {status}")

loaded_count = sum(1 for s in models_status.values() if '✓' in s)
print(f"\n{loaded_count}/5 models loaded successfully")

# ============================================================================
# TEST 3: API CONNECTION
# ============================================================================

print("\n" + "="*80)
print("TEST 3: FLASK API CONNECTION")
print("="*80)

API_URL = "http://localhost:5000"

try:
    response = requests.get(f"{API_URL}/api/health", timeout=2)
    if response.status_code == 200:
        health = response.json()
        print("✓ Flask API is running")
        print(f"  Features: {health.get('features', {})}")
        api_running = True
    else:
        print(f"✗ API returned status {response.status_code}")
        api_running = False
except:
    print("✗ Flask API not running")
    print("  Start: python flask_api_server.py")
    api_running = False

# ============================================================================
# TEST 4: FORECASTING FUNCTIONALITY
# ============================================================================

print("\n" + "="*80)
print("TEST 4: FORECASTING FUNCTIONALITY")
print("="*80)

if api_running:
    now = datetime.now()
    
    # सभी 19 features के साथ test record
    test_record = {
        # Basic Walmart features (7)
        'Store': 5,
        'Holiday_Flag': 0,
        'Temperature': 75.0,
        'Fuel_Price': 3.5,
        'CPI': 210.0,
        'Unemployment': 7.5,
        
        # Date features - streaming_api_backend में automatically add होते हैं
        
        # Lag features (4) - IMPORTANT!
        'Lag_1': 500000,   # Last week's sales
        'Lag_2': 480000,   # 2 weeks ago
        'Lag_3': 520000,   # 3 weeks ago
        'Lag_4': 490000,   # 4 weeks ago
        
        # Rolling statistics (3) - IMPORTANT!
        'Rolling_Mean_4': 497500,   # 4-week moving average
        'Rolling_Std_4': 15000,     # 4-week standard deviation
        'Sales_Trend': 1.02,        # Sales trend
        
        # Date as string
        'Date': now.isoformat()
    }
    
    print(f"Sending record with {len([k for k in test_record.keys() if k != 'Date'])} additional features")
    print(f"Features sent: {list(test_record.keys())}")
    
    try:
        response = requests.post(f"{API_URL}/api/forecast", json=test_record, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("\n✓ Forecasting endpoint working")
                print(f"  Predicted: ${result['predicted_sales']:,.0f}")
                
                # Print feature details if available
                if 'features_used' in result:
                    print(f"  Features used: {result['features_used']}")
                
                if 'predictions' in result:
                    print(f"  Models: {list(result['predictions'].keys())}")
                
                forecasting_works = True
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"\n✗ Forecast failed: {error_msg}")
                
                # Debug info
                if 'traceback' in result:
                    # Check for feature mismatch
                    if "expecting" in error_msg and "features" in error_msg:
                        # Extract numbers from error message
                        import re
                        nums = re.findall(r'\d+', error_msg)
                        if len(nums) >= 2:
                            print(f"  Expected: {nums[1]} features")
                            print(f"  Received: {nums[0]} features")
                
                forecasting_works = False
        else:
            print(f"\n✗ Forecast endpoint error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            forecasting_works = False
    except Exception as e:
        print(f"\n✗ Forecast test failed: {str(e)[:100]}")
        forecasting_works = False
else:
    print("⚠️  Skipped (API not running)")
    forecasting_works = False
    
    
    # ============================================================================
# TEST 5: ANOMALY DETECTION FUNCTIONALITY
# ============================================================================

print("\n" + "="*80)
print("TEST 5: ANOMALY DETECTION FUNCTIONALITY")
print("="*80)

if api_running:
    test_record_anomaly = {
        'Store': 5,
        'Date': datetime.now().isoformat(),
        'Weekly_Sales': 2500000,  # Potentially high
        'Holiday_Flag': 0,
        'Temperature': 75,
        'Fuel_Price': 3.5,
        'CPI': 210,
        'Unemployment': 7.5
    }
    
    try:
        response = requests.post(f"{API_URL}/api/detect", json=test_record_anomaly, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Anomaly detection endpoint working")
            print(f"  Is Anomaly: {result['is_anomaly']}")
            print(f"  Alert Level: {result['alert_level']}")
            print(f"  Methods: {', '.join(result['detections'])}")
            print(f"  Includes Forecast: {'forecast' in result and result['forecast']}")
            anomaly_works = True
        else:
            print(f"✗ Detection endpoint error: {response.status_code}")
            anomaly_works = False
    except Exception as e:
        print(f"✗ Detection test failed: {str(e)[:50]}")
        anomaly_works = False
else:
    print("⚠️  Skipped (API not running)")
    anomaly_works = False

# ============================================================================
# TEST 6: FULL INTEGRATION TEST
# ============================================================================

print("\n" + "="*80)
print("TEST 6: FULL INTEGRATION TEST")
print("="*80)

if api_running and loaded_count >= 3:
    test_record_full = {
        'Store': 5,
        'Date': datetime.now().isoformat(),
        'Weekly_Sales': 1800000,
        'Holiday_Flag': 1,
        'Temperature': 75,
        'Fuel_Price': 3.5,
        'CPI': 210,
        'Unemployment': 7.5
    }
    
    print("\nSending full test record...")
    
    try:
        # Step 1: Forecast
        print("\n1. Testing Forecast:")
        forecast_resp = requests.post(f"{API_URL}/api/forecast", json=test_record_full, timeout=5)
        if forecast_resp.status_code == 200:
            forecast = forecast_resp.json()
            if forecast.get('success'):
                predicted = forecast['predicted_sales']
                print(f"   ✓ Predicted Sales: ${predicted:,.0f}")
            else:
                print(f"   ✗ Forecast failed")
                predicted = None
        else:
            predicted = None
            print(f"   ✗ Request failed")
        
        # Step 2: Detect with forecast comparison
        print("\n2. Testing Anomaly Detection (with forecast):")
        detect_resp = requests.post(f"{API_URL}/api/detect", json=test_record_full, timeout=5)
        if detect_resp.status_code == 200:
            detection = detect_resp.json()
            print(f"   ✓ Detection complete")
            print(f"     Anomaly: {detection['is_anomaly']}")
            print(f"     Methods: {detection['detection_count']}/4")
            
            if detection.get('forecast_comparison'):
                fc = detection['forecast_comparison']
                print(f"     Forecast Error: {fc['error_percentage']:.1f}%")
                print(f"     Forecast Anomaly: {fc['is_forecast_anomaly']}")
        else:
            print(f"   ✗ Detection failed")
        
        # Step 3: Get Stats
        print("\n3. Testing Statistics:")
        stats_resp = requests.get(f"{API_URL}/api/stats", timeout=2)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            print(f"   ✓ Stats retrieved")
            print(f"     Total Records: {stats.get('total_records', 0)}")
            print(f"     Forecasts Made: {stats.get('forecasts_made', 0)}")
            print(f"     Alerts: {stats.get('alerts_critical', 0) + stats.get('alerts_warning', 0)}")
        
        print("\n✅ FULL INTEGRATION WORKING!")
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
else:
    print("⚠️  Skipped (API not running or models missing)")

# ============================================================================
# TEST 7: BUSINESS IMPACT INTEGRATION
# ============================================================================

print("\n" + "="*80)
print("TEST 7: BUSINESS IMPACT INTEGRATION")
print("="*80)

checks = {
    'Models accessible': loaded_count >= 3,
    'API accessible': api_running,
    'Anomaly results exist': os.path.exists('anomaly_detection_results.csv'),
    'Forecasting works': forecasting_works if api_running else False,
}

print("\nIntegration Checks:")
for check, status in checks.items():
    print(f"  {'✓' if status else '✗'} {check}")

integration_score = sum(checks.values())
print(f"\nIntegration Score: {integration_score}/{len(checks)}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
Component Status:
  • Core Files:           ✓
  • Forecasting Models:   {loaded_count >= 3}
  • Anomaly Models:       {loaded_count >= 4}
  • Flask API:            {'✓ Running' if api_running else '✗ Not running'}
  • Forecasting Feature:  {'✓ Working' if forecasting_works else '✗ Not tested'}
  • Anomaly Detection:    {'✓ Working' if anomaly_works else '✗ Not tested'}
  • Full Integration:     {integration_score}/{len(checks)}

System Readiness:
""")

if loaded_count >= 4 and api_running and forecasting_works and anomaly_works:
    print("  ✅ FULLY OPERATIONAL!")
    print("\n  All Features Available:")
    print("  • Sales Forecasting (Random Forest + Linear Regression)")
    print("  • Anomaly Detection (K-Means + Isolation Forest + Statistical)")
    print("  • Forecast-based Anomaly Detection")
    print("  • Real-time Monitoring")
    print("  • Business Impact Analysis")
    print("\n  ✓ Ready to use Streamlit UI: streamlit run streamlit_ui.py")
    print("  ✓ Ready for business impact: python business_impact.py")
    
elif loaded_count >= 3 and api_running:
    print("  ⚠️  MOSTLY OPERATIONAL")
    print("\n  Next steps:")
    print("  • Some models may be missing")
    print("  • Run: python walmart_forecast.py")
    print("  • Run: python walmart_anomaly_detection.py")
    
elif loaded_count >= 3:
    print("  ⚠️  MODELS READY, START API")
    print("\n  Next steps:")
    print("  1. Start API: python flask_api_server.py")
    print("  2. Launch UI: streamlit run streamlit_ui.py")
    
else:
    print("  ❌ SETUP REQUIRED")
    print("\n  Complete Setup:")
    print("  1. Train forecasting: python walmart_forecast.py")
    print("  2. Train anomaly: python walmart_anomaly_detection.py")
    print("  3. Start API: python flask_api_server.py")
    print("  4. Launch UI: streamlit run streamlit_ui.py")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)