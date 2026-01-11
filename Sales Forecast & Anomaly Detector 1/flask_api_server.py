"""
COMPLETE FLASK REST API SERVER
==============================
Full endpoints: Forecasting + Anomaly Detection + Monitoring
Run: python flask_api_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import sys

# Import the complete API backend
from streaming_api_backend import api

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)

# ============================================================================
# HOME & DOCS
# ============================================================================

@app.route('/')
def home():
    """API documentation"""
    return jsonify({
        'name': 'Walmart Complete ML System API',
        'version': '2.0',
        'features': ['Sales Forecasting', 'Anomaly Detection', 'Real-time Monitoring'],
        'endpoints': {
            'Forecasting': {
                'POST /api/forecast': 'Predict future sales',
                'GET /api/forecasts': 'Get forecast history'
            },
            'Anomaly Detection': {
                'POST /api/detect': 'Detect anomalies in sales',
                'GET /api/alerts': 'Get recent alerts'
            },
            'Monitoring': {
                'GET /api/stats': 'Dashboard statistics',
                'GET /api/live-data': 'Live monitoring data',
                'GET /api/store/<store_id>': 'Store summary'
            },
            'System': {
                'POST /api/reset': 'Reset statistics',
                'GET /api/health': 'Health check'
            }
        }
    })

# ============================================================================
# FORECASTING ENDPOINTS
# ============================================================================

@app.route('/api/forecast', methods=['POST'])
def forecast_sales():
    """
    Predict future sales
    
    Request:
    {
        "Store": 5,
        "Date": "2023-01-15",
        "Holiday_Flag": 0,
        "Temperature": 75,
        "Fuel_Price": 3.5,
        "CPI": 210,
        "Unemployment": 7.5
    }
    
    Response:
    {
        "success": true,
        "predicted_sales": 1500000,
        "confidence_interval": {"lower": 1400000, "upper": 1600000},
        "predictions": {
            "random_forest": 1520000,
            "linear_regression": 1480000
        }
    }
    """
    try:
        data = request.get_json()
        
        required = ['Store', 'Date', 'Holiday_Flag', 'Temperature', 
                   'Fuel_Price', 'CPI', 'Unemployment']
        
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        if isinstance(data['Date'], str):
            data['Date'] = datetime.fromisoformat(data['Date'].replace('Z', ''))
        
        result = api.forecast_sales(data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecasts', methods=['GET'])
def get_forecasts():
    """
    Get forecast history
    
    Query: ?limit=50
    """
    try:
        limit = int(request.args.get('limit', 50))
        forecasts = api.get_forecasts(limit=limit)
        
        return jsonify({
            'count': len(forecasts),
            'forecasts': forecasts
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ANOMALY DETECTION ENDPOINTS
# ============================================================================

@app.route('/api/detect', methods=['POST'])
def detect_anomaly():
    """
    Detect anomalies (includes forecasting)
    
    Request:
    {
        "Store": 5,
        "Date": "2023-01-15",
        "Weekly_Sales": 2500000,
        "Holiday_Flag": 0,
        "Temperature": 75,
        "Fuel_Price": 3.5,
        "CPI": 210,
        "Unemployment": 7.5
    }
    
    Response:
    {
        "is_anomaly": true,
        "alert_level": "CRITICAL",
        "confidence": 100.0,
        "detections": ["Statistical", "K-Means", "Forecast-based"],
        "forecast": {
            "predicted_sales": 1500000,
            "confidence_interval": {...}
        },
        "forecast_comparison": {
            "actual_sales": 2500000,
            "predicted_sales": 1500000,
            "error_percentage": 66.7,
            "is_forecast_anomaly": true
        }
    }
    """
    try:
        data = request.get_json()
        
        required = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 
                   'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        if isinstance(data['Date'], str):
            data['Date'] = datetime.fromisoformat(data['Date'].replace('Z', ''))
        
        result = api.process_record(data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    try:
        level = request.args.get('level', None)
        limit = int(request.args.get('limit', 50))
        
        if level and level not in ['CRITICAL', 'WARNING', 'INFO']:
            return jsonify({'error': 'Invalid level'}), 400
        
        alerts = api.get_alerts(level=level, limit=limit)
        
        return jsonify({
            'count': len(alerts),
            'alerts': alerts
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Dashboard statistics"""
    try:
        stats = api.get_dashboard_stats()
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-data', methods=['GET'])
def get_live_data():
    """Live monitoring data"""
    try:
        limit = int(request.args.get('limit', 50))
        data = api.get_live_data(limit=limit)
        
        return jsonify({
            'count': len(data),
            'data': data
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/store/<int:store_id>', methods=['GET'])
def get_store_summary(store_id):
    """Store summary"""
    try:
        summary = api.get_store_summary(store_id)
        return jsonify(summary), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.route('/api/reset', methods=['POST'])
def reset_statistics():
    """Reset statistics"""
    try:
        result = api.reset_stats()
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'forecasting': api.models.is_available('forecast_rf'),
            'anomaly_detection': api.models.is_available('kmeans') or api.models.is_available('isolation')
        }
    }), 200

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("WALMART COMPLETE ML SYSTEM API")
    print("="*80)
    print("\n🚀 Server starting on http://localhost:5000")
    print("\n📊 Features Available:")
    print("  ✓ Sales Forecasting (Random Forest + Linear Regression)")
    print("  ✓ Anomaly Detection (K-Means + Isolation Forest + Statistical)")
    print("  ✓ Forecast-based Anomaly Detection")
    print("  ✓ Real-time Monitoring")
    
    print("\n🔗 Key Endpoints:")
    print("  POST   /api/forecast      - Predict future sales")
    print("  POST   /api/detect        - Detect anomalies")
    print("  GET    /api/forecasts     - Get forecast history")
    print("  GET    /api/alerts        - Get alerts")
    print("  GET    /api/stats         - Dashboard stats")
    print("  GET    /api/live-data     - Live monitoring")
    print("  GET    /api/store/<id>    - Store summary")
    
    print("\n⚡ Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )