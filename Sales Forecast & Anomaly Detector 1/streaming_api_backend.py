"""
COMPLETE STREAMING API BACKEND - WITH FORECASTING + ANOMALY DETECTION
====================================================================
Full integration: Forecasting + Anomaly Detection + Real-time monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration"""
    ALERT_THRESHOLDS = {
        'deviation_threshold': 3.0,
        'wow_change_threshold': 0.5,
        'consensus_required': 2,
        'critical_deviation': 5.0,
        'forecast_error_threshold': 0.30,  # 30% difference from forecast = anomaly
    }
    
    ALERT_LEVELS = {
        'INFO': {'priority': 1, 'color': '#4CAF50', 'icon': '🟢'},
        'WARNING': {'priority': 2, 'color': '#FF9800', 'icon': '🟡'},
        'CRITICAL': {'priority': 3, 'color': '#F44336', 'icon': '🔴'}
    }
    
    MODEL_PATHS = {
        'forecast_rf': 'random_forest_model.pkl',
        'forecast_lr': 'linear_regression_model.pkl',
        'forecast_scaler': 'scaler.pkl',
        'kmeans': 'kmeans_anomaly_model.pkl',
        'isolation': 'isolation_forest_model.pkl',
    }

# ============================================================================
# MODEL MANAGER - LOADS ALL MODELS
# ============================================================================

class ModelManager:
    """Load and manage ALL ML models (Forecasting + Anomaly)"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        
        # 1. FORECASTING MODELS
        try:
            with open(Config.MODEL_PATHS['forecast_rf'], 'rb') as f:
                self.models['forecast_rf'] = pickle.load(f)
            print("✓ Random Forest forecasting model loaded")
        except:
            self.models['forecast_rf'] = None
            print("⚠ Random Forest model not found")
        
        try:
            with open(Config.MODEL_PATHS['forecast_lr'], 'rb') as f:
                self.models['forecast_lr'] = pickle.load(f)
            print("✓ Linear Regression forecasting model loaded")
        except:
            self.models['forecast_lr'] = None
            print("⚠ Linear Regression model not found")
        
        try:
            with open(Config.MODEL_PATHS['forecast_scaler'], 'rb') as f:
                self.models['forecast_scaler'] = pickle.load(f)
            print("✓ Feature scaler loaded")
        except:
            self.models['forecast_scaler'] = None
            print("⚠ Feature scaler not found")
        
        # 2. ANOMALY DETECTION MODELS
        try:
            with open(Config.MODEL_PATHS['kmeans'], 'rb') as f:
                kmeans_data = pickle.load(f)
                self.models['kmeans'] = kmeans_data['model']
                self.models['kmeans_scaler'] = kmeans_data['scaler']
                self.models['kmeans_threshold'] = kmeans_data['threshold']
            print("✓ K-Means anomaly model loaded")
        except:
            self.models['kmeans'] = None
            print("⚠ K-Means model not found")
        
        try:
            with open(Config.MODEL_PATHS['isolation'], 'rb') as f:
                self.models['isolation'] = pickle.load(f)
            print("✓ Isolation Forest model loaded")
        except:
            self.models['isolation'] = None
            print("⚠ Isolation Forest model not found")
    
    def is_available(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return self.models.get(model_name) is not None

# ============================================================================
# DATA STORE
# ============================================================================

class DataStore:
    """In-memory data storage"""
    
    def __init__(self):
        self.historical_data = self.load_historical_data()
        self.store_baselines = self.calculate_baselines()
        self.recent_sales = {}
        self.alerts = []
        self.forecasts = []  # Store forecast history
        self.monitoring_stats = {
            'total_records': 0,
            'forecasts_made': 0,
            'alerts_info': 0,
            'alerts_warning': 0,
            'alerts_critical': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'forecast_accuracy': 0
        }
        self.live_data = []
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical data"""
        try:
            df = pd.read_csv('walmart.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except:
            dates = pd.date_range(start='2010-02-05', end='2012-10-26', freq='W')
            np.random.seed(42)
            return pd.DataFrame({
                'Store': np.random.choice(range(1, 46), len(dates)),
                'Date': dates,
                'Weekly_Sales': np.random.uniform(200000, 3000000, len(dates)),
                'Holiday_Flag': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
                'Temperature': np.random.uniform(20, 100, len(dates)),
                'Fuel_Price': np.random.uniform(2.5, 4.5, len(dates)),
                'CPI': np.random.uniform(125, 230, len(dates)),
                'Unemployment': np.random.uniform(3, 15, len(dates))
            })
    
    def calculate_baselines(self) -> Dict:
        """Calculate baseline statistics per store"""
        baselines = {}
        for store in self.historical_data['Store'].unique():
            store_data = self.historical_data[self.historical_data['Store'] == store]
            baselines[store] = {
                'mean': float(store_data['Weekly_Sales'].mean()),
                'std': float(store_data['Weekly_Sales'].std()),
                'min': float(store_data['Weekly_Sales'].min()),
                'max': float(store_data['Weekly_Sales'].max())
            }
        return baselines
    
    def add_recent_sale(self, store: int, sales: float):
        """Add to rolling window"""
        if store not in self.recent_sales:
            self.recent_sales[store] = []
        
        self.recent_sales[store].append(sales)
        
        if len(self.recent_sales[store]) > 4:
            self.recent_sales[store].pop(0)
    
    def get_recent_sales(self, store: int, n: int = 4) -> List[float]:
        """Get recent sales for a store"""
        return self.recent_sales.get(store, [])[-n:]

# ============================================================================
# SALES FORECASTER - NEW!
# ============================================================================

class SalesForecaster:
    """Sales forecasting using trained models"""
    
    def __init__(self, model_manager: ModelManager, data_store: DataStore):
        self.models = model_manager
        self.data = data_store
    
    def prepare_features(self, record: Dict, include_lags: bool = False) -> np.ndarray:
        """Prepare features for forecasting"""
        features = [
            record.get('Store', 1),
            record.get('Holiday_Flag', 0),
            record.get('Temperature', 70),
            record.get('Fuel_Price', 3.5),
            record.get('CPI', 200),
            record.get('Unemployment', 7)
        ]
        
        # Add time features if date provided
        if 'Date' in record:
            date = pd.to_datetime(record['Date'])
            features.extend([
                date.year,
                date.month,
                date.isocalendar()[1],  # week
                date.day,
                date.dayofweek,
                date.quarter
            ])
        else:
            features.extend([2023, 1, 1, 1, 0, 1])  # default
        
        # Add lag features if available
        if include_lags:
            store = record.get('Store', 1)
            recent = self.data.get_recent_sales(store, 4)
            
            if len(recent) >= 1:
                features.append(recent[-1])
            else:
                features.append(self.data.store_baselines.get(store, {}).get('mean', 500000))
            
            if len(recent) >= 2:
                features.append(recent[-2])
            else:
                features.append(self.data.store_baselines.get(store, {}).get('mean', 500000))
            
            if len(recent) >= 4:
                features.append(recent[-4])
            else:
                features.append(self.data.store_baselines.get(store, {}).get('mean', 500000))
            
            # Rolling mean
            if len(recent) > 0:
                features.append(np.mean(recent))
            else:
                features.append(self.data.store_baselines.get(store, {}).get('mean', 500000))
            
            # Rolling std
            if len(recent) > 1:
                features.append(np.std(recent))
            else:
                features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def forecast_sales(self, record: Dict) -> Dict:
        """Forecast sales for given record"""
        
        # Check if forecasting models available
        if not self.models.is_available('forecast_rf'):
            return {
                'success': False,
                'error': 'Forecasting models not available',
                'predicted_sales': None,
                'confidence_interval': None
            }
        
        try:
            # Prepare features
            X = self.prepare_features(record, include_lags=True)
            
            # Scale features
            if self.models.is_available('forecast_scaler'):
                X_scaled = self.models.models['forecast_scaler'].transform(X)
            else:
                X_scaled = X
            
            # Get predictions from both models
            predictions = {}
            
            if self.models.is_available('forecast_rf'):
                pred_rf = self.models.models['forecast_rf'].predict(X_scaled)[0]
                predictions['random_forest'] = float(pred_rf)
            
            if self.models.is_available('forecast_lr'):
                pred_lr = self.models.models['forecast_lr'].predict(X_scaled)[0]
                predictions['linear_regression'] = float(pred_lr)
            
            # Average prediction
            if predictions:
                predicted_sales = np.mean(list(predictions.values()))
                std_dev = np.std(list(predictions.values())) if len(predictions) > 1 else predicted_sales * 0.1
                
                result = {
                    'success': True,
                    'predicted_sales': float(predicted_sales),
                    'predictions': predictions,
                    'confidence_interval': {
                        'lower': float(predicted_sales - 1.96 * std_dev),
                        'upper': float(predicted_sales + 1.96 * std_dev)
                    },
                    'store': record.get('Store', 1),
                    'date': record.get('Date', datetime.now().isoformat())
                }
                
                # Store forecast
                self.data.forecasts.append(result)
                self.data.monitoring_stats['forecasts_made'] += 1
                
                return result
            else:
                return {
                    'success': False,
                    'error': 'No models available for prediction'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_with_forecast(self, record: Dict, forecast: Dict) -> Dict:
        """Compare actual sales with forecast"""
        
        if not forecast.get('success'):
            return None
        
        actual = record.get('Weekly_Sales', 0)
        predicted = forecast['predicted_sales']
        
        error = actual - predicted
        error_pct = abs(error) / predicted * 100 if predicted > 0 else 0
        
        is_anomaly = error_pct > Config.ALERT_THRESHOLDS['forecast_error_threshold'] * 100
        
        return {
            'actual_sales': float(actual),
            'predicted_sales': float(predicted),
            'error': float(error),
            'error_percentage': float(error_pct),
            'is_forecast_anomaly': is_anomaly,
            'within_confidence': (
                forecast['confidence_interval']['lower'] <= actual <= 
                forecast['confidence_interval']['upper']
            )
        }

# ============================================================================
# ANOMALY DETECTOR (Enhanced with Forecast)
# ============================================================================

class AnomalyDetector:
    """Enhanced anomaly detection with forecasting"""
    
    def __init__(self, model_manager: ModelManager, data_store: DataStore, forecaster: SalesForecaster):
        self.models = model_manager
        self.data = data_store
        self.forecaster = forecaster
        self.config = Config.ALERT_THRESHOLDS
    
    def calculate_features(self, record: Dict) -> Dict:
        """Calculate detection features"""
        store = record.get('Store', 1)
        sales = record.get('Weekly_Sales', 0)
        
        self.data.add_recent_sale(store, sales)
        
        baseline = self.data.store_baselines.get(store, {
            'mean': sales,
            'std': sales * 0.2
        })
        
        features = {}
        features['z_score'] = (sales - baseline['mean']) / (baseline['std'] + 1e-8)
        
        recent = self.data.get_recent_sales(store)
        if len(recent) >= 2:
            rolling_mean = np.mean(recent[:-1])
            rolling_std = np.std(recent[:-1])
            features['deviation'] = (sales - rolling_mean) / (rolling_std + 1e-8)
            features['wow_change'] = (sales - recent[-2]) / recent[-2]
        else:
            features['deviation'] = 0
            features['wow_change'] = 0
        
        return features
    
    def detect_statistical(self, features: Dict) -> tuple:
        """Statistical detection"""
        score = 0
        reasons = []
        
        if abs(features['z_score']) > self.config['deviation_threshold']:
            score += 1
            reasons.append({
                'method': 'Statistical',
                'metric': 'Z-Score',
                'value': round(features['z_score'], 2),
                'threshold': self.config['deviation_threshold']
            })
        
        if abs(features['deviation']) > self.config['deviation_threshold']:
            score += 1
            reasons.append({
                'method': 'Statistical',
                'metric': 'Deviation',
                'value': round(features['deviation'], 2),
                'threshold': self.config['deviation_threshold']
            })
        
        if abs(features['wow_change']) > self.config['wow_change_threshold']:
            score += 1
            reasons.append({
                'method': 'Statistical',
                'metric': 'WoW Change',
                'value': round(features['wow_change'] * 100, 1),
                'threshold': self.config['wow_change_threshold'] * 100
            })
        
        return score >= 1, score, reasons
    
    def detect_kmeans(self, record: Dict) -> tuple:
        """K-Means detection"""
        if not self.models.is_available('kmeans'):
            return False, []
        
        X = np.array([[
            record.get('Weekly_Sales', 0),
            record.get('Temperature', 70),
            record.get('Fuel_Price', 3.5),
            record.get('CPI', 200),
            record.get('Unemployment', 7),
            0, 0, 0
        ]])
        
        X_scaled = self.models.models['kmeans_scaler'].transform(X)
        distance = np.min(np.linalg.norm(
            X_scaled - self.models.models['kmeans'].cluster_centers_, 
            axis=1
        ))
        
        threshold = self.models.models['kmeans_threshold']
        is_anomaly = distance > threshold
        
        reason = {
            'method': 'K-Means',
            'metric': 'Distance',
            'value': round(float(distance), 2),
            'threshold': round(float(threshold), 2)
        } if is_anomaly else None
        
        return is_anomaly, [reason] if reason else []
    
    def detect_isolation_forest(self, record: Dict) -> tuple:
        """Isolation Forest detection"""
        if not self.models.is_available('isolation'):
            return False, []
        
        X = np.array([[
            record.get('Weekly_Sales', 0),
            record.get('Temperature', 70),
            record.get('Fuel_Price', 3.5),
            record.get('CPI', 200),
            record.get('Unemployment', 7),
            0, 0, 0
        ]])
        
        prediction = self.models.models['isolation'].predict(X)[0]
        score = self.models.models['isolation'].score_samples(X)[0]
        
        is_anomaly = prediction == -1
        
        reason = {
            'method': 'Isolation Forest',
            'metric': 'Score',
            'value': round(float(score), 3),
            'threshold': 0
        } if is_anomaly else None
        
        return is_anomaly, [reason] if reason else []
    
    def detect(self, record: Dict) -> Dict:
        """Complete detection with forecasting"""
        
        # Step 1: Generate forecast
        forecast = self.forecaster.forecast_sales(record)
        
        # Step 2: Calculate features
        features = self.calculate_features(record)
        
        # Step 3: Statistical detection
        stat_anomaly, stat_score, stat_reasons = self.detect_statistical(features)
        
        # Step 4: ML model detection
        kmeans_anomaly, kmeans_reasons = self.detect_kmeans(record)
        isolation_anomaly, isolation_reasons = self.detect_isolation_forest(record)
        
        # Step 5: Forecast-based detection
        forecast_comparison = None
        forecast_anomaly = False
        if forecast.get('success') and 'Weekly_Sales' in record:
            forecast_comparison = self.forecaster.compare_with_forecast(record, forecast)
            if forecast_comparison:
                forecast_anomaly = forecast_comparison['is_forecast_anomaly']
                if forecast_anomaly:
                    stat_reasons.append({
                        'method': 'Forecast',
                        'metric': 'Prediction Error',
                        'value': round(forecast_comparison['error_percentage'], 1),
                        'threshold': self.config['forecast_error_threshold'] * 100
                    })
        
        # Consensus
        detections = []
        if stat_anomaly:
            detections.append('Statistical')
        if kmeans_anomaly:
            detections.append('K-Means')
        if isolation_anomaly:
            detections.append('Isolation Forest')
        if forecast_anomaly:
            detections.append('Forecast-based')
        
        total_detections = len(detections)
        is_anomaly = total_detections >= self.config['consensus_required']
        
        alert_level = self.get_alert_level(features, total_detections)
        
        all_reasons = stat_reasons + kmeans_reasons + isolation_reasons
        
        result = {
            'is_anomaly': is_anomaly,
            'alert_level': alert_level,
            'confidence': round(total_detections / 4 * 100, 1),  # Out of 4 methods
            'detections': detections,
            'detection_count': total_detections,
            'reasons': all_reasons,
            'features': {
                'z_score': round(features['z_score'], 2),
                'deviation': round(features['deviation'], 2),
                'wow_change': round(features['wow_change'] * 100, 1)
            },
            'forecast': forecast if forecast.get('success') else None,
            'forecast_comparison': forecast_comparison,
            'record': {
                'store': record.get('Store', 1),
                'date': record.get('Date', datetime.now().isoformat()),
                'sales': round(record.get('Weekly_Sales', 0), 2),
                'holiday': bool(record.get('Holiday_Flag', 0))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def get_alert_level(self, features: Dict, detection_count: int) -> str:
        """Determine alert severity"""
        if abs(features.get('z_score', 0)) > self.config['critical_deviation']:
            return 'CRITICAL'
        elif detection_count >= 2:
            return 'WARNING'
        else:
            return 'INFO'

# ============================================================================
# STREAMING API
# ============================================================================

class StreamingAPI:
    """Complete API with forecasting + anomaly detection"""
    
    def __init__(self):
        print("Initializing Streaming API...")
        self.models = ModelManager()
        self.data = DataStore()
        self.forecaster = SalesForecaster(self.models, self.data)
        self.detector = AnomalyDetector(self.models, self.data, self.forecaster)
        print("✓ API initialized with forecasting + anomaly detection")
    
    def forecast_sales(self, record: Dict) -> Dict:
        """Forecast sales for given record"""
        return self.forecaster.forecast_sales(record)
    
    def process_record(self, record: Dict) -> Dict:
        """Process record: forecast + detect anomalies"""
        result = self.detector.detect(record)
        
        if result['is_anomaly']:
            self.data.alerts.append(result)
            level = result['alert_level'].lower()
            self.data.monitoring_stats[f'alerts_{level}'] += 1
        
        self.data.live_data.append(result)
        if len(self.data.live_data) > 100:
            self.data.live_data.pop(0)
        
        self.data.monitoring_stats['total_records'] += 1
        
        return result
    
    def get_alerts(self, level: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        alerts = self.data.alerts
        
        if level:
            alerts = [a for a in alerts if a['alert_level'] == level]
        
        return alerts[-limit:]
    
    def get_forecasts(self, limit: int = 50) -> List[Dict]:
        """Get recent forecasts"""
        return self.data.forecasts[-limit:]
    
    def get_dashboard_stats(self) -> Dict:
        """Get statistics"""
        stats = self.data.monitoring_stats.copy()
        
        total = stats['total_records']
        if total > 0:
            stats['alert_rate'] = round(
                (stats['alerts_critical'] + stats['alerts_warning'] + stats['alerts_info']) / total * 100, 
                2
            )
        else:
            stats['alert_rate'] = 0
        
        stats['recent_alerts'] = len([a for a in self.data.alerts[-10:]])
        
        return stats
    
    def get_live_data(self, limit: int = 50) -> List[Dict]:
        """Get recent processed records"""
        return self.data.live_data[-limit:]
    
    def get_store_summary(self, store: int) -> Dict:
        """Get store summary"""
        baseline = self.data.store_baselines.get(store, {})
        recent = self.data.get_recent_sales(store)
        
        store_alerts = [a for a in self.data.alerts if a['record']['store'] == store]
        
        return {
            'store': store,
            'baseline_mean': baseline.get('mean', 0),
            'baseline_std': baseline.get('std', 0),
            'recent_sales': recent,
            'recent_average': round(np.mean(recent), 2) if recent else 0,
            'alert_count': len(store_alerts),
            'critical_alerts': len([a for a in store_alerts if a['alert_level'] == 'CRITICAL']),
            'last_alert': store_alerts[-1] if store_alerts else None
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.data.monitoring_stats = {
            'total_records': 0,
            'forecasts_made': 0,
            'alerts_info': 0,
            'alerts_warning': 0,
            'alerts_critical': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'forecast_accuracy': 0
        }
        self.data.alerts = []
        self.data.forecasts = []
        self.data.live_data = []
        
        return {'status': 'success', 'message': 'Statistics reset'}

# ============================================================================
# MAIN API INSTANCE
# ============================================================================

api = StreamingAPI()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING API")
    print("="*70)
    
    test_record = {
        'Store': 5,
        'Date': datetime.now().isoformat(),
        'Weekly_Sales': 1500000,
        'Holiday_Flag': 0,
        'Temperature': 75,
        'Fuel_Price': 3.5,
        'CPI': 210,
        'Unemployment': 7.5
    }
    
    # Test forecast
    print("\n1. Testing Forecasting:")
    forecast = api.forecast_sales(test_record)
    if forecast.get('success'):
        print(f"✓ Predicted Sales: ${forecast['predicted_sales']:,.0f}")
        print(f"  Confidence Interval: ${forecast['confidence_interval']['lower']:,.0f} - ${forecast['confidence_interval']['upper']:,.0f}")
    else:
        print(f"✗ Forecast failed: {forecast.get('error')}")
    
    # Test anomaly detection
    print("\n2. Testing Anomaly Detection:")
    result = api.process_record(test_record)
    print(f"{'✓ Anomaly' if result['is_anomaly'] else '✗ Normal'}: {result['alert_level']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Methods: {', '.join(result['detections'])}")
    
    print("\n✅ API Ready!")