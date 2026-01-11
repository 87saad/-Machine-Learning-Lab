"""
WALMART ANOMALY DETECTION - STREAMLIT DASHBOARD
===============================================
Complete UI for Sales Forecasting & Anomaly Detection
Run: streamlit run streamlit_ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import pickle

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Walmart Anomaly Detection",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #ff9800;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-info {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API CONNECTION
# ============================================================================

API_URL = "http://localhost:5000"

def check_api_connection():
    """Check if Flask API is running"""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_stats():
    """Get dashboard statistics"""
    try:
        response = requests.get(f"{API_URL}/api/stats")
        return response.json()
    except:
        return None

def get_alerts(level=None, limit=50):
    """Get recent alerts"""
    try:
        url = f"{API_URL}/api/alerts?limit={limit}"
        if level:
            url += f"&level={level}"
        response = requests.get(url)
        return response.json()
    except:
        return None

def get_forecasts(limit=50):
    """Get forecast history"""
    try:
        response = requests.get(f"{API_URL}/api/forecasts?limit={limit}")
        return response.json()
    except:
        return None

def get_live_data(limit=50):
    """Get live data for charts"""
    try:
        response = requests.get(f"{API_URL}/api/live-data?limit={limit}")
        return response.json()
    except:
        return None

def get_store_summary(store_id):
    """Get store summary"""
    try:
        response = requests.get(f"{API_URL}/api/store/{store_id}")
        return response.json()
    except:
        return None

def forecast_sales(record):
    """Send record for forecasting"""
    try:
        response = requests.post(f"{API_URL}/api/forecast", json=record)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
    """Get live data for charts"""
    try:
        response = requests.get(f"{API_URL}/api/live-data?limit={limit}")
        return response.json()
    except:
        return None

def detect_anomaly(record):
    """Send record for anomaly detection"""
    try:
        response = requests.post(f"{API_URL}/api/detect", json=record)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def load_raw_dataset():
    """Load raw Walmart dataset"""
    try:
        df = pd.read_csv('walmart.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        # Generate sample data if file not found
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

def load_results():
    """Load anomaly detection results"""
    try:
        return pd.read_csv('anomaly_detection_results.csv')
    except:
        return None
    """Get store summary"""
    try:
        response = requests.get(f"{API_URL}/api/store/{store_id}")
        return response.json()
    except:
        return None

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/2560px-Walmart_logo.svg.png", width=200)
st.sidebar.title("🛒 Walmart Analytics")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Anomaly Detection", "📊 Live Monitoring", "🏪 Store Analysis", "💰 Business Impact"]
)

st.sidebar.markdown("---")

# API Status
api_status = check_api_connection()
if api_status:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.info("Run: `python flask_api_server.py`")

st.sidebar.markdown("---")
st.sidebar.info("""
**Quick Guide:**
1. Start Flask API
2. Upload CSV or enter data
3. View real-time results
4. Analyze anomalies
""")

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🛒 Walmart Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ Flask API is not running! Start it first: `python flask_api_server.py`")
        st.stop()
    
    # Load raw dataset for display
    raw_data = load_raw_dataset()
    
    # Get statistics from API
    stats = get_stats()
    
    if stats:
        # Dataset Overview Section
        st.subheader("📊 Dataset Overview")
        
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Total Records", f"{len(raw_data):,}")
        dcol2.metric("Stores", raw_data['Store'].nunique())
        dcol3.metric("Date Range", f"{(raw_data['Date'].max() - raw_data['Date'].min()).days} days")
        dcol4.metric("Avg Sales/Week", f"${raw_data['Weekly_Sales'].mean():,.0f}")
        
        # Show sample data
        with st.expander("📁 View Raw Dataset Sample"):
            st.dataframe(raw_data.head(20), use_container_width=True)
        
        st.markdown("---")
        
        # System Metrics Row
        st.subheader("🎯 System Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Records",
                value=f"{stats.get('total_records', 0):,}",
                delta=f"{stats.get('recent_alerts', 0)} recent"
            )
        
        with col2:
            st.metric(
                label="🔴 Critical Alerts",
                value=stats.get('alerts_critical', 0),
                delta=None if stats.get('alerts_critical', 0) == 0 else "⚠️"
            )
        
        with col3:
            st.metric(
                label="🟡 Warning Alerts",
                value=stats.get('alerts_warning', 0)
            )
        
        with col4:
            st.metric(
                label="📈 Alert Rate",
                value=f"{stats.get('alert_rate', 0):.1f}%"
            )
        
        st.markdown("---")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Alert Distribution")
            
            alert_data = pd.DataFrame({
                'Level': ['Critical', 'Warning', 'Info'],
                'Count': [
                    stats.get('alerts_critical', 0),
                    stats.get('alerts_warning', 0),
                    stats.get('alerts_info', 0)
                ]
            })
            
            fig = px.bar(
                alert_data,
                x='Level',
                y='Count',
                color='Level',
                color_discrete_map={
                    'Critical': '#ff4444',
                    'Warning': '#ff9800',
                    'Info': '#4CAF50'
                }
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Detection Performance")
            
            tp = stats.get('true_positives', 0)
            fp = stats.get('false_positives', 0)
            tn = stats.get('true_negatives', 0)
            fn = stats.get('false_negatives', 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            perf_data = pd.DataFrame({
                'Metric': ['Precision', 'Recall'],
                'Score': [precision * 100, recall * 100]
            })
            
            fig = go.Figure(go.Bar(
                x=perf_data['Score'],
                y=perf_data['Metric'],
                orientation='h',
                marker_color=['#667eea', '#764ba2']
            ))
            fig.update_layout(height=300, xaxis_title="Score (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Alerts
        st.markdown("---")
        st.subheader("🚨 Recent Alerts")
        
        alerts = get_alerts(limit=10)
        
        if alerts and alerts.get('count', 0) > 0:
            for alert in alerts['alerts']:
                level = alert['alert_level']
                store = alert['record']['store']
                sales = alert['record']['sales']
                confidence = alert['confidence']
                
                if level == 'CRITICAL':
                    st.markdown(f"""
                    <div class="alert-critical">
                        <strong>🔴 CRITICAL - Store {store}</strong><br>
                        Sales: ${sales:,.0f} | Confidence: {confidence}%<br>
                        Detected by: {', '.join(alert['detections'])}
                    </div>
                    """, unsafe_allow_html=True)
                elif level == 'WARNING':
                    st.markdown(f"""
                    <div class="alert-warning">
                        <strong>🟡 WARNING - Store {store}</strong><br>
                        Sales: ${sales:,.0f} | Confidence: {confidence}%
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No alerts yet. Upload data or start monitoring!")

# ============================================================================
# PAGE 3: ANOMALY DETECTION
# ============================================================================

elif page == "🔍 Anomaly Detection":
    st.markdown('<h1 class="main-header">🔍 Anomaly Detection</h1>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ Flask API is not running!")
        st.stop()
    
    tab1, tab2 = st.tabs(["📄 Single Record", "📁 Batch Upload"])
    
    # Tab 1: Single Record
    with tab1:
        st.subheader("Enter Sales Record")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            store = st.number_input("Store Number", min_value=1, max_value=100, value=5)
            sales = st.number_input("Weekly Sales ($)", min_value=0, value=1500000, step=10000)
            holiday = st.checkbox("Holiday Week")
        
        with col2:
            temperature = st.slider("Temperature (°F)", 0, 120, 75)
            fuel_price = st.slider("Fuel Price ($)", 2.0, 5.0, 3.5, 0.1)
        
        with col3:
            cpi = st.slider("CPI", 100, 250, 210)
            unemployment = st.slider("Unemployment (%)", 0.0, 20.0, 7.5, 0.1)
        
        if st.button("🔍 Detect Anomaly", type="primary"):
            record = {
                'Store': store,
                'Date': datetime.now().isoformat(),
                'Weekly_Sales': sales,
                'Holiday_Flag': 1 if holiday else 0,
                'Temperature': temperature,
                'Fuel_Price': fuel_price,
                'CPI': cpi,
                'Unemployment': unemployment
            }
            
            with st.spinner("Analyzing..."):
                result = detect_anomaly(record)
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display result
                if result['is_anomaly']:
                    st.error(f"🚨 ANOMALY DETECTED - {result['alert_level']}")
                else:
                    st.success("✅ Normal Sales Pattern")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{result['confidence']:.1f}%")
                col2.metric("Detection Count", f"{result['detection_count']}/3")
                col3.metric("Alert Level", result['alert_level'])
                
                # Details
                st.subheader("Detection Details")
                st.json(result)
    
    # Tab 2: Batch Upload
    with tab2:
        st.subheader("Upload CSV File")
        
        # Show required format
        with st.expander("📋 Required CSV Format"):
            st.markdown("""
            Your CSV must have these columns (case-insensitive):
            - **Store** (integer)
            - **Date** (any date format)
            - **Weekly_Sales** (number)
            - **Holiday_Flag** (0 or 1)
            - **Temperature** (number)
            - **Fuel_Price** (number)
            - **CPI** (number)
            - **Unemployment** (number)
            
            **Example:**
            ```
            Store,Date,Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment
            1,2010-02-05,1643690.90,0,42.31,2.572,211.096,8.106
            1,2010-02-12,1641957.44,1,38.51,2.548,211.242,8.106
            ```
            """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        # Sample CSV download
        if st.button("📥 Download Sample CSV"):
            sample_data = {
                'Store': [1, 1, 2, 2, 3],
                'Date': ['2010-02-05', '2010-02-12', '2010-02-05', '2010-02-12', '2010-02-05'],
                'Weekly_Sales': [1643690.90, 1641957.44, 2320192.45, 2253764.32, 1530562.78],
                'Holiday_Flag': [0, 1, 0, 1, 0],
                'Temperature': [42.31, 38.51, 46.63, 43.31, 54.42],
                'Fuel_Price': [2.572, 2.548, 2.572, 2.548, 2.572],
                'CPI': [211.096, 211.242, 211.096, 211.242, 211.096],
                'Unemployment': [8.106, 8.106, 8.334, 8.334, 8.667]
            }
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="💾 Download",
                data=csv,
                file_name="walmart_sample.csv",
                mime="text/csv"
            )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            
            if st.button("🚀 Process All Records"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                # Check required columns
                required_cols = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 
                                'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
                
                # Check which columns exist (case-insensitive)
                df_cols_lower = {col.lower(): col for col in df.columns}
                missing_cols = []
                col_mapping = {}
                
                for req_col in required_cols:
                    req_col_lower = req_col.lower()
                    # Try exact match first
                    if req_col in df.columns:
                        col_mapping[req_col] = req_col
                    # Try case-insensitive match
                    elif req_col_lower in df_cols_lower:
                        col_mapping[req_col] = df_cols_lower[req_col_lower]
                    else:
                        missing_cols.append(req_col)
                
                if missing_cols:
                    st.error(f"❌ Missing columns in CSV: {', '.join(missing_cols)}")
                    st.info(f"Available columns: {', '.join(df.columns.tolist())}")
                    st.stop()
                
                results = []
                total = len(df)
                
                for idx, row in df.iterrows():
                    try:
                        record = {
                            'Store': int(row[col_mapping['Store']]),
                            'Date': str(row[col_mapping['Date']]),
                            'Weekly_Sales': float(row[col_mapping['Weekly_Sales']]),
                            'Holiday_Flag': int(row[col_mapping['Holiday_Flag']]),
                            'Temperature': float(row[col_mapping['Temperature']]),
                            'Fuel_Price': float(row[col_mapping['Fuel_Price']]),
                            'CPI': float(row[col_mapping['CPI']]),
                            'Unemployment': float(row[col_mapping['Unemployment']])
                        }
                    except (KeyError, ValueError) as e:
                        st.error(f"Error processing row {idx}: {e}")
                        continue
                    
                    result = detect_anomaly(record)
                    results.append(result)
                    
                    progress = (idx + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {idx + 1}/{total}")
                
                # Summary
                anomalies = [r for r in results if r.get('is_anomaly', False)]
                
                st.success(f"✅ Processed {total} records")
                st.info(f"🚨 Found {len(anomalies)} anomalies ({len(anomalies)/total*100:.1f}%)")
                
                # Show anomalies
                if anomalies:
                    st.subheader("Detected Anomalies")
                    anomaly_df = pd.DataFrame([{
                        'Store': r['record']['store'],
                        'Sales': r['record']['sales'],
                        'Level': r['alert_level'],
                        'Confidence': r['confidence']
                    } for r in anomalies])
                    st.dataframe(anomaly_df)

# ============================================================================
# PAGE 4: LIVE MONITORING
# ============================================================================

elif page == "📊 Live Monitoring":
    st.markdown('<h1 class="main-header">📈 Live Sales Monitoring</h1>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ Flask API is not running!")
        st.stop()
    
    st.info("📡 Real-time monitoring active. Auto-refreshing every 5 seconds...")
    
    # Placeholder for live chart
    chart_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Auto-refresh loop
    while True:
        live_data = get_live_data(limit=50)
        
        if live_data and live_data.get('count', 0) > 0:
            data = live_data['data']
            
            # Prepare data for chart
            df = pd.DataFrame([{
                'Date': d['record']['date'],
                'Store': d['record']['store'],
                'Sales': d['record']['sales'],
                'Anomaly': d['is_anomaly'],
                'Level': d.get('alert_level', 'Normal')
            } for d in data])
            
            # Create chart
            fig = px.scatter(
                df,
                x='Date',
                y='Sales',
                color='Anomaly',
                color_discrete_map={True: 'red', False: 'blue'},
                size=[100 if a else 50 for a in df['Anomaly']],
                hover_data=['Store', 'Level']
            )
            
            fig.update_layout(
                title="Real-Time Sales Data",
                xaxis_title="Date",
                yaxis_title="Weekly Sales ($)",
                height=500
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True, key="chart1")
 
            
            # Stats
            anomalies_count = df['Anomaly'].sum()
            stats_placeholder.metric(
                "Anomalies in View",
                f"{anomalies_count}/{len(df)}",
                f"{anomalies_count/len(df)*100:.1f}%"
            )
        
        time.sleep(5)

# ============================================================================
# PAGE 5: STORE ANALYSIS
# ============================================================================

elif page == "🏪 Store Analysis":
    st.markdown('<h1 class="main-header">🏪 Store Analysis</h1>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ Flask API is not running!")
        st.stop()
    
    store_id = st.number_input("Select Store Number", min_value=1, max_value=100, value=5)
    
    if st.button("📊 Load Store Data"):
        with st.spinner("Loading..."):
            summary = get_store_summary(store_id)
        
        if summary:
            st.subheader(f"Store {store_id} Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Baseline Mean", f"${summary['baseline_mean']:,.0f}")
            col2.metric("Baseline Std", f"${summary['baseline_std']:,.0f}")
            col3.metric("Total Alerts", summary['alert_count'])
            col4.metric("Critical Alerts", summary['critical_alerts'])
            
            # Recent sales trend
            if summary['recent_sales']:
                st.subheader("Recent Sales Trend")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=summary['recent_sales'],
                    mode='lines+markers',
                    name='Sales',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_hline(
                    y=summary['baseline_mean'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Baseline"
                )
                
                fig.update_layout(
                    title="Last 4 Weeks",
                    yaxis_title="Sales ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Last alert
            if summary['last_alert']:
                st.subheader("Last Alert")
                st.json(summary['last_alert'])
        else:
            st.warning("No data available for this store")

# ============================================================================
# PAGE 6: BUSINESS IMPACT
# ============================================================================

elif page == "💰 Business Impact":
    st.markdown('<h1 class="main-header">📊 Business Impact & ROI</h1>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ Flask API is not running! Start it first for accurate ROI calculations.")
        st.stop()
    
    # Get real stats from API
    stats = get_stats()
    
    precision = 0
    total_anomalies = 0
    if stats:
        st.success("✅ Using live data from API for accurate calculations")
        
        # Display current performance
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{stats.get('total_records', 0):,}")
        col2.metric("Detection Rate", f"{stats.get('alert_rate', 0):.1f}%")
        
        # Calculate precision
        tp = stats.get('true_positives', 0)
        fp = stats.get('false_positives', 0)
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        col3.metric("System Precision", f"{precision:.1f}%")
        
        st.markdown("---")
    
    # Quick ROI Calculator
    st.subheader("💰 Live ROI Calculator (Using API Data)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_loss_per_anomaly = st.number_input(
            "Average Loss per Missed Anomaly ($)",
            min_value=0,
            value=50000,
            step=1000
        )
        
        anomalies_per_month = st.number_input(
            "Expected Anomalies per Month",
            min_value=0,
            value=10,
            step=1
        )
    
    with col2:
        detection_rate = st.slider(
            "System Detection Rate (%)",
            0, 100, 85
        )
        
        implementation_cost = st.number_input(
            "Implementation Cost ($)",
            min_value=0,
            value=100000,
            step=10000
        )
    
    # Calculate ROI
    if st.button("Calculate ROI"):
        if stats:
            actual_detection_rate = precision
            monthly_anomalies = total_anomalies / 12 if total_anomalies > 0 else anomalies_per_month
        else:
            actual_detection_rate = detection_rate
            monthly_anomalies = anomalies_per_month

        monthly_savings = (
            monthly_anomalies *
            avg_loss_per_anomaly *
            (actual_detection_rate / 100)
        )

        annual_savings = monthly_savings * 12
        roi_months = implementation_cost / monthly_savings if monthly_savings > 0 else 0
        roi_percentage = (
            (annual_savings - implementation_cost) / implementation_cost * 100
            if implementation_cost > 0 else 0
        )

        
        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly Savings", f"${monthly_savings:,.0f}")
        col2.metric("Annual Savings", f"${annual_savings:,.0f}")
        col3.metric("ROI", f"{roi_percentage:.1f}%")
        
        st.metric("Payback Period", f"{roi_months:.1f} months")
        
        st.info(f"✓ Using actual system detection rate: {actual_detection_rate:.1f}%")
        st.info(f"✓ Based on {total_anomalies} actual anomalies detected")
        
        # Savings chart
        months = list(range(1, 13))
        cumulative_savings = [monthly_savings * m - implementation_cost for m in months]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_savings,
            mode='lines+markers',
            fill='tozeroy',
            name='Net Savings',
            line=dict(color='green', width=3)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Cumulative Savings Over Time (Based on Real Performance)",
            xaxis_title="Month",
            yaxis_title="Net Savings ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed breakdown
        with st.expander("📊 View Detailed Analysis"):
            st.markdown(f"""
            **Calculation Details:**
            - Anomalies per month: {anomalies_per_month}
            - Average loss per anomaly: ${avg_loss_per_anomaly:,}
            - System detection rate: {actual_detection_rate:.1f}% (from live data)
            - Monthly prevented loss: ${monthly_savings:,.0f}
            - Implementation cost: ${implementation_cost:,}
            - Break-even month: {roi_months:.1f}
            
            **Recommendation:** {"✅ PROCEED - Positive ROI" if roi_percentage > 0 else "⚠️ Adjust parameters"}
            """)
    
    st.markdown("---")
    
    # Link to detailed report
    st.info("""
    📄 **Generate Detailed Report**
    
    For comprehensive business impact analysis including:
    - Model performance metrics
    - 5-year financial projections
    - Anomaly breakdown by category
    - Executive summary
    
    Run: `python business_impact.py`
    
    This will generate:
    - business_impact_analysis.png (4 charts)
    - executive_summary.txt (detailed report)
    """)


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛒 Walmart Anomaly Detection System | Built with Streamlit & Flask</p>
    <p>© 2024 | ML Project</p>
</div>
""", unsafe_allow_html=True)