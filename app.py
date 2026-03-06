import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# Safe optional imports for heavy/fringe libraries
HAS_PROPHET = False
HAS_XGB = False
HAS_TA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    pass

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    pass

try:
    import ta
    HAS_TA = True
except Exception:
    pass

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TF = True
except Exception:
    HAS_TF = False

st.set_page_config(page_title='Advanced Stock Analysis Dashboard', layout='wide', page_icon='📈')

st.title('📈 Advanced Bank of America Stock Analysis')
st.markdown("""
This dashboard provides an end-to-end analysis of stock data including **Exploratory Data Analysis (EDA), Technical Indicators, Forecasting Models (ARIMA/ML), and Trading Strategy Backtesting**.
""")

st.sidebar.header('📂 Upload Dataset')
file = st.sidebar.file_uploader('Upload CSV', type=['csv'])

if file:
    # 1. Load Data
    df = pd.read_csv(file)

    # 2. Clean Column Names
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower()
    
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    close_col = next((c for c in df.columns if 'close' in c or 'last' in c), None)
    volume_col = next((c for c in df.columns if 'vol' in c), None)
    
    if not close_col:
        st.error(f"Could not find a 'Close' price column. Available columns are: {', '.join(original_cols)}")
        st.stop()
        
    # 3. Data Preprocessing
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        
    df = df.ffill()

    # Clean up currency strings if necessary
    if df[close_col].dtype == 'object':
        df[close_col] = pd.to_numeric(df[close_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    if volume_col and df[volume_col].dtype == 'object':
        df[volume_col] = pd.to_numeric(df[volume_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')

    # 4. Feature Engineering
    df['MA7'] = df[close_col].rolling(7).mean()
    df['MA30'] = df[close_col].rolling(30).mean()
    df['MA90'] = df[close_col].rolling(90).mean()
    df['Return'] = df[close_col].pct_change()
    df['Volatility'] = df['Return'].rolling(30).std()
    
    if HAS_TA:
        df['RSI'] = ta.momentum.RSIIndicator(df[close_col]).rsi()
        macd = ta.trend.MACD(df[close_col])
        df['MACD'] = macd.macd()
        bb = ta.volatility.BollingerBands(df[close_col])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
    else:
        # Fallbacks
        df['RSI'] = df['Return'].rolling(14).mean() * 100 # Rough proxy
        df['MACD'] = df['MA7'] - df['MA30']

    st.sidebar.success("✅ Dataset loaded & features engineered!")
    
    # -------------------------
    # Top-Level KPI Metrics
    # -------------------------
    st.markdown("### 📊 Market Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    current_price = df[close_col].iloc[-1]
    prev_price = df[close_col].iloc[-2] if len(df) > 1 else current_price
    pct_change = ((current_price - prev_price) / prev_price) * 100
    
    kpi1.metric("Current Price", f"${current_price:,.2f}", f"{pct_change:.2f}%")
    
    if volume_col:
        avg_vol = df[volume_col].mean()
        curr_vol = df[volume_col].iloc[-1]
        vol_change = ((curr_vol - avg_vol) / avg_vol) * 100 if avg_vol else 0
        kpi2.metric("Latest Volume", f"{curr_vol:,.0f}", f"{vol_change:.2f}% vs Avg")
    else:
        kpi2.metric("Volume", "N/A")
        
    high_52 = df[close_col].rolling(252).max().iloc[-1] if len(df) >= 252 else df[close_col].max()
    low_52 = df[close_col].rolling(252).min().iloc[-1] if len(df) >= 252 else df[close_col].min()
    
    kpi3.metric("52-Week High", f"${high_52:,.2f}")
    kpi4.metric("52-Week Low", f"${low_52:,.2f}")
    
    with st.expander("💡 View Insight Summary"):
        st.write(f"**Average Close Price:** ${df[close_col].mean():.2f}")
        st.write(f"**Highest Close Price:** ${df[close_col].max():.2f}")
        st.write(f"**Lowest Close Price:** ${df[close_col].min():.2f}")
        if volume_col:
            st.write(f"**Average Volume:** {df[volume_col].mean():,.0f}")
        st.write(f"**Models Available:** ARIMA, ML ({'XGBoost' if HAS_XGB else 'RandomForest'})" + (", LSTM" if HAS_TF else "") + (", Prophet" if HAS_PROPHET else ""))

    st.markdown("---")
    
    # Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA & Indicators", 
        "🤖 Forecasting Models", 
        "📈 Trading Strategy", 
        "⚠️ Risk Analysis",
        "📑 Advanced Reports"
    ])
    
    # -------------------------
    # TAB 1: EDA & Indicators
    # -------------------------
    with tab1:
        st.header("Exploratory Data Analysis")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Stock Price & Moving Averages")
        fig = go.Figure()
        x_axis = df.index.to_numpy() if date_col else np.arange(len(df))
        fig.add_trace(go.Scatter(x=x_axis, y=df[close_col].to_numpy(), mode='lines', name='Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x_axis, y=df['MA30'].to_numpy(), mode='lines', name='MA30', line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=x_axis, y=df['MA90'].to_numpy(), mode='lines', name='MA90', line=dict(color='green', width=2)))
        fig.update_layout(title="Interactive Price Trend", xaxis_title="Date", yaxis_title="Price", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if volume_col:
                st.subheader("Volume Distribution")
                fig_vol = px.histogram(df, x=volume_col, nbins=40, color_discrete_sequence=['purple'])
                fig_vol.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_vol, use_container_width=True)
                
            st.subheader("Volatility (30-day Rolling)")
            fig_volat = go.Figure()
            fig_volat.add_trace(go.Scatter(x=x_axis, y=df['Volatility'].to_numpy(), mode='lines', line=dict(color='red')))
            fig_volat.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_volat, use_container_width=True)
            
        with col2:
            st.subheader("Correlation Matrix")
            # Using Streamlit's native dataframe to display correlation
            st.dataframe(df.corr(numeric_only=True))
            
            st.subheader("Daily Returns Distribution")
            fig_ret = px.histogram(df.dropna(subset=['Return']), x='Return', nbins=50, color_discrete_sequence=['teal'])
            fig_ret.add_vline(x=0, line_dash="dash", line_color="black")
            fig_ret.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_ret, use_container_width=True)
            
            st.subheader("MACD Indicator")
            fig_macd = go.Figure()
            macd_val = df['MACD'].dropna()
            fig_macd.add_trace(go.Scatter(x=macd_val.index.to_numpy(), y=macd_val.to_numpy(), fill='tozeroy', mode='lines', line=dict(color='green')))
            fig_macd.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_macd, use_container_width=True)

        st.subheader("Time Series Decomposition")
        try:
            # period=30 to approximate a monthly pattern visually in daily stock trading data
            decomposition = seasonal_decompose(df[close_col].dropna(), model='additive', period=30)
            fig_decomp = decomposition.plot()
            fig_decomp.set_size_inches(12, 8)
            st.pyplot(fig_decomp)
        except Exception as e:
            st.warning(f"Decomposition skipped (dataset may be too short or irregular): {e}")

    # -------------------------
    # TAB 2: Forecasting Models
    # -------------------------
    with tab2:
        st.header("Predictive Models")
        st.write("Train and evaluate models on historical data. Click the buttons below to run heavy computations.")
        
        st.subheader("1. Machine Learning Forecasting")
        features = ['MA7', 'MA30', 'MA90', 'RSI', 'MACD']
        df_ml = df.dropna(subset=features + [close_col])
        
        if len(df_ml) > 100:
            X = df_ml[features]
            y = df_ml[close_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model_type = "XGBoost" if HAS_XGB else "RandomForest"
            model_ml = XGBRegressor() if HAS_XGB else RandomForestRegressor(random_state=42)
            model_ml.fit(X_train, y_train)
            pred = model_ml.predict(X_test)
            
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("ML Model Used", model_type)
            col_m2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
            col_m3.metric("Root Mean Sq Error (RMSE)", f"{rmse:.2f}")
            
            fig_ml = go.Figure()
            x_test_axis = df_ml.index[-len(y_test):].to_numpy() if date_col else np.arange(len(y_test))
            fig_ml.add_trace(go.Scatter(x=x_test_axis, y=y_test.to_numpy(), mode='lines', name='Actual Price'))
            fig_ml.add_trace(go.Scatter(x=x_test_axis, y=pred, mode='lines', name='Predicted Price', line=dict(dash='dash', color='red')))
            fig_ml.update_layout(title=f"{model_type} vs Actual (Test Set)", hovermode="x unified", height=400)
            st.plotly_chart(fig_ml, use_container_width=True)
            
            # Feature Importance Insight
            if hasattr(model_ml, 'feature_importances_'):
                st.subheader("🤖 Driver Insights: Feature Importance")
                importances = model_ml.feature_importances_
                indices = np.argsort(importances)
                fig_feat = px.bar(x=importances[indices], y=[features[i] for i in indices], orientation='h', color_discrete_sequence=['skyblue'])
                fig_feat.update_layout(title=f"What drives {model_type} predictions?", height=300)
                st.plotly_chart(fig_feat, use_container_width=True)
                
        else:
            st.warning("Not enough clean data points for ML training.")

        st.markdown("---")
        
        if st.button("▶️ Run ARIMA Forecast (30 Days)"):
            with st.spinner("Training ARIMA model..."):
                train_size = int(len(df) * 0.8)
                train = df.iloc[:train_size]
                try:
                    arima_model = ARIMA(train[close_col], order=(5,1,0))
                    arima_fit = arima_model.fit()
                    forecast_arima = arima_fit.forecast(30)
                    
                    st.success("ARIMA model trained successfully!")
                    
                    # Generate dynamic date index for forecast if applicable
                    last_date = df.index[-1]
                    if date_col:
                         future_dates = pd.date_range(start=last_date, periods=31, freq='B')[1:]
                    else:
                         future_dates = range(len(df), len(df)+30)

                    fig_ar = go.Figure()
                    fig_ar.add_trace(go.Scatter(x=future_dates, y=forecast_arima, mode='lines+markers', name='Forecast', marker=dict(color='orange')))
                    fig_ar.update_layout(title="ARIMA 30-Day Forecast", hovermode="x unified", height=400)
                    st.plotly_chart(fig_ar, use_container_width=True)
                except Exception as e:
                    st.error(f"ARIMA Error: {e}")

        if HAS_TF:
            st.markdown("---")
            if st.button("▶️ Run Deep Learning LSTM Forecast"):
                with st.spinner("Training LSTM Network (may take a minute)..."):
                    try:
                        close_data = df[[close_col]].dropna()
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(close_data)

                        sequence_length = 60
                        X_lstm, y_lstm = [], []
                        for i in range(sequence_length, len(scaled_data)):
                            X_lstm.append(scaled_data[i-sequence_length:i])
                            y_lstm.append(scaled_data[i])

                        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                        split = int(len(X_lstm) * 0.8)

                        X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
                        y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

                        model_lstm = Sequential()
                        model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1],1)))
                        model_lstm.add(LSTM(50))
                        model_lstm.add(Dense(1))
                        model_lstm.compile(optimizer='adam', loss='mse')

                        model_lstm.fit(X_train_lstm, y_train_lstm, epochs=3, batch_size=32, verbose=0)
                        
                        predictions = model_lstm.predict(X_test_lstm)
                        predictions = scaler.inverse_transform(predictions)
                        real_data = scaler.inverse_transform(y_test_lstm)
                        
                        st.success("LSTM Training Complete!")
                        fig_lstm = go.Figure()
                        fig_lstm.add_trace(go.Scatter(y=real_data.flatten(), mode='lines', name='Actual Price'))
                        fig_lstm.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='LSTM Prediction', line=dict(dash='dash', color='purple')))
                        fig_lstm.update_layout(title="LSTM Deep Learning Forecast vs Actual", hovermode="x unified", height=400)
                        st.plotly_chart(fig_lstm, use_container_width=True)
                    except Exception as e:
                        st.error(f"LSTM Training Error: {e}")

    # -------------------------
    # TAB 3: Trading Strategy
    # -------------------------
    with tab3:
        st.header("Moving Average Crossover Strategy")
        st.write("Strategy: **Buy** when MA7 crosses above MA30. **Sell** when MA7 crosses below MA30.")
        
        df['Buy'] = np.where(df['MA7'] > df['MA30'], 1, 0)
        df['Sell'] = np.where(df['MA7'] < df['MA30'], 1, 0)
        
        # We only want to mark the exact crossover point, not every day it's above/below
        df['Buy_Signal'] = df['Buy'].diff() == 1
        df['Sell_Signal'] = df['Sell'].diff() == 1
        
        st.subheader("Buy & Sell Signals")
        fig_sig = go.Figure()
        
        x_full = df.index.to_numpy()
        fig_sig.add_trace(go.Scatter(x=x_full, y=df[close_col].to_numpy(), mode='lines', name='Close Price', line=dict(color='black', width=1)))
        fig_sig.add_trace(go.Scatter(x=x_full, y=df['MA7'].to_numpy(), mode='lines', name='MA7', line=dict(color='blue', width=1, dash='dot')))
        fig_sig.add_trace(go.Scatter(x=x_full, y=df['MA30'].to_numpy(), mode='lines', name='MA30', line=dict(color='orange', width=1, dash='dot')))
        
        buy_points = df[df['Buy_Signal']]
        sell_points = df[df['Sell_Signal']]
        
        fig_sig.add_trace(go.Scatter(x=buy_points.index.to_numpy(), y=buy_points[close_col].to_numpy(), mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=12)))
        fig_sig.add_trace(go.Scatter(x=sell_points.index.to_numpy(), y=sell_points[close_col].to_numpy(), mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=12)))
        
        fig_sig.update_layout(title="Interactive Trading Signals", hovermode="x unified", height=500)
        st.plotly_chart(fig_sig, use_container_width=True)
        
        st.subheader("Backtesting Pipeline")
        initial_capital = 10000
        position = 0
        cash = initial_capital
        portfolio_values = []
        
        for i in range(len(df)):
            price = df[close_col].iloc[i]
            
            # Execute Buy
            if df['Buy_Signal'].iloc[i] and position == 0:
                position = cash / price
                cash = 0
            
            # Execute Sell
            elif df['Sell_Signal'].iloc[i] and position > 0:
                cash = position * price
                position = 0
                
            portfolio_values.append(cash + position * price)
            
        df['Portfolio_Value'] = portfolio_values
        
        # Calculate Strategy KPIs
        strategy_return = ((df['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital) * 100
        buy_hold_return = ((df[close_col].iloc[-1] - df[close_col].iloc[0]) / df[close_col].iloc[0]) * 100
        total_trades = df['Buy_Signal'].sum() + df['Sell_Signal'].sum()
        
        st.markdown("### 🏆 Strategy Performance")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Final Portfolio", f"${df['Portfolio_Value'].iloc[-1]:,.2f}")
        
        # Color coding the delta based on outperforming vs underperforming Buy & Hold
        diff = strategy_return - buy_hold_return
        col_b2.metric("Strategy Return", f"{strategy_return:.2f}%", f"{diff:.2f}% vs Buy&Hold")
        col_b3.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
        col_b4.metric("Total Trades Executed", int(total_trades))
        
        st.subheader("Portfolio Growth")
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=df.index.to_numpy(), y=np.array(df['Portfolio_Value']), mode='lines', line=dict(color='darkgreen')))
        fig_port.update_layout(title="Backtested Portfolio Value", xaxis_title="Date", yaxis_title="Capital ($)", hovermode="x unified", height=400)
        st.plotly_chart(fig_port, use_container_width=True)

    # -------------------------
    # TAB 4: Risk Analysis
    # -------------------------
    with tab4:
        st.header("Portfolio Risk Metrics")
        
        returns = df['Portfolio_Value'].pct_change().dropna()
        
        # Sharpe Ratio
        risk_free_rate = 0.01
        std_dev = returns.std()
        sharpe_ratio = 0
        if std_dev != 0:
            sharpe_ratio = (returns.mean() - risk_free_rate/252) / std_dev
            
        # Maximum Drawdown
        cum_max = df['Portfolio_Value'].cummax()
        drawdown = df['Portfolio_Value'] / cum_max - 1
        max_drawdown = drawdown.min()
        
        # Annualized Volatility
        annual_vol = std_dev * np.sqrt(252)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Sharpe Ratio", f"{sharpe_ratio:.4f}")
        col_r2.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
        col_r3.metric("Annualized Volatility", f"{annual_vol:.2%}")
        
        st.subheader("Drawdown Curve")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=x_axis, y=np.array(drawdown), fill='tozeroy', mode='lines', line=dict(color='red')))
        fig_dd.update_layout(title="Drawdown over Time (£)", yaxis_title="Percentage", hovermode="x unified", height=350)
        st.plotly_chart(fig_dd, use_container_width=True)

    # -------------------------
    # TAB 5: Bulk Visualizations
    # -------------------------
    with tab5:
        st.header("📄 Automated Bulk Visualizations")
        st.write("Generates comprehensive trend & distribution profiles for all numeric datasets automatically (up to 40 charts as configured).")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        chart_count = 0
        max_charts = 40
        
        for col in numeric_cols:
            if chart_count >= max_charts:
                break
                
            st.markdown(f"#### **{col.upper()} Analysis**")
            col_a, col_b = st.columns(2)
            
            with col_a:
                fig_line = go.Figure()
                x_b = df.index.to_numpy() if date_col else np.arange(len(df))
                fig_line.add_trace(go.Scatter(x=x_b, y=df[col].to_numpy(), mode='lines', line=dict(color='teal')))
                fig_line.update_layout(title=f"{col} - Line Trend", height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_line, use_container_width=True)
                chart_count += 1
                
            if chart_count >= max_charts:
                break
                
            with col_b:
                fig_hist = px.histogram(df.dropna(subset=[col]), x=col, nbins=40, color_discrete_sequence=['indigo'])
                fig_hist.update_layout(title=f"{col} - Distribution", height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_hist, use_container_width=True)
                chart_count += 1
                
            st.markdown("---")
            
        if chart_count >= max_charts:
            st.info("Reached maximum logical chart rendering limit (40) for performance.")

else:
    st.info("Upload a CSV file from the sidebar to start the analysis.")
    st.markdown("""
    ### Dashboard Features:
    * **EDA**: Visualize stock price trends, volume distributions, volatility, and correlation heatmaps.
    * **Forecast Models**: Compare ML models like Random Forest / XGBoost, ARIMA time-series models, and Deep Learning LSTMs.
    * **Trading Signals**: Dynamic algorithm tracking moving average crossovers (MA7 vs MA30) to generate Buy & Sell indicators.
    * **Risk Management**: Measure strategy profitability, Sharpe ratios, and max drawdown curves.
    """)
