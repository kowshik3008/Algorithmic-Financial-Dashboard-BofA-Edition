import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        fig, ax = plt.subplots(figsize=(12, 5))
        x_axis = df.index if date_col else range(len(df))
        ax.plot(x_axis, df[close_col], label='Close', color='blue')
        ax.plot(x_axis, df['MA30'], label='MA30', color='orange', alpha=0.8)
        ax.plot(x_axis, df['MA90'], label='MA90', color='green', alpha=0.8)
        ax.set_title("Price Trend")
        ax.legend()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            if volume_col:
                st.subheader("Volume Distribution")
                fig_vol, ax_vol = plt.subplots(figsize=(6, 4))
                ax_vol.hist(df[volume_col].dropna(), bins=40, color='purple')
                st.pyplot(fig_vol)
                
            st.subheader("Volatility (30-day Rolling)")
            fig_volat, ax_volat = plt.subplots(figsize=(6, 4))
            ax_volat.plot(df.index if date_col else range(len(df)), df['Volatility'])
            st.pyplot(fig_volat)
            
        with col2:
            st.subheader("Correlation Matrix")
            # Using Streamlit's native dataframe to display correlation
            st.dataframe(df.corr(numeric_only=True))
            
            st.subheader("Daily Returns Distribution")
            fig_ret, ax_ret = plt.subplots(figsize=(6, 4))
            ax_ret.hist(df['Return'].dropna(), bins=50, color='teal')
            ax_ret.axvline(0, color='black', linestyle='dashed', linewidth=1)
            st.pyplot(fig_ret)
            
            st.subheader("MACD Indicator")
            fig_macd, ax_macd = plt.subplots(figsize=(6, 4))
            macd_val = df['MACD'].dropna()
            ax_macd.fill_between(macd_val.index, macd_val, alpha=0.5, color='green')
            ax_macd.plot(macd_val.index, macd_val, color='green')
            st.pyplot(fig_macd)

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
            
            fig_ml, ax_ml = plt.subplots(figsize=(12, 4))
            x_test_axis = df_ml.index[-len(y_test):] if date_col else range(len(y_test))
            ax_ml.plot(x_test_axis, y_test, label='Actual Price')
            ax_ml.plot(x_test_axis, pred, label='Predicted Price', linestyle='--', color='red')
            ax_ml.legend()
            ax_ml.set_title(f"{model_type} vs Actual (Test Set)")
            st.pyplot(fig_ml)
            
            # Feature Importance Insight
            if hasattr(model_ml, 'feature_importances_'):
                st.subheader("🤖 Driver Insights: Feature Importance")
                fig_feat, ax_feat = plt.subplots(figsize=(10, 3))
                importances = model_ml.feature_importances_
                indices = np.argsort(importances)
                ax_feat.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
                ax_feat.set_yticks(range(len(indices)), [features[i] for i in indices])
                ax_feat.set_title(f"What drives {model_type} predictions?")
                st.pyplot(fig_feat)
                
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
                    fig_ar, ax_ar = plt.subplots(figsize=(10, 4))
                    ax_ar.plot(range(30), forecast_arima, color='orange', marker='o')
                    ax_ar.set_title("ARIMA 30-Day Forecast")
                    st.pyplot(fig_ar)
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
                        fig_lstm, ax_lstm = plt.subplots(figsize=(12, 4))
                        ax_lstm.plot(real_data, label='Actual Price')
                        ax_lstm.plot(predictions, label='LSTM Prediction', linestyle='--', color='purple')
                        ax_lstm.legend()
                        st.pyplot(fig_lstm)
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
        fig_sig, ax_sig = plt.subplots(figsize=(14, 6))
        
        ax_sig.plot(x_axis, df[close_col], label='Close Price', color='black', alpha=0.6)
        ax_sig.plot(x_axis, df['MA7'], label='MA7', color='blue', alpha=0.4)
        ax_sig.plot(x_axis, df['MA30'], label='MA30', color='orange', alpha=0.4)
        
        buy_points = df[df['Buy_Signal']]
        sell_points = df[df['Sell_Signal']]
        
        ax_sig.scatter(x_axis[df['Buy_Signal']], buy_points[close_col], marker='^', color='green', s=100, label='Buy')
        ax_sig.scatter(x_axis[df['Sell_Signal']], sell_points[close_col], marker='v', color='red', s=100, label='Sell')
        
        ax_sig.legend()
        st.pyplot(fig_sig)
        
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
        fig_port, ax_port = plt.subplots(figsize=(10, 4))
        ax_port.plot(df.index if date_col else range(len(df)), df['Portfolio_Value'])
        st.pyplot(fig_port)

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
        fig_dd, ax_dd = plt.subplots(figsize=(12, 4))
        ax_dd.fill_between(x_axis, drawdown, color='red', alpha=0.3)
        ax_dd.plot(x_axis, drawdown, color='red')
        ax_dd.set_title("Drawdown over Time")
        ax_dd.set_ylabel("Percentage")
        st.pyplot(fig_dd)

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
                fig_line, ax_line = plt.subplots(figsize=(6, 3))
                ax_line.plot(df.index if date_col else range(len(df)), df[col], color='teal')
                ax_line.set_title(f"{col} - Line Trend")
                st.pyplot(fig_line)
                chart_count += 1
                
            if chart_count >= max_charts:
                break
                
            with col_b:
                fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
                ax_hist.hist(df[col].dropna(), bins=40, color='indigo', alpha=0.7)
                ax_hist.set_title(f"{col} - Distribution")
                st.pyplot(fig_hist)
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
