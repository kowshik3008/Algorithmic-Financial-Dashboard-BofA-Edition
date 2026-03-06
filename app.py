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

# checking for optional heavy ML packages
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TF = True
except ImportError:
    HAS_TF = False


def preprocess_data(df):
    df.columns = [c.strip().lower() for c in df.columns]
    
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    close_col = next((c for c in df.columns if 'close' in c or 'last' in c), None)
    vol_col = next((c for c in df.columns if 'vol' in c), None)
    
    if not close_col:
        return None, None, None, None
        
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col).sort_index()
        
    df = df.ffill()

    # fix string-based currencies 
    if df[close_col].dtype == 'object':
        df[close_col] = pd.to_numeric(df[close_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    if vol_col and df[vol_col].dtype == 'object':
        df[vol_col] = pd.to_numeric(df[vol_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')

    return df, close_col, vol_col, date_col


def add_technical_indicators(df, close_col):
    df['MA7'] = df[close_col].rolling(7).mean()
    df['MA30'] = df[close_col].rolling(30).mean()
    df['MA90'] = df[close_col].rolling(90).mean()
    df['Return'] = df[close_col].pct_change()
    df['Volatility'] = df['Return'].rolling(30).std()
    
    if HAS_TA:
        df['RSI'] = ta.momentum.RSIIndicator(df[close_col]).rsi()
        df['MACD'] = ta.trend.MACD(df[close_col]).macd()
        bb = ta.volatility.BollingerBands(df[close_col])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
    else:
        # manual fallback logic if the user doesn't have the `ta` library installed
        df['RSI'] = df['Return'].rolling(14).mean() * 100 
        df['MACD'] = df['MA7'] - df['MA30']
        
    return df


def render_overview_kpis(df, close_col, vol_col):
    st.markdown("### Market Overview")
    kpis = st.columns(4)
    
    curr_px = df[close_col].iloc[-1]
    prev_px = df[close_col].iloc[-2] if len(df) > 1 else curr_px
    pct_chg = ((curr_px - prev_px) / prev_px) * 100
    kpis[0].metric("Current Price", f"${curr_px:,.2f}", f"{pct_chg:.2f}%")
    
    if vol_col:
        avg_v = df[vol_col].mean()
        curr_v = df[vol_col].iloc[-1]
        v_chg = ((curr_v - avg_v) / avg_v) * 100 if avg_v else 0
        kpis[1].metric("Latest Volume", f"{curr_v:,.0f}", f"{v_chg:.2f}% vs Avg")
    else:
        kpis[1].metric("Volume", "N/A")
        
    high52 = df[close_col].rolling(252).max().iloc[-1] if len(df) >= 252 else df[close_col].max()
    low52 = df[close_col].rolling(252).min().iloc[-1] if len(df) >= 252 else df[close_col].min()
    
    kpis[2].metric("52-Week High", f"${high52:,.2f}")
    kpis[3].metric("52-Week Low", f"${low52:,.2f}")
    
    with st.expander("Quick Stats"):
        st.write(f"**Avg Price:** ${df[close_col].mean():.2f}")
        st.write(f"**Max Price:** ${df[close_col].max():.2f}")
        st.write(f"**Min Price:** ${df[close_col].min():.2f}")
        if vol_col:
            st.write(f"**Avg Vol:** {df[vol_col].mean():,.0f}")
            
    st.markdown("---")


def render_eda(df, close_col, vol_col):
    st.header("Exploratory Data Analysis")
    st.dataframe(df.head())
    
    st.subheader("Price & Moving Averages")
    fig = go.Figure()
    x_axis = df.index
    fig.add_trace(go.Scatter(x=x_axis, y=df[close_col], name='Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_axis, y=df['MA30'], name='MA30', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=x_axis, y=df['MA90'], name='MA90', line=dict(color='green', width=2)))
    fig.update_layout(hovermode="x unified", height=500, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if vol_col:
            st.subheader("Volume Dist")
            fig_vol = px.histogram(df, x=vol_col, nbins=40, color_discrete_sequence=['purple'])
            fig_vol.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_vol, use_container_width=True)
            
        st.subheader("30-Day Volatility")
        fig_volat = go.Figure()
        fig_volat.add_trace(go.Scatter(x=x_axis, y=df['Volatility'], line=dict(color='red')))
        fig_volat.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_volat, use_container_width=True)
        
    with c2:
        st.subheader("Correlations")
        st.dataframe(df.corr(numeric_only=True))
        
        st.subheader("Daily Returns")
        fig_ret = px.histogram(df.dropna(subset=['Return']), x='Return', nbins=50, color_discrete_sequence=['teal'])
        fig_ret.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_ret, use_container_width=True)
        
    st.subheader("Seasonal Decomposition")
    try:
        decomp = seasonal_decompose(df[close_col].dropna(), period=30)
        fig_decomp = decomp.plot()
        fig_decomp.set_size_inches(10, 6)
        st.pyplot(fig_decomp)
    except Exception:
        st.info("Not enough data for seasonal decomposition.")


def render_models(df, close_col, date_col):
    st.header("Forecasting Models")
    
    feats = ['MA7', 'MA30', 'MA90', 'RSI', 'MACD']
    clean_df = df.dropna(subset=feats + [close_col])
    
    if len(clean_df) > 100:
        st.subheader("Machine Learning Baseline")
        X = clean_df[feats]
        y = clean_df[close_col]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        mdl_name = "XGBoost" if HAS_XGB else "RandomForest"
        model = XGBRegressor() if HAS_XGB else RandomForestRegressor(random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", mdl_name)
        c2.metric("MAE", f"{mean_absolute_error(y_te, preds):.2f}")
        c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_te, preds)):.2f}")
        
        fig = go.Figure()
        x_te_axis = clean_df.index[-len(y_te):]
        fig.add_trace(go.Scatter(x=x_te_axis, y=y_te, name='Actual'))
        fig.add_trace(go.Scatter(x=x_te_axis, y=preds, name='Predicted', line=dict(dash='dash', color='red')))
        fig.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if hasattr(model, 'feature_importances_'):
            st.write("**Feature Importance**")
            imps = model.feature_importances_
            idx = np.argsort(imps)
            fig_imp = px.bar(x=imps[idx], y=[feats[i] for i in idx], orientation='h')
            fig_imp.update_layout(height=300)
            st.plotly_chart(fig_imp, use_container_width=True)

    if st.button("Run ARIMA (30-day forecast)"):
        with st.spinner("Fitting ARIMA..."):
            try:
                tr_size = int(len(df) * 0.8)
                tr_data = df.iloc[:tr_size]
                fit = ARIMA(tr_data[close_col], order=(5,1,0)).fit()
                fcst = fit.forecast(30)
                
                last_d = df.index[-1] if date_col else len(df)
                f_dates = pd.date_range(start=last_d, periods=31, freq='B')[1:] if date_col else range(len(df), len(df)+30)
                
                fig_ar = go.Figure()
                fig_ar.add_trace(go.Scatter(x=f_dates, y=fcst, mode='lines+markers', name='Forecast', marker=dict(color='orange')))
                st.plotly_chart(fig_ar, use_container_width=True)
            except Exception as e:
                st.error(f"ARIMA failed: {e}")

    if HAS_TF and st.button("Run LSTM Neural Net"):
        with st.spinner("Training LSTM..."):
            try:
                data = df[[close_col]].dropna()
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(data)
                
                seq_len = 60
                X_lstm, y_lstm = [], []
                for i in range(seq_len, len(scaled)):
                    X_lstm.append(scaled[i-seq_len:i])
                    y_lstm.append(scaled[i])
                    
                X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                split = int(len(X_lstm) * 0.8)
                
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_lstm[:split], y_lstm[:split], epochs=3, batch_size=32, verbose=0)
                
                p = scaler.inverse_transform(model.predict(X_lstm[split:]))
                real = scaler.inverse_transform(y_lstm[split:])
                
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(y=real.flatten(), name='Actual'))
                fig_lstm.add_trace(go.Scatter(y=p.flatten(), name='Prediction', line=dict(dash='dash', color='purple')))
                st.plotly_chart(fig_lstm, use_container_width=True)
            except Exception as e:
                st.error(f"LSTM failed: {e}")


def render_strategy(df, close_col):
    st.header("Trading Strategy Backtest")
    st.write("Simple MA Crossover (7-day vs 30-day)")
    
    df['buy'] = (df['MA7'] > df['MA30']).astype(int)
    df['sell'] = (df['MA7'] < df['MA30']).astype(int)
    df['buy_sig'] = df['buy'].diff() == 1
    df['sell_sig'] = df['sell'].diff() == 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[close_col], name='Price', line=dict(color='black', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA7'], name='MA7', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], name='MA30', line=dict(color='orange', dash='dot')))
    
    buys = df[df['buy_sig']]
    sells = df[df['sell_sig']]
    fig.add_trace(go.Scatter(x=buys.index, y=buys[close_col], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=12)))
    fig.add_trace(go.Scatter(x=sells.index, y=sells[close_col], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=12)))
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # simple backtest loop
    cash = 10000.0
    pos = 0.0
    vals = []
    
    for i in range(len(df)):
        px = df[close_col].iloc[i]
        if df['buy_sig'].iloc[i] and pos == 0:
            pos = cash / px
            cash = 0
        elif df['sell_sig'].iloc[i] and pos > 0:
            cash = pos * px
            pos = 0
        vals.append(cash + pos * px)
        
    df['port_val'] = vals
    ret = ((vals[-1] - 10000) / 10000) * 100
    bh_ret = ((df[close_col].iloc[-1] - df[close_col].iloc[0]) / df[close_col].iloc[0]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Value", f"${vals[-1]:,.2f}")
    c2.metric("Return", f"{ret:.2f}%", f"{(ret - bh_ret):.2f}% vs B&H")
    c3.metric("Buy & Hold", f"{bh_ret:.2f}%")
    c4.metric("Trades", int(df['buy_sig'].sum() + df['sell_sig'].sum()))
    
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=df.index, y=df['port_val'], line=dict(color='darkgreen')))
    fig_val.update_layout(height=300, title="Portfolio Value")
    st.plotly_chart(fig_val, use_container_width=True)


def render_risk(df):
    st.header("Risk Metrics")
    rets = df['port_val'].pct_change().dropna()
    
    std = rets.std()
    sharpe = (rets.mean() - 0.01/252) / std if std != 0 else 0
    cum_max = df['port_val'].cummax()
    dd = df['port_val'] / cum_max - 1
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c2.metric("Max Drawdown", f"{dd.min():.2%}")
    c3.metric("Ann. Volatility", f"{std * np.sqrt(252):.2%}")
    
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=dd, fill='tozeroy', line=dict(color='red')))
    fig_dd.update_layout(height=350)
    st.plotly_chart(fig_dd, use_container_width=True)


def render_bulk_charts(df):
    st.header("Bulk Analysis")
    st.write("Auto-generated distributions for all numeric columns.")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    count = 0
    
    for c in num_cols:
        if count >= 30:
            st.info("Hit rendering limit (30 charts max).")
            break
            
        c1, c2 = st.columns(2)
        with c1:
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=df.index, y=df[c], line=dict(color='teal')))
            fig_l.update_layout(title=f"{c} trend", height=200, margin=dict(t=30,b=10))
            st.plotly_chart(fig_l, use_container_width=True)
            count += 1
            
        with c2:
            fig_h = px.histogram(df, x=c, nbins=30)
            fig_h.update_layout(title=f"{c} dist", height=200, margin=dict(t=30,b=10))
            st.plotly_chart(fig_h, use_container_width=True)
            count += 1
        st.markdown("---")


def main():
    st.set_page_config(page_title='Stock Analysis', layout='wide')
    st.title('Stock Analysis Dashboard')
    st.markdown("An end-to-end analysis of stock data with EDA, technical indicators, forecasts, and backtesting.")
    
    file = st.sidebar.file_uploader('Upload stock data (CSV)', type=['csv'])
    if not file:
        st.info("Please upload a CSV to begin.")
        return
        
    df = pd.read_csv(file)
    df, close_col, vol_col, date_col = preprocess_data(df)
    
    if df is None:
        st.error("No 'Close' column found in the dataset.")
        return
        
    df = add_technical_indicators(df, close_col)
    render_overview_kpis(df, close_col, vol_col)
    
    tabs = st.tabs(["EDA", "Forecasts", "Backtest", "Risk", "Bulk Charts"])
    
    with tabs[0]:
        render_eda(df, close_col, vol_col)
    with tabs[1]:
        render_models(df, close_col, date_col)
    with tabs[2]:
        render_strategy(df, close_col)
    with tabs[3]:
        render_risk(df)
    with tabs[4]:
        render_bulk_charts(df)

if __name__ == "__main__":
    main()
