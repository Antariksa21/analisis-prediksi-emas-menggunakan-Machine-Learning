"""
Aura Gold Backend - FastAPI API on Vercel Serverless
Author: Antigravity AI
Description: Endpoint GET /api/predict untuk mengambil data emas & kurs live,
             melatih model Random Forest secara on-the-fly, dan mengembalikan data dalam format JSON.
             Mendukung simulasi sinyal pasar lewat parameter query (?force_signal=sell|hold|buy).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aura Gold API", description="API untuk Prediksi Harga Emas & ROI Tracker")

# Aktifkan CORS agar frontend murni (atau local port berbeda) bisa melakukan fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_fallback_prediction(error_msg: str):
    """Membuat data prediksi simulasi jika koneksi ke Yahoo Finance gagal atau dibatasi."""
    logger.warning(f"Menggunakan data fallback karena: {error_msg}")
    
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Hari kerja
    
    n_days = len(date_range)
    # Mulai dari harga Rp 1.250.000 per gram
    prices = [1250000.0]
    for _ in range(n_days - 1):
        # Simulasi pergerakan acak harian dengan kecenderungan naik
        change = np.random.normal(0.0004, 0.0095)
        prices.append(prices[-1] * (1 + change))
        
    df = pd.DataFrame(index=date_range)
    df['Close'] = prices
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    
    history_data = []
    for date, row in df.iterrows():
        history_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": float(row['Close']),
            "ma7": float(row['MA_7']) if not np.isnan(row['MA_7']) else None,
            "ma21": float(row['MA_21']) if not np.isnan(row['MA_21']) else None
        })
        
    today_close = float(df['Close'].iloc[-1])
    yesterday_close = float(df['Close'].iloc[-2])
    
    # Prediksi besok
    pred_change = np.random.normal(0.0015, 0.006)
    predicted_tomorrow = today_close * (1 + pred_change)
    
    return {
        "success": True,
        "is_mock": True,
        "exchange_rate": 16250.0,
        "today_price": today_close,
        "yesterday_price": yesterday_close,
        "predicted_tomorrow": float(predicted_tomorrow),
        "metrics": {
            "r2": 0.8942,
            "mape": 0.96
        },
        "history": history_data,
        "error_detail": error_msg
    }

def fetch_and_train_prediction():
    """Mengambil data live, memproses fitur, dan melatih Random Forest untuk prediksi."""
    # 1. Unduh data historis
    # GC=F (Emas Futures USD / Troy Ounce)
    gold_ticker = yf.Ticker("GC=F")
    gold_df = gold_ticker.history(period="5y")
    
    # USDIDR=X (Kurs USD ke Rupiah)
    usdidr_ticker = yf.Ticker("USDIDR=X")
    usdidr_df = usdidr_ticker.history(period="5y")
    
    if gold_df.empty or usdidr_df.empty:
        raise ValueError("Data historis tidak dapat diunduh dari Yahoo Finance.")
        
    # Lokalkan index timezone agar naive
    gold_df.index = gold_df.index.tz_localize(None)
    usdidr_df.index = usdidr_df.index.tz_localize(None)
    
    # Urutkan secara kronologis
    gold_df = gold_df.sort_index()
    usdidr_df = usdidr_df.sort_index()
    
    # Gabungkan data berdasarkan index tanggal (karena hari bursa mungkin sedikit berbeda)
    merged_df = pd.merge(
        gold_df[['Close', 'Open', 'High', 'Low', 'Volume']], 
        usdidr_df[['Close']], 
        left_index=True, 
        right_index=True, 
        how='left', 
        suffixes=('', '_Rate')
    )
    
    # Isi data kurs yang kosong (forward fill dan backward fill)
    merged_df['Close_Rate'] = merged_df['Close_Rate'].ffill().bfill()
    
    # Konversi USD/troy ounce ke IDR/gram
    # 1 troy ounce = 31.1034768 gram
    G_PER_OZ = 31.1034768
    
    merged_df['Close_IDR'] = (merged_df['Close'] * merged_df['Close_Rate']) / G_PER_OZ
    merged_df['Open_IDR'] = (merged_df['Open'] * merged_df['Close_Rate']) / G_PER_OZ
    merged_df['High_IDR'] = (merged_df['High'] * merged_df['Close_Rate']) / G_PER_OZ
    merged_df['Low_IDR'] = (merged_df['Low'] * merged_df['Close_Rate']) / G_PER_OZ
    
    # Buat dataset utama dalam IDR
    df = merged_df[['Open_IDR', 'High_IDR', 'Low_IDR', 'Close_IDR', 'Volume']].copy()
    df.rename(columns={
        'Open_IDR': 'Open',
        'High_IDR': 'High',
        'Low_IDR': 'Low',
        'Close_IDR': 'Close'
    }, inplace=True)
    
    # 2. Rekayasa Fitur (Feature Engineering)
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_2'] = df['Close'].shift(2)
    df['Close_Lag_3'] = df['Close'].shift(3)
    df['Close_Lag_5'] = df['Close'].shift(5)
    
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_5d'] = df['Daily_Return'].rolling(window=5).std()
    
    # Target: Harga esok hari
    df['Target'] = df['Close'].shift(-1)
    
    # Hapus NaN untuk training
    df_ml = df.dropna().copy()
    
    feature_cols = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
        'MA_7', 'MA_21', 'Daily_Return', 'Volatility_5d'
    ]
    
    X = df_ml[feature_cols]
    y = df_ml['Target']
    
    if len(X) < 50:
        raise ValueError("Jumlah data latih terlalu sedikit.")
        
    # Split 80/20 secara kronologis
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Latih model evaluasi
    eval_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    
    # Hitung metrik akurasi
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Latih model final pada semua data
    final_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    
    # Prediksi besok berdasarkan data hari ini
    today_features = df.iloc[-1:][feature_cols]
    predicted_tomorrow = final_model.predict(today_features)[0]
    
    # Potong data historis untuk grafik (terakhir 252 hari perdagangan ~ 1 tahun bursa)
    history_df = df.tail(252).copy()
    
    history_data = []
    for date, row in history_df.iterrows():
        history_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": float(row['Close']),
            "ma7": float(row['MA_7']) if not np.isnan(row['MA_7']) else None,
            "ma21": float(row['MA_21']) if not np.isnan(row['MA_21']) else None
        })
        
    today_close = float(df['Close'].iloc[-1])
    yesterday_close = float(df['Close'].iloc[-2])
    exchange_rate = float(merged_df['Close_Rate'].iloc[-1])
    
    return {
        "success": True,
        "is_mock": False,
        "exchange_rate": exchange_rate,
        "today_price": today_close,
        "yesterday_price": yesterday_close,
        "predicted_tomorrow": float(predicted_tomorrow),
        "metrics": {
            "r2": float(r2),
            "mape": float(mape)
        },
        "history": history_data
    }

@app.get("/api/predict")
def predict_endpoint(force_signal: str = None):
    """Endpoint utama untuk mengambil hasil analisis harga emas dengan modifikasi sinyal opsional."""
    try:
        data = fetch_and_train_prediction()
    except Exception as e:
        logger.error(f"Gagal melakukan prediksi live: {str(e)}")
        data = generate_fallback_prediction(str(e))
        
    # Manipulasi harga besok jika parameter force_signal diaktifkan untuk demo
    if force_signal:
        force_signal = force_signal.lower()
        today_price = data["today_price"]
        if force_signal == "sell":
            # Turun 1% untuk memicu sinyal JUAL (< -0.4%)
            data["predicted_tomorrow"] = today_price * 0.99
        elif force_signal == "hold":
            # Naik tipis 0.05% untuk memicu sinyal HOLD (di antara -0.4% dan 0.4%)
            data["predicted_tomorrow"] = today_price * 1.0005
        elif force_signal == "buy":
            # Naik 1% untuk memicu sinyal BELI (> 0.4%)
            data["predicted_tomorrow"] = today_price * 1.01
            
    return data

@app.get("/api/health")
def health_endpoint():
    """Endpoint status pengecekan API."""
    return {"status": "healthy", "time": datetime.now().isoformat()}
