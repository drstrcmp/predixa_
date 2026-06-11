import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def ambil_data_bursa(ticker, start_date="2020-01-01"):
    """Mengambil data historis OHLCV dari Yahoo Finance"""
    print(f"[*] Mengunduh data historis untuk {ticker}...")
    df = yf.download(ticker, start=start_date)
    return df

def ekstrak_fitur_megaprofit(df):
    """
    FEATURE ENGINEERING ENGINE
    Menerjemahkan 11 Materi Analisis Teknikal menjadi Fitur Matematika ML
    """
    df = df.copy()
    
    # --- MATERI 8: MOVING AVERAGE & CROSSOVER ---
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    # Fitur Crossover: 1 jika EMA cepat di atas EMA lambat (Uptrend)
    df['MA_Crossover'] = np.where(df['EMA20'] > df['EMA50'], 1, 0)
    
    # --- MATERI 6: RELATIVE STRENGTH INDEX (RSI) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # --- MATERI 7: STOCHASTIC OSCILLATOR ---
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9)) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # --- MATERI 1 & BUKU EDIANTO ONG: ANATOMI CANDLESTICK & VOLUME ---
    body = (df['Close'] - df['Open']).abs()
    candle_range = df['High'] - df['Low'] + 1e-9
    df['Is_Doji'] = np.where(body / candle_range < 0.1, 1, 0)
    df['Is_Marubozu'] = np.where(body / candle_range > 0.85, 1, 0)
    
    # Volume Trend (Teori Dow: Akumulasi ditandai Volume Naik saat Harga Menguat)
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Vol_Strong'] = np.where(df['Volume'] > df['Vol_MA5'], 1, 0)
    
    # --- MATERI 2 & 10: SUPPORT/RESISTANCE ROLLING & FIBONACCI RETRACEMENT ---
    df['Roll_High20'] = df['High'].rolling(window=20).max()
    df['Roll_Low20'] = df['Low'].rolling(window=20).min()
    
    # Kalkulasi Garis Golden Ratio Fibonacci
    range_20 = df['Roll_High20'] - df['Roll_Low20'] + 1e-9
    df['Fib_382'] = df['Roll_High20'] - (0.382 * range_20)
    df['Fib_500'] = df['Roll_High20'] - (0.500 * range_20)
    df['Fib_618'] = df['Roll_High20'] - (0.618 * range_20)
    
    # Fitur Jarak Harga Terhadap Dinding Psikologis S/R
    df['Dist_to_Resist'] = (df['Roll_High20'] - df['Close']) / df['Close']
    df['Dist_to_Support'] = (df['Close'] - df['Roll_Low20']) / df['Close']
    
    # Bersihkan baris awal yang kosong akibat perhitungan rolling window
    return df.dropna()

def buat_target_label(df, horizon=5, threshold=0.015):
    """
    MANAJEMEN FILTER STRATEGIS (Buku Edianto Ong - Valid Filter 1.5%)
    Mengubah target menjadi 3 Kelas Klasifikasi:
    0 = HOLD (Abaikan noise fluktuasi kecil)
    1 = BUY (Prediksi naik tajam melebihi +1.5%)
    2 = SELL (Prediksi turun tajam melebihi -1.5%)
    """
    df = df.copy()
    # Mengukur imbal hasil ke depan sesuai horizon hari
    df['Forward_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    
    conditions = [
        (df['Forward_Return'] > threshold),       # Naik > 1.5% -> BUY
        (df['Forward_Return'] < -threshold)       # Turun > 1.5% -> SELL
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0) # Sisanya -> HOLD
    
    return df.dropna()

def latih_engine_predixa(df):
    """Melatih Model XGBoost dengan Hyperparameter Anti-Overfitting"""
    # Tentukan fitur yang akan dibaca oleh otak AI
    fitur_kolom = [
        'MA_Crossover', 'RSI14', 'Stoch_K', 'Stoch_D', 
        'Is_Doji', 'Is_Marubozu', 'Vol_Strong', 
        'Dist_to_Resist', 'Dist_to_Support', 'Fib_382', 'Fib_500', 'Fib_618'
    ]
    
    X = df[fitur_kolom]
    y = df['Target']
    
    # Split Data: 80% Training, 20% Testing (Secara kronologis berurutan)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, shuffle=False)
    
    # Konfigurasi XGBoost khusus untuk Multiclass dengan regularisasi ketat
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,              # Batasi kedalaman pohon agar tidak menghafal data kotor
        learning_rate=0.05,
        subsample=0.8,            # Gunakan 80% baris acak untuk mencegah overfitting
        colsample_bytree=0.8,     # Gunakan 80% fitur acak per pohon
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
    )
    
    print("[*] Memulai proses training XGBoost...")
    model.fit(X_train, y_train)
    
    # Pengujian Ketajaman Akurasi Model
    y_pred = model.predict(X_test)
    skor_akurasi = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"[ SUCCESS ] AKURASI BARU PREDIXA ENGINE: {skor_accuracy * 100:.2f}%")
    print("="*50)
    print("\nLaporan Klasifikasi Detal:")
    print(classification_report(y_test, y_pred, target_names=['HOLD (0)', 'BUY (1)', 'SELL (2)']))
    
    return model

# --- RUNNER SIMULASI LABORATORIUM ---
if __name__ == "__main__":
    # Kita uji coba menggunakan kelinci percobaan utama kita: ELSA.JK
    data_mentah = ambil_data_bursa("ELSA.JK", start_date="2020-01-01")
    data_fitur = ekstrak_fitur_megaprofit(data_mentah)
    data_final = buat_target_label(data_fitur, horizon=5, threshold=0.015)
    
    # Jalankan training
    model_terlatih = latih_engine_predixa(data_final)
