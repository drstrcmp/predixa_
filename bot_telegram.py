import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
# Di bagian atas file bot_telegram.py kamu:
from predixa_core_engine import ekstrak_fitur_megaprofit

def kirim_notif_telegram(saham, harga_sekarang, rekomendasi, target):
    # Menggunakan token dan ID yang sudah kamu siapkan
    token = "8923854691:AAHA0uC3l8nQj_JS5qKpZ73xKYOttuSTUGA"
    chat_id = "5089846959"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    pesan = (
        f"🤖 *PREDIXA AI AUTOBOT* 🤖\n\n"
        f"📈 *Saham:* {saham}\n"
        f"💰 *Harga Saat Ini:* Rp {harga_sekarang:,.0f}\n"
        f"🎯 *Target AI (14 Hari):* Rp {target:,.0f}\n"
        f"⚡ *Rekomendasi:* *{rekomendasi}*\n\n"
        f"_Sistem berjalan otomatis dari server GitHub Actions._"
    )
    
    payload = {
        "chat_id": chat_id,
        "text": pesan,
        "parse_mode": "Markdown"
    }
    
    try:
        requests.post(url, data=payload)
        print(f"Notifikasi {saham} berhasil terkirim ke Telegram!")
    except Exception as e:
        print(f"Gagal mengirim Telegram untuk {saham}: {e}")

def analisa_saham(kode_saham):
    print(f"Menganalisa pergerakan {kode_saham}...")
    data = yf.download(kode_saham, period="3y", progress=False)
    
    if data.empty:
        print(f"Data {kode_saham} kosong atau tidak ditemukan.")
        return

    # Penanganan format data yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'][kode_saham].dropna()
    else:
        close_prices = data['Close'].dropna()

    df = pd.DataFrame({'Close': close_prices})
    
    # Feature Engineering (Lags)
    lags = 5
    for i in range(1, lags + 1):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    
    X = df[[f'Lag_{i}' for i in range(1, lags + 1)]]
    y = df['Close']
    
    # Training Model
    model = XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    model.fit(X, y)
    
    # Prediksi
    last_lags = X.iloc[-1].values.tolist()
    future_preds = []
    hari_ke_depan = 14
    
    for _ in range(hari_ke_depan):
        pred = model.predict(np.array([last_lags]))[0]
        future_preds.append(pred)
        last_lags = [pred] + last_lags[:-1]
        
    harga_sekarang = float(df['Close'].iloc[-1])
    harga_target = float(future_preds[-1])
    persen_perubahan = ((harga_target - harga_sekarang) / harga_sekarang) * 100
    
    # Logika Trigger Notifikasi
    if persen_perubahan > 2.0:
        rekomendasi = "BUY"
    elif persen_perubahan < -2.0:
        rekomendasi = "SELL"
    else:
        rekomendasi = "HOLD"
        
    # Bot HANYA mengirim pesan jika ada sinyal BUY atau SELL
    if rekomendasi in ["BUY", "SELL"]:
        kirim_notif_telegram(kode_saham, harga_sekarang, rekomendasi, harga_target)
    else:
        print(f"[{kode_saham}] Status HOLD. Tidak ada notifikasi yang dikirim.")

if __name__ == "__main__":
    # Daftar saham yang akan dipantau otomatis oleh bot
    daftar_pantauan = ["WIFI.JK", "ELSA.JK", "MEDC.JK", "SMRA.JK", "BSDE.JK", 
                       "BBCA.JK", "BBRI.JK", "SIDO.JK", "BJTM.JK", "TLKM.JK"]
    
    print("Memulai proses pengecekan pasar...")
    for saham in daftar_pantauan:
        analisa_saham(saham)
    print("Proses selesai.")
