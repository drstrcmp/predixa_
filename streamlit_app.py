import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import timedelta

# Konfigurasi Halaman
st.set_page_config(page_title="Predixa AI", page_icon="📈", layout="wide")

st.title("📈 Predixa AI - Prediksi Semua Saham Indonesia")
st.markdown("Cari saham, lihat prediksi arah harga, dan rekomendasi Beli/Jual berdasarkan model Machine Learning.")

# --- 1. FITUR PENCARIAN SAHAM ---
col_search, col_slider = st.columns([2, 1])
with col_search:
    kode_input = st.text_input("Masukkan Kode Saham (Contoh: BBRI, SIDO, BJTM, TLKM):", "BBRI")
with col_slider:
    hari_ke_depan = st.slider("Prediksi berapa hari ke depan?", min_value=7, max_value=30, value=14)

# Format kode saham agar sesuai dengan format Yahoo Finance untuk pasar Indonesia (.JK)
kode_saham = kode_input.upper().strip()
if not kode_saham.endswith(".JK"):
    kode_saham += ".JK"

if st.button("Analisis & Prediksi Grafik"):
    with st.spinner(f'Mengambil data {kode_saham} dan melatih model AI...'):
        
        # --- 2. AMBIL DATA REAL-TIME ---
        # Mengambil data 3 tahun terakhir agar model punya cukup pola untuk dipelajari
        data = yf.download(kode_saham, period="3y", progress=False)
        
        if data.empty:
            st.error("Saham tidak ditemukan. Pastikan kode saham yang dimasukkan benar.")
        else:
            # Menangani potensi format MultiIndex dari update yfinance terbaru
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close'][kode_saham].dropna()
            else:
                close_prices = data['Close'].dropna()

            df = pd.DataFrame({'Close': close_prices})
            
            # --- 3. FEATURE ENGINEERING (Autoregressive Lags) ---
            # Menggunakan 5 hari ke belakang (lag) untuk memprediksi hari esoknya
            lags = 5
            for i in range(1, lags + 1):
                df[f'Lag_{i}'] = df['Close'].shift(i)
            
            df.dropna(inplace=True)
            
            X = df[[f'Lag_{i}' for i in range(1, lags + 1)]]
            y = df['Close']
            
            # --- 4. TRAINING MODEL OTOMATIS ---
            # Menggunakan Regressor (bukan Classifier) agar bisa menebak harga absolut, bukan sekadar 1 atau 0
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            model.fit(X, y)
            
            # --- 5. PREDIKSI MASA DEPAN (Iteratif) ---
            last_lags = X.iloc[-1].values.tolist()
            future_preds = []
            
            for _ in range(hari_ke_depan):
                pred = model.predict(np.array([last_lags]))[0]
                future_preds.append(pred)
                # Geser lag: masukkan hasil prediksi terbaru ke lag 1, dan buang lag paling lama
                last_lags = [pred] + last_lags[:-1]
                
            # Siapkan sumbu X (Tanggal) untuk grafik masa depan
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, hari_ke_depan + 1)]
            
            # --- 6. LOGIKA REKOMENDASI BUY/SELL ---
            harga_sekarang = float(df['Close'].iloc[-1])
            harga_target = float(future_preds[-1])
            persentase_perubahan = ((harga_target - harga_sekarang) / harga_sekarang) * 100
            
            # Tentukan batas threshold untuk Buy/Sell/Hold
            if persentase_perubahan > 1.5:
                rekomendasi = "BUY"
                warna_indikator = "normal"
            elif persentase_perubahan < -1.5:
                rekomendasi = "SELL"
                warna_indikator = "inverse"
            else:
                rekomendasi = "HOLD"
                warna_indikator = "off"
                
            # Tampilkan metrik utama
            st.divider()
            st.subheader(f"Hasil Analisis: {kode_input.upper()}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Harga Saat Ini", f"Rp {harga_sekarang:,.0f}")
            with col2:
                st.metric(f"Target ({hari_ke_depan} Hari)", f"Rp {harga_target:,.0f}", f"{persentase_perubahan:+.2f}%", delta_color=warna_indikator)
            with col3:
                st.metric("Rekomendasi AI", rekomendasi)
                
            # --- 7. VISUALISASI GRAFIK INTERAKTIF ---
            fig = go.Figure()
            
            # Plot 100 hari historis terakhir agar grafik masa depan lebih jelas terlihat (tidak tenggelam oleh data 3 tahun)
            hist_plot = df.iloc[-100:]
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Close'], mode='lines', name='Harga Historis', line=dict(color='#00b4d8', width=2)))
            
            # Gabungkan titik terakhir historis ke prediksi pertama agar garis tidak putus di grafik
            future_x = [hist_plot.index[-1]] + future_dates
            future_y = [hist_plot['Close'].iloc[-1]] + future_preds
            
            fig.add_trace(go.Scatter(x=future_x, y=future_y, mode='lines', name='Prediksi AI', line=dict(color='#ff9f1c', width=2, dash='dash')))
            
            fig.update_layout(
                title=f'Pergerakan dan Prediksi Harga {kode_input.upper()}',
                xaxis_title='Tanggal',
                yaxis_title='Harga (Rupiah)',
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            # Tampilkan grafik
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Catatan Analisis: Sistem memprediksi menggunakan Autoregressive XGBoost. Dalam realitanya, pasar saham sangat fluktuatif. Gunakan data ini sebagai referensi pelengkap metodemu, bukan acuan mutlak finansial.")
