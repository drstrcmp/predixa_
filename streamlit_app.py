import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import timedelta

# 1. KONFIGURASI HALAMAN UTAMA
st.set_page_config(
    page_title="Predixa AI - Premium Market Dashboard", 
    page_icon="📈", 
    layout="wide"
)

# 2. CUSTOM CSS UNTUK MEMPERCANTIK TAMPILAN (Tetap Ringan & Responsif)
st.markdown("""
    <style>
        /* Gaya Font & Background Dasar */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Desain Header Banner */
        .header-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            border: 1px solid #334155;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .header-title {
            color: #f8fafc;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .header-subtitle {
            color: #94a3b8;
            font-size: 1rem;
            font-weight: 400;
        }
        
        /* Desain Kartu Kontainer Kustom */
        .section-card {
            background-color: #0f172a;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #1e293b;
            margin-bottom: 1.5rem;
        }
        
        /* Animasi halu pada tombol analisis */
        .stButton>button {
            background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.6rem 2rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2) !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4) !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. HEADER BANNER UTAMA
st.markdown("""
    <div class="header-container">
        <div class="header-title">📈 Predixa AI</div>
        <div class="header-subtitle">Platform analisis cerdas bertenaga Machine Learning untuk proyeksi arah pergerakan pasar saham Indonesia.</div>
    </div>
""", unsafe_allow_html=True)

# 4. MONITORING PASAR UTAMA: GRAFIK IHSG (JKSE)
st.subheader("📊 Kondisi Pasar Terkini (IHSG)")

# Ambil data IHSG secara realtime lewat cache agar loading sangat cepat
@st.cache_data(ttl=600)
def ambil_data_ihsg():
    ihsg_data = yf.download("^JKSE", period="1y", progress=False)
    return ihsg_data

try:
    df_ihsg = ambil_data_ihsg()
    if not df_ihsg.empty:
        # Menangani format MultiIndex dari update yfinance terbaru jika ada
        if isinstance(df_ihsg.columns, pd.MultiIndex):
            ihsg_close = df_ihsg['Close']['^JKSE'].dropna()
            ihsg_open = df_ihsg['Open']['^JKSE'].dropna()
        else:
            ihsg_close = df_ihsg['Close'].dropna()
            ihsg_open = df_ihsg['Open'].dropna()
            
        ihsg_sekarang = float(ihsg_close.iloc[-1])
        ihsg_sebelumnya = float(ihsg_close.iloc[-2]) if len(ihsg_close) > 1 else ihsg_sekarang
        perubahan_harga = ihsg_sekarang - ihsg_sebelumnya
        persen_perubahan = (perubahan_harga / ihsg_sebelumnya) * 100
        
        # Tampilkan Metrik IHSG dengan visual bersih
        col_m1, col_m2 = st.columns([1, 4])
        with col_m1:
            st.metric(
                label="Indeks Harga Saham Gabungan",
                value=f"{ihsg_sekarang:,.2f}",
                delta=f"{perubahan_harga:+.2f} ({persen_perubahan:+.2f}%)"
            )
        
        with col_m2:
            # Grafik IHSG Premium Style
            fig_ihsg = go.Figure()
            fig_ihsg.add_trace(go.Scatter(
                x=ihsg_close.index, 
                y=ihsg_close.values, 
                mode='lines', 
                name='IHSG', 
                line=dict(color='#10b981' if perubahan_harga >= 0 else '#ef4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.05)' if perubahan_harga >= 0 else 'rgba(239, 68, 68, 0.05)'
            ))
            fig_ihsg.update_layout(
                height=180,
                margin=dict(l=20, r=20, t=10, b=10),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, colors='#64748b'),
                yaxis=dict(showgrid=True, gridcolor='#1e293b', colors='#64748b', side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(fig_ihsg, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("Gagal memuat data grafik IHSG saat ini.")
except Exception as e:
    st.error(f"Koneksi ke data pasar terganggu: {e}")

st.divider()

# 5. PANEL PENCARIAN & FITUR PREDIKSI SAHAM SPESIFIK
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔍 Cari & Prediksi Tren Saham Perusahaan")

col_search, col_slider = st.columns([2, 1])
with col_search:
    kode_input = st.text_input("Masukkan Kode Saham Perusahaan (Contoh: BBRI, SIDO, BJTM, TLKM):", "BBRI")
with col_slider:
    hari_ke_depan = st.slider("Durasi proyeksi jangka pendek (hari kedepan):", min_value=7, max_value=30, value=14)

kode_saham = kode_input.upper().strip()
if not kode_saham.endswith(".JK"):
    kode_saham += ".JK"

st.markdown('</div>', unsafe_allow_html=True)

# Prosedur Analisis Utama saat tombol ditekan
if st.button("Jalankan Algoritma Prediksi"):
    with st.spinner(f'Mengkalibrasi ulang model Machine Learning untuk {kode_saham}...'):
        
        # Penarikan Data Historis Saham terkait
        data = yf.download(kode_saham, period="3y", progress=False)
        
        if data.empty:
            st.error("Kode saham tidak valid atau tidak terdaftar di bursa.")
        else:
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close'][kode_saham].dropna()
                open_prices = data['Open'][kode_saham].dropna()
                high_prices = data['High'][kode_saham].dropna()
                low_prices = data['Low'][kode_saham].dropna()
            else:
                close_prices = data['Close'].dropna()
                open_prices = data['Open'].dropna()
                high_prices = data['High'].dropna()
                low_prices = data['Low'].dropna()

            df = pd.DataFrame({'Close': close_prices})
            
            # Feature Engineering: Autoregressive Lags (Menggunakan data pola 5 hari ke belakang)
            lags = 5
            for i in range(1, lags + 1):
                df[f'Lag_{i}'] = df['Close'].shift(i)
            
            df.dropna(inplace=True)
            
            X = df[[f'Lag_{i}' for i in range(1, lags + 1)]]
            y = df['Close']
            
            # Pelatihan Model Regressor secara instan dan dinamis
            model = XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
            model.fit(X, y)
            
            # Estimasi Proyeksi Iteratif Masa Depan
            last_lags = X.iloc[-1].values.tolist()
            future_preds = []
            
            for _ in range(hari_ke_depan):
                pred = model.predict(np.array([last_lags]))[0]
                future_preds.append(pred)
                last_lags = [pred] + last_lags[:-1]
                
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, hari_ke_depan + 1)]
            
            # Logika Algoritma Penentuan Rekomendasi Aksi (Buy / Sell / Hold)
            harga_sekarang = float(df['Close'].iloc[-1])
            harga_target = float(future_preds[-1])
            persentase_perubahan = ((harga_target - harga_sekarang) / harga_sekarang) * 100
            
            if persentase_perubahan > 2.0:
                rekomendasi = "🚀 BUY"
                warna_indikator = "normal"
            elif persentase_perubahan < -2.0:
                rekomendasi = "🚨 SELL"
                warna_indikator = "inverse"
            else:
                rekomendasi = "⚖️ HOLD"
                warna_indikator = "off"
                
            # Render Informasi Metrik Hasil Analisis
            st.markdown(f"### 📋 Panel Eksekutif Hasil Proyeksi: {kode_input.upper()}")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Harga Penutupan Terakhir", f"Rp {harga_sekarang:,.0f}")
            with col_res2:
                st.metric(f"Target Harga ({hari_ke_depan} Hari)", f"Rp {harga_target:,.0f}", f"{persentase_perubahan:+.2f}%", delta_color=warna_indikator)
            with col_res3:
                st.metric("Rekomendasi Strategi AI", rekomendasi)
                
            # Pembuatan Grafik Proyeksi Candlestick + Line Gabungan Premium
            fig_pred = go.Figure()
            
            # Batasi visual historis 80 poin ke belakang agar grafik tetap fokus dan sedap dipandang
            hist_plot = df.iloc[-80:]
            
            # Ambil potongan data candle pendukung grafik historis
            if isinstance(data.columns, pd.MultiIndex):
                hist_open = open_prices.loc[hist_plot.index]
                hist_high = high_prices.loc[hist_plot.index]
                hist_low = low_prices.loc[hist_plot.index]
            else:
                hist_open = open_prices.loc[hist_plot.index]
                hist_high = high_prices.loc[hist_plot.index]
                hist_low = low_prices.loc[hist_plot.index]
            
            # Plot data historis asli dalam bentuk area line premium yang bersih
            fig_pred.add_trace(go.Scatter(
                x=hist_plot.index, 
                y=hist_plot['Close'], 
                mode='lines', 
                name='Harga Riil Pasar', 
                line=dict(color='#0ea5e9', width=2.5)
            ))
            
            # Hubungkan ujung grafik historis dengan titik awal proyeksi agar mulus
            pred_x_axis = [hist_plot.index[-1]] + future_dates
            pred_y_axis = [hist_plot['Close'].iloc[-1]] + future_preds
            
            # Plot garis prediksi masa depan dengan pola putus-putus oranye terang
            fig_pred.add_trace(go.Scatter(
                x=pred_x_axis, 
                y=pred_y_axis, 
                mode='lines+markers', 
                name='Estimasi Jalur AI', 
                line=dict(color='#f97316', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            # Percantik layout Axis & Grid Chart agar terlihat berkelas dunia
            fig_pred.update_layout(
                title=f'Peta Proyeksi Arah Tren Nilai Saham {kode_input.upper()}',
                xaxis_title='Garis Waktu (Tanggal)',
                yaxis_title='Nilai Konversi (Rupiah)',
                template='plotly_dark',
                hovermode='x unified',
                height=450,
                paper_bgcolor='#0b0f19',
                plot_bgcolor='#0b0f19',
                xaxis=dict(showgrid=True, gridcolor='#1e293b', linewidth=1, linecolor='#334155'),
                yaxis=dict(showgrid=True, gridcolor='#1e293b', linewidth=1, linecolor='#334155'),
                legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02, bgcolor='rgba(15, 23, 42, 0.8)')
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.caption("🚨 Disclaimer Informasi: Prediksi kalkulasi ini diproses secara otomatis berbasis algoritma XGBoost Regressor autoregresif. Pergerakan pasar bursa riil dipengaruhi volatilitas berita makro ekonomi global. Selalu kombinasikan dengan manajemen risiko finansial mandiri.")
