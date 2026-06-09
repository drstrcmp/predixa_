import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import timedelta
import requests

# --- FUNGSI NOTIFIKASI TELEGRAM ---
def kirim_notif_telegram(saham, harga_sekarang, rekomendasi, target):
    token = "8923854691:AAHA0uC3l8nQj_JS5qKpZ73xKYOttuSTUGA"
    chat_id = "5089846959"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    pesan = (
        f"🤖 *PREDIXA AI ALERT* 🤖\n\n"
        f"📈 *Saham:* {saham}\n"
        f"💰 *Harga Saat Ini:* Rp {harga_sekarang:,.0f}\n"
        f"🎯 *Target AI:* Rp {target:,.0f}\n"
        f"⚡ *Rekomendasi:* *{rekomendasi}*\n\n"
        f"_Silakan cek dashboard web untuk grafik detailnya!_"
    )
    
    payload = {
        "chat_id": chat_id,
        "text": pesan,
        "parse_mode": "Markdown"
    }
    
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.toast(f"Gagal mengirim Telegram: {e}")

# 1. KONFIGURASI HALAMAN UTAMA
st.set_page_config(
    page_title="Predixa AI - Premium Market Dashboard", 
    page_icon="📈", 
    layout="wide"
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .header-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 2.5rem; border-radius: 16px; margin-bottom: 2rem;
            border: 1px solid #334155; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .header-title { color: #f8fafc; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }
        .header-subtitle { color: #94a3b8; font-size: 1rem; font-weight: 400; }
        .section-card { background-color: #0f172a; padding: 1.5rem; border-radius: 12px; border: 1px solid #1e293b; margin-bottom: 1.5rem; }
        .stButton>button {
            background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%) !important;
            color: white !important; border: none !important; padding: 0.6rem 2rem !important;
            border-radius: 8px !important; font-weight: 600 !important; transition: all 0.3s ease !important;
        }
        .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4) !important; }
    </style>
""", unsafe_allow_html=True)

# 3. HEADER
st.markdown("""
    <div class="header-container">
        <div class="header-title">📈 Predixa AI</div>
        <div class="header-subtitle">Platform analisis cerdas dengan integrasi notifikasi otomatis Telegram.</div>
    </div>
""", unsafe_allow_html=True)

# 4. MONITORING PASAR UTAMA (IHSG)
st.subheader("📊 Kondisi Pasar Terkini (IHSG)")
@st.cache_data(ttl=600)
def ambil_data_ihsg():
    return yf.download("^JKSE", period="1y", progress=False)

try:
    df_ihsg = ambil_data_ihsg()
    if not df_ihsg.empty:
        if isinstance(df_ihsg.columns, pd.MultiIndex):
            ihsg_close = df_ihsg['Close']['^JKSE'].dropna()
        else:
            ihsg_close = df_ihsg['Close'].dropna()
            
        ihsg_sekarang = float(ihsg_close.iloc[-1])
        ihsg_sebelumnya = float(ihsg_close.iloc[-2]) if len(ihsg_close) > 1 else ihsg_sekarang
        perubahan = ihsg_sekarang - ihsg_sebelumnya
        persen = (perubahan / ihsg_sebelumnya) * 100
        
        col_m1, col_m2 = st.columns([1, 4])
        with col_m1:
            st.metric("Indeks Harga Saham Gabungan", f"{ihsg_sekarang:,.2f}", f"{perubahan:+.2f} ({persen:+.2f}%)")
        
        with col_m2:
            fig_ihsg = go.Figure()
            fig_ihsg.add_trace(go.Scatter(
                x=ihsg_close.index, y=ihsg_close.values, mode='lines', 
                line=dict(color='#10b981' if perubahan >= 0 else '#ef4444', width=2),
                fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.05)' if perubahan >= 0 else 'rgba(239, 68, 68, 0.05)'
            ))
            fig_ihsg.update_layout(
                height=180, margin=dict(l=20, r=20, t=10, b=10), template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, color='#64748b'), yaxis=dict(showgrid=True, gridcolor='#1e293b', color='#64748b', side='right')
            )
            st.plotly_chart(fig_ihsg, use_container_width=True, config={'displayModeBar': False})
except Exception as e:
    st.error(f"Koneksi terganggu: {e}")

st.divider()

# 5. PANEL PENCARIAN & FITUR PREDIKSI SAHAM
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔍 Cari, Prediksi, & Kirim Sinyal")

col_search, col_slider = st.columns([2, 1])
with col_search:
    kode_input = st.text_input("Masukkan Kode Saham Perusahaan (Contoh: BBRI, SIDO, BJTM, TLKM):", "BBRI")
with col_slider:
    hari_ke_depan = st.slider("Durasi proyeksi AI (hari kedepan):", min_value=7, max_value=30, value=14)

kode_saham = kode_input.upper().strip()
if not kode_saham.endswith(".JK"):
    kode_saham += ".JK"
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Jalankan Algoritma & Cek Menitan"):
    with st.spinner(f'Mengambil data Harian & Menitan untuk {kode_saham}...'):
        # Ambil data harian untuk AI
        data_daily = yf.download(kode_saham, period="3y", progress=False)
        # Ambil data menitan (intraday) untuk akurasi entry point hari ini
        data_minute = yf.download(kode_saham, period="1d", interval="1m", progress=False)
        
        if data_daily.empty:
            st.error("Kode saham tidak valid.")
        else:
            if isinstance(data_daily.columns, pd.MultiIndex):
                close_prices = data_daily['Close'][kode_saham].dropna()
            else:
                close_prices = data_daily['Close'].dropna()

            df = pd.DataFrame({'Close': close_prices})
            
            # Machine Learning Model
            lags = 5
            for i in range(1, lags + 1):
                df[f'Lag_{i}'] = df['Close'].shift(i)
            df.dropna(inplace=True)
            X = df[[f'Lag_{i}' for i in range(1, lags + 1)]]
            y = df['Close']
            
            model = XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
            model.fit(X, y)
            
            last_lags = X.iloc[-1].values.tolist()
            future_preds = []
            for _ in range(hari_ke_depan):
                pred = model.predict(np.array([last_lags]))[0]
                future_preds.append(pred)
                last_lags = [pred] + last_lags[:-1]
                
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, hari_ke_depan + 1)]
            
            # Logika Buy/Sell
            harga_sekarang = float(df['Close'].iloc[-1])
            harga_target = float(future_preds[-1])
            persen_perubahan = ((harga_target - harga_sekarang) / harga_sekarang) * 100
            
            if persen_perubahan > 2.0:
                rekomendasi = "BUY"
                warna = "normal"
            elif persen_perubahan < -2.0:
                rekomendasi = "SELL"
                warna = "inverse"
            else:
                rekomendasi = "HOLD"
                warna = "off"

            # Trigger Notifikasi Telegram jika Buy/Sell
            if rekomendasi in ["BUY", "SELL"]:
                kirim_notif_telegram(kode_input.upper(), harga_sekarang, rekomendasi, harga_target)
                st.toast("Sinyal Telegram berhasil dikirim ke perangkat Anda!", icon="📱")
                
            # Render Metrik
            st.markdown(f"### 📋 Panel Eksekutif: {kode_input.upper()}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Harga Terakhir", f"Rp {harga_sekarang:,.0f}")
            c2.metric(f"Target ({hari_ke_depan} Hari)", f"Rp {harga_target:,.0f}", f"{persen_perubahan:+.2f}%", delta_color=warna)
            c3.metric("Rekomendasi Strategi AI", f"🚀 {rekomendasi}" if rekomendasi == "BUY" else (f"🚨 {rekomendasi}" if rekomendasi == "SELL" else f"⚖️ {rekomendasi}"))
            
            # TAB SISTEM UNTUK GRAFIK
            tab1, tab2 = st.tabs(["📈 Proyeksi AI (Harian)", "⏱️ Live Menit per Menit (Intraday)"])
            
            with tab1:
                fig_pred = go.Figure()
                hist_plot = df.iloc[-80:]
                fig_pred.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Close'], mode='lines', name='Harga Riil', line=dict(color='#0ea5e9', width=2.5)))
                
                pred_x = [hist_plot.index[-1]] + future_dates
                pred_y = [hist_plot['Close'].iloc[-1]] + future_preds
                fig_pred.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines+markers', name='Estimasi AI', line=dict(color='#f97316', width=2, dash='dash'), marker=dict(size=4)))
                
                fig_pred.update_layout(
                    title='Proyeksi Arah Tren Harian', template='plotly_dark', hovermode='x unified', height=400,
                    paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19',
                    xaxis=dict(showgrid=True, gridcolor='#1e293b'), yaxis=dict(showgrid=True, gridcolor='#1e293b'),
                    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02, bgcolor='rgba(15, 23, 42, 0.8)')
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
            with tab2:
                if not data_minute.empty:
                    if isinstance(data_minute.columns, pd.MultiIndex):
                        minute_close = data_minute['Close'][kode_saham].dropna()
                    else:
                        minute_close = data_minute['Close'].dropna()
                        
                    fig_min = go.Figure()
                    fig_min.add_trace(go.Scatter(
                        x=minute_close.index, y=minute_close.values, mode='lines', name='Harga Menitan',
                        line=dict(color='#10b981', width=1.5), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)'
                    ))
                    fig_min.update_layout(
                        title=f'Pergerakan Harga Hari Ini (Interval 1 Menit) - Cocok untuk Entry Beli/Jual',
                        template='plotly_dark', hovermode='x unified', height=400,
                        paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19',
                        xaxis=dict(showgrid=True, gridcolor='#1e293b'), yaxis=dict(showgrid=True, gridcolor='#1e293b')
                    )
                    st.plotly_chart(fig_min, use_container_width=True)
                    st.caption("Gunakan grafik menit ini untuk mencari harga terbaik (kapan turun/naik dalam hari ini) sebelum melakukan eksekusi di broker/aplikasi investasi.")
                else:
                    st.info("Data menit per menit belum tersedia. Bursa mungkin sedang tutup atau data tertunda.")
