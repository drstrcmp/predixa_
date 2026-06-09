import streamlit as st

st.set_page_config(
    page_title="Predixa AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Predixa AI")

st.subheader("Prediksi Saham Indonesia")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("BBCA", "BUY", "+2.3%")

with col2:
    st.metric("BBRI", "BUY", "+1.8%")

with col3:
    st.metric("TLKM", "SELL", "-0.9%")
