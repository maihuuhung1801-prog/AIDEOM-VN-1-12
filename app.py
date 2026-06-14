# -*- coding: utf-8 -*-
import streamlit as st

from modules import bai01, bai02, bai03, bai04, bai05, bai06, bai07, bai08, bai09, bai10, bai11
from dashboard import m6_dashboard

st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
    page_icon="🇻🇳",
    layout="wide"
)

# =========================
# GLOBAL STYLE (làm UI xịn hơn)
# =========================
st.markdown("""
<style>
    .main {
        background-color: #0f1117;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR (UI UPGRADE)
# =========================
st.sidebar.markdown("""
<div style="text-align:center;">
    <h2 style="color:#4CAF50;">🇻🇳 AIDEOM-VN</h2>
    <p style="color:gray; font-size:13px;">Optimization & Decision System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📌 Navigation",
    ["Trang chủ"] + [f"Bài {i}" for i in range(1, 12)] + ["Đồ án (Bài 12)"]
)

# =========================
# PAGE MAP (GIỮ NGUYÊN)
# =========================
pages = {
    "Trang chủ": None,
    "Bài 1": bai01, "Bài 2": bai02, "Bài 3": bai03,
    "Bài 4": bai04, "Bài 5": bai05, "Bài 6": bai06,
    "Bài 7": bai07, "Bài 8": bai08, "Bài 9": bai09,
    "Bài 10": bai10, "Bài 11": bai11,
    "Đồ án (Bài 12)": m6_dashboard
}

# =========================
# HOME PAGE (NÂNG CẤP MẠNH)
# =========================
if menu == "Trang chủ":

    st.markdown("""
    <div style="padding:20px 0px;">
        <h1 style="font-size:40px;">📊 AIDEOM-VN Dashboard</h1>
        <p style="color:gray; font-size:18px;">
            Hệ thống hỗ trợ ra quyết định & tối ưu hóa mô hình kinh tế
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="padding:20px; border-radius:12px; background:#1c1f26;">
            <h3>📘 11 Modules</h3>
            <p style="color:gray;">Linear & Stochastic Optimization</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="padding:20px; border-radius:12px; background:#1c1f26;">
            <h3>📈 Analytics</h3>
            <p style="color:gray;">Data-driven decision making</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="padding:20px; border-radius:12px; background:#1c1f26;">
            <h3>⚙️ Optimization</h3>
            <p style="color:gray;">Pyomo / OR Models</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👉 Chọn module bên trái để bắt đầu phân tích")

# =========================
# MODULE RUNNER (GIỮ NGUYÊN LOGIC)
# =========================
else:
    try:
        pages[menu].run()
    except Exception as e:
        st.error(f"Lỗi module: {e}")