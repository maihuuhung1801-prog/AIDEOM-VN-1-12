"""
Module M1 — Dự báo kinh tế vĩ mô 2026–2030
=============================================
Kỹ thuật: Hàm sản xuất Cobb-Douglas + OLS calibration + dự báo tuyến tính TFP.
Đầu vào : vietnam_macro_2020_2025.csv
Đầu ra  : DataFrame GDP, TFP, lao động 2026–2030 cho 5 kịch bản S1–S5.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# ── Tham số hàm sản xuất (Bài 1 / Bài 8) ──────────────────────
ALPHA_K  = 0.33
ALPHA_L  = 0.42
ALPHA_D  = 0.10
ALPHA_AI = 0.08
ALPHA_H  = 0.07

DATA_DIR = Path(__file__).parent.parent / "data"

# Tham số tăng trưởng lao động & TFP theo kịch bản
SCENARIO_PARAMS: Dict[str, Dict] = {
    "S1": {"tfp_growth": 0.010, "label": "Truyền thống",     "color": "#607D8B"},
    "S2": {"tfp_growth": 0.018, "label": "Số hóa nhanh",     "color": "#1976D2"},
    "S3": {"tfp_growth": 0.025, "label": "AI dẫn dắt",       "color": "#E53935"},
    "S4": {"tfp_growth": 0.013, "label": "Bao trùm số",      "color": "#43A047"},
    "S5": {"tfp_growth": 0.020, "label": "Tối ưu cân bằng",  "color": "#7B1FA2"},
}

ALLOCATION: Dict[str, Dict] = {
    "S1": {"K": 0.70, "D": 0.10, "AI": 0.10, "H": 0.10},
    "S2": {"K": 0.25, "D": 0.45, "AI": 0.15, "H": 0.15},
    "S3": {"K": 0.20, "D": 0.20, "AI": 0.45, "H": 0.15},
    "S4": {"K": 0.30, "D": 0.20, "AI": 0.10, "H": 0.40},
    "S5": {"K": 0.28, "D": 0.25, "AI": 0.22, "H": 0.25},
}


def load_macro() -> pd.DataFrame:
    """Đọc dữ liệu vĩ mô 2020–2025."""
    return pd.read_csv(DATA_DIR / "vietnam_macro_2020_2025.csv")


def calibrate_tfp(df: pd.DataFrame) -> float:
    """
    Hiệu chỉnh TFP A0 từ GDP và các yếu tố sản xuất năm 2025.
    Sử dụng giá trị proxy: D ~ digital_economy_share, AI ~ digital_index trung bình,
    H ~ labor với trọng số năng suất.
    """
    row = df[df["year"] == 2025].iloc[0]
    Y  = float(row["GDP_trillion_VND"])
    K  = Y * 2.8           # Tỷ lệ K/Y ≈ 2.8 (ước lượng từ số liệu WB)
    L  = float(row["population_million"]) * 0.505  # tỷ lệ tham gia LĐ
    D  = float(row["digital_economy_share_GDP_pct"])
    AI = 45.0              # AI readiness trung bình ngành (sectors_2024)
    H  = 38.0              # % lao động qua đào tạo

    A = Y / (K**ALPHA_K * L**ALPHA_L * D**ALPHA_D * AI**ALPHA_AI * H**ALPHA_H)
    return A


def forecast(scenario: str = "S5", horizon: int = 5) -> pd.DataFrame:
    """
    Dự báo GDP, TFP, lao động 2026–(2026+horizon-1).

    Parameters
    ----------
    scenario : str  — một trong S1..S5
    horizon  : int  — số năm dự báo (mặc định 5 → 2026–2030)

    Returns
    -------
    DataFrame với cột: year, Y, K, L, D, AI, H, A, growth_pct
    """
    df_macro = load_macro()
    A0 = calibrate_tfp(df_macro)

    row25 = df_macro[df_macro["year"] == 2025].iloc[0]
    Y0 = float(row25["GDP_trillion_VND"])
    K0 = Y0 * 2.8
    L0 = float(row25["population_million"]) * 0.505
    D0 = float(row25["digital_economy_share_GDP_pct"])
    AI0 = 45.0
    H0 = 38.0

    params = SCENARIO_PARAMS[scenario]
    alloc  = ALLOCATION[scenario]
    tfp_g  = params["tfp_growth"]

    inv_rate  = 0.30   # tổng đầu tư / GDP
    delta_K   = 0.05
    L_growth  = 0.008  # tăng lao động ~0.8%/năm
    D_growth  = {"S1": 0.05, "S2": 0.15, "S3": 0.12, "S4": 0.08, "S5": 0.12}[scenario]
    AI_growth = {"S1": 0.04, "S2": 0.10, "S3": 0.18, "S4": 0.06, "S5": 0.12}[scenario]
    H_growth  = {"S1": 0.02, "S2": 0.05, "S3": 0.04, "S4": 0.10, "S5": 0.06}[scenario]

    records = []
    K, L, D, AI, H, A, Y = K0, L0, D0, AI0, H0, A0, Y0

    for t in range(horizon):
        year = 2026 + t
        Y_prev = Y

        # Cập nhật trạng thái
        I_total = Y * inv_rate
        K  = (1 - delta_K) * K + I_total * alloc["K"]
        D  = D  * (1 + D_growth)
        AI = AI * (1 + AI_growth)
        H  = H  * (1 + H_growth)
        L  = L  * (1 + L_growth)
        A  = A  * (1 + tfp_g)

        Y = A * K**ALPHA_K * L**ALPHA_L * D**ALPHA_D * AI**ALPHA_AI * H**ALPHA_H
        g = (Y / Y_prev - 1) * 100

        records.append({
            "year": year, "scenario": scenario, "label": params["label"],
            "Y": round(Y, 2), "K": round(K, 2), "L": round(L, 3),
            "D": round(D, 2), "AI": round(AI, 2), "H": round(H, 2),
            "A": round(A, 6), "growth_pct": round(g, 2)
        })

    return pd.DataFrame(records)


def forecast_all_scenarios(horizon: int = 5) -> pd.DataFrame:
    """Dự báo tất cả 5 kịch bản, gộp kết quả."""
    frames = [forecast(s, horizon) for s in SCENARIO_PARAMS]
    return pd.concat(frames, ignore_index=True)

def run():
    st.title("📊 Bài 1: Dự báo kinh tế vĩ mô 2026–2030")
    st.markdown("""
    Sử dụng mô hình hàm sản xuất **Cobb-Douglas** để dự báo GDP dựa trên các yếu tố: 
    Vốn (K), Lao động (L), Công nghệ (D), AI và Nhân lực (H).
    """)

    # 1. Chạy dự báo
    df = forecast_all_scenarios()

    # 2. Sidebar để lọc kịch bản
    st.sidebar.subheader("Cấu hình hiển thị")
    selected_scenarios = st.sidebar.multiselect(
        "Chọn kịch bản:", 
        options=df["label"].unique(), 
        default=df["label"].unique()
    )

    # 3. Lọc dữ liệu
    df_filtered = df[df["label"].isin(selected_scenarios)]

    # 4. Hiển thị bảng dữ liệu
    st.subheader("Bảng kết quả dự báo")
    st.dataframe(df_filtered.style.format({"Y": "{:.2f}", "growth_pct": "{:.2f}%"}))

    # 5. Hiển thị biểu đồ GDP
    st.subheader("Biểu đồ tăng trưởng GDP dự báo")
    chart_data = df_filtered.pivot(index="year", columns="label", values="Y")
    st.line_chart(chart_data)