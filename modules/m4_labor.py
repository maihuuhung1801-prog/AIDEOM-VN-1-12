"""
Module M4 — Mô phỏng thị trường lao động & việc làm ròng (NetJob)
===================================================================
Kỹ thuật: Markov Chain dịch chuyển lao động + mô hình thay thế/bổ sung AI.
Đầu vào : vietnam_sectors_2024.csv, kịch bản S1–S5.
Đầu ra  : NetJob từng ngành 2026–2030, ma trận chuyển trạng thái lao động.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).parent.parent / "data"

SECTORS = [
    "Nông-Lâm-Thủy sản", "Công nghiệp chế biến", "Xây dựng",
    "Khai khoáng",        "Bán buôn-Bán lẻ",     "Tài chính-Ngân hàng",
    "Logistics-Vận tải",  "ICT-Công nghệ số",     "Giáo dục-Đào tạo",
    "Y tế"
]
N_SECTORS = len(SECTORS)

# Tốc độ thay thế AI (% lao động/năm) theo kịch bản
AI_REPLACE_RATE: Dict[str, np.ndarray] = {
    "S1": np.array([0.003, 0.020, 0.010, 0.025, 0.015, 0.022, 0.012, 0.005, 0.008, 0.006]),
    "S2": np.array([0.005, 0.030, 0.015, 0.030, 0.025, 0.035, 0.020, 0.010, 0.012, 0.010]),
    "S3": np.array([0.010, 0.060, 0.025, 0.055, 0.045, 0.065, 0.040, 0.015, 0.020, 0.018]),
    "S4": np.array([0.004, 0.022, 0.012, 0.022, 0.018, 0.025, 0.014, 0.008, 0.010, 0.008]),
    "S5": np.array([0.006, 0.038, 0.016, 0.035, 0.028, 0.042, 0.024, 0.010, 0.014, 0.012]),
}

# Tốc độ tạo việc làm mới từ số hóa (% lao động/năm)
DIGITAL_JOB_CREATE: Dict[str, np.ndarray] = {
    "S1": np.array([0.002, 0.012, 0.006, 0.008, 0.010, 0.012, 0.008, 0.020, 0.005, 0.006]),
    "S2": np.array([0.008, 0.025, 0.015, 0.012, 0.030, 0.040, 0.025, 0.060, 0.020, 0.018]),
    "S3": np.array([0.006, 0.035, 0.012, 0.015, 0.025, 0.055, 0.030, 0.090, 0.025, 0.022]),
    "S4": np.array([0.015, 0.020, 0.012, 0.010, 0.022, 0.028, 0.018, 0.040, 0.035, 0.030]),
    "S5": np.array([0.009, 0.028, 0.013, 0.013, 0.026, 0.042, 0.023, 0.068, 0.028, 0.023]),
}


def build_markov_matrix(scenario: str, year_offset: int = 0) -> np.ndarray:
    """
    Xây dựng ma trận chuyển trạng thái Markov (N×N) cho thị trường lao động.
    P[i, j] = xác suất lao động từ ngành i sang ngành j.
    """
    P = np.eye(N_SECTORS)

    replace = AI_REPLACE_RATE[scenario]
    create  = DIGITAL_JOB_CREATE[scenario]

    # Lao động thay thế bởi AI rời khỏi ngành (→ đào tạo lại hoặc thất nghiệp)
    # Phần lớn chuyển sang ICT (ngành 7) và Giáo dục (ngành 8)
    for i in range(N_SECTORS):
        outflow = min(replace[i] * (1 + year_offset * 0.05), 0.15)
        P[i, i]  -= outflow
        P[i, 7]  += outflow * 0.50   # → ICT
        P[i, 8]  += outflow * 0.30   # → Giáo dục
        P[i, i]  += outflow * 0.20   # → tự nâng cấp trong ngành

    # Đảm bảo mỗi hàng tổng = 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums
    np.fill_diagonal(P, np.maximum(P.diagonal(), 0))

    return P


def simulate_labor(scenario: str = "S5", T: int = 5) -> pd.DataFrame:
    """
    Mô phỏng phân bổ lao động và NetJob 2026–2030.

    Returns
    -------
    DataFrame: year, sector, labor_million, net_job_created, net_job_lost,
               net_job_balance, employment_rate_change
    """
    df_s = pd.read_csv(DATA_DIR / "vietnam_sectors_2024.csv")
    L0   = df_s["labor_million"].values.astype(float)

    replace_base = AI_REPLACE_RATE[scenario]
    create_base  = DIGITAL_JOB_CREATE[scenario]

    records = []
    L = L0.copy()

    for t in range(T):
        year = 2026 + t

        # Tốc độ tăng dần theo thời gian (học hỏi & tích lũy)
        replace_t = replace_base * (1 + t * 0.08)
        create_t  = create_base  * (1 + t * 0.12)

        jobs_lost    = L * replace_t
        jobs_created = L * create_t
        net_balance  = jobs_created - jobs_lost

        L_new = np.maximum(L + net_balance, 0)

        for i, sector in enumerate(SECTORS):
            records.append({
                "year":       year,
                "scenario":   scenario,
                "sector":     sector,
                "labor_million":       round(float(L_new[i]), 3),
                "net_job_created":     round(float(jobs_created[i]) * 1e6, 0),
                "net_job_lost":        round(float(jobs_lost[i])    * 1e6, 0),
                "net_job_balance":     round(float(net_balance[i])  * 1e6, 0),
                "net_balance_pct":     round(float(net_balance[i] / (L[i]+1e-9)) * 100, 2),
            })

        L = L_new

    return pd.DataFrame(records)


def national_netjob_summary(scenario: str = "S5") -> pd.DataFrame:
    """Tổng hợp việc làm ròng toàn quốc theo năm."""
    df = simulate_labor(scenario)
    summary = df.groupby("year").agg(
        total_labor_million=("labor_million", "sum"),
        net_created=("net_job_created", "sum"),
        net_lost=("net_job_lost", "sum"),
        net_balance=("net_job_balance", "sum"),
    ).reset_index()
    summary["net_balance_pct"] = (
        summary["net_balance"] / (summary["total_labor_million"]*1e6) * 100
    ).round(2)
    return summary


def compare_scenarios_labor() -> pd.DataFrame:
    """So sánh NetJob 2030 theo 5 kịch bản."""
    rows = []
    labels = {"S1":"Truyền thống","S2":"Số hóa nhanh","S3":"AI dẫn dắt",
              "S4":"Bao trùm số","S5":"Tối ưu cân bằng"}
    for s in labels:
        df = national_netjob_summary(s)
        row_2030 = df[df["year"] == 2030].iloc[0]
        rows.append({
            "scenario": s, "label": labels[s],
            "net_created_2030": int(row_2030["net_created"]),
            "net_lost_2030":    int(row_2030["net_lost"]),
            "net_balance_2030": int(row_2030["net_balance"]),
            "balance_pct":      float(row_2030["net_balance_pct"]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_labor("S5")
    print("=== MÔ PHỎNG LAO ĐỘNG (S5, 2030) ===")
    print(df[df["year"]==2030][
        ["sector","labor_million","net_job_created","net_job_lost","net_job_balance"]
    ].to_string(index=False))

    print("\n=== SO SÁNH 5 KỊCH BẢN (VIỆC LÀM RÒNG 2030) ===")
    print(compare_scenarios_labor().to_string(index=False))
def run():
    st.title("👷 Bài 4: Mô phỏng thị trường lao động (NetJob)")
    st.markdown("Sử dụng **Markov Chain** để tính toán sự dịch chuyển lao động và mô hình hóa tác động thay thế/bổ sung của AI đến việc làm.")
    
    st.subheader("Dự báo cân bằng việc làm (NetJob) đến năm 2030")
    df_labor = compare_scenarios_labor()
    st.dataframe(df_labor)

import streamlit as st
import pandas as pd
import numpy as np

# Giữ nguyên các import cũ của bạn ở trên đầu file
# Đảm bảo from __future__ import annotations nằm ở dòng 1

def run():
    st.title("👷 Bài 4: Mô phỏng thị trường lao động (NetJob)")
    st.markdown("Phân tích dịch chuyển lao động và tác động của AI đến việc làm.")

    # 1. Chọn kịch bản
    scenario = st.selectbox("Chọn kịch bản dự báo:", ["S1", "S2", "S3", "S4", "S5"])

    try:
        # 2. Gọi hàm tính toán
        # Nếu hàm compare_scenarios_labor() của bạn trả về DataFrame, ta hiển thị trực tiếp
        df = compare_scenarios_labor()
        
        # 3. Hiển thị bảng
        st.subheader(f"Kết quả mô phỏng cho kịch bản {scenario}")
        st.dataframe(df, use_container_width=True)

        # 4. Vẽ biểu đồ NetJob nếu có dữ liệu phù hợp
        if "net_balance_2030" in df.columns:
            st.subheader("Biểu đồ cân bằng việc làm 2030")
            st.bar_chart(df.set_index("label")["net_balance_2030"])
        
    except Exception as e:
        st.error(f"Lỗi khi chạy mô phỏng Bài 4: {e}")
        st.write("Kiểm tra lại xem file dữ liệu (vietnam_sectors_2024.csv) đã có trong thư mục data/ chưa?")

if __name__ == "__main__":
    run()