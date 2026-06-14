"""
Module M3 — Tối ưu phân bổ ngân sách (LP tĩnh + Dynamic)
==========================================================
Kỹ thuật: scipy.optimize (SLSQP) — kết hợp Bài 4 (LP) và Bài 8 (dynamic).
Đầu vào : kịch bản S1–S5, beta matrix từ dữ liệu thực.
Đầu ra  : ma trận phân bổ ngân sách tối ưu (6 vùng × 4 hạng mục) và quỹ đạo động.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, LinearConstraint
from typing import Dict, Tuple

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Tham số ────────────────────────────────────────────────────
REGIONS = [
    "Trung du MN phía Bắc", "Đồng bằng sông Hồng",
    "BTB + DH Trung Bộ",    "Tây Nguyên",
    "Đông Nam Bộ",          "ĐB sông Cửu Long"
]
INVEST_TYPES = ["Hạ tầng (I)", "Dữ liệu (D)", "AI", "Nhân lực (H)"]

# Beta matrix (6 vùng × 4 hạng mục) — tác động cận biên GDP
BETA = np.array([
    [0.85, 0.72, 0.65, 0.60],
    [1.35, 1.42, 1.55, 1.25],
    [0.95, 0.88, 0.90, 0.82],
    [0.75, 0.65, 0.60, 0.58],
    [1.40, 1.48, 1.60, 1.30],
    [0.92, 0.85, 0.88, 0.78],
])

# Phân bổ cơ cấu theo kịch bản
SCENARIO_ALLOC: Dict[str, np.ndarray] = {
    "S1": np.array([0.70, 0.10, 0.10, 0.10]),
    "S2": np.array([0.25, 0.45, 0.15, 0.15]),
    "S3": np.array([0.20, 0.20, 0.45, 0.15]),
    "S4": np.array([0.30, 0.20, 0.10, 0.40]),
    "S5": np.array([0.28, 0.25, 0.22, 0.25]),
}

BUDGET_TOTAL  = 120_000.0   # tỷ VND
MIN_PER_REGION = 5_000.0
MAX_PER_REGION = 30_000.0
MIN_INFRA      = 500.0


def optimize_static(scenario: str = "S5",
                    budget: float = BUDGET_TOTAL) -> Tuple[np.ndarray, float, Dict]:
    """
    Tối ưu phân bổ tĩnh (1 năm): tối đa hóa GDP contribution.
    Biến: x[6×4] = ngân sách cho từng (vùng, hạng mục).

    Returns
    -------
    X_opt   : (6, 4) ma trận phân bổ tối ưu
    gdp_val : giá trị GDP đóng góp tối đa
    info    : dict thông tin bổ sung
    """
    alloc = SCENARIO_ALLOC[scenario]
    n_var = 24  # 6×4

    # Hàm mục tiêu: âm để minimize
    def objective(x):
        X = x.reshape(6, 4)
        return -(BETA * X).sum()

    # Ràng buộc
    constraints = []

    # C1: Tổng ngân sách ≤ budget
    def c_total(x): return budget - x.sum()
    constraints.append({"type": "ineq", "fun": c_total})

    # C2: Ngân sách mỗi vùng ≤ MAX_PER_REGION
    for r in range(6):
        def c_max_r(x, r=r): return MAX_PER_REGION - x[r*4:(r+1)*4].sum()
        constraints.append({"type": "ineq", "fun": c_max_r})

    # C3: Ngân sách mỗi vùng ≥ MIN_PER_REGION (bao trùm)
    for r in range(6):
        def c_min_r(x, r=r): return x[r*4:(r+1)*4].sum() - MIN_PER_REGION
        constraints.append({"type": "ineq", "fun": c_min_r})

    # C4: Hạ tầng mỗi vùng ≥ MIN_INFRA
    for r in range(6):
        def c_infra(x, r=r): return x[r*4] - MIN_INFRA
        constraints.append({"type": "ineq", "fun": c_infra})

    # C5: Cơ cấu theo kịch bản (±15% linh hoạt)
    for j in range(4):
        target = alloc[j] * budget
        def c_struct_lo(x, j=j, t=target): return x[j::4].sum() - t * 0.85
        def c_struct_hi(x, j=j, t=target): return t * 1.15 - x[j::4].sum()
        constraints.append({"type": "ineq", "fun": c_struct_lo})
        constraints.append({"type": "ineq", "fun": c_struct_hi})

    # Khởi tạo: phân bổ đều
    x0 = np.ones(n_var) * budget / n_var
    bounds = [(0, MAX_PER_REGION)] * n_var

    res = minimize(objective, x0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-10})

    X_opt = res.x.reshape(6, 4)
    gdp_val = -res.fun

    df_result = pd.DataFrame(X_opt, index=REGIONS, columns=INVEST_TYPES)
    df_result["Tổng vùng"] = df_result.sum(axis=1)
    df_result.loc["Tổng ngành"] = df_result.sum()

    info = {
        "scenario": scenario,
        "converged": res.success,
        "gdp_contribution": round(gdp_val, 1),
        "total_budget": round(res.x.sum(), 1),
        "alloc_table": df_result.round(1),
    }
    return X_opt, gdp_val, info


def optimize_dynamic(scenario: str = "S5",
                     T: int = 5,
                     annual_budget: float = 24_000.0) -> pd.DataFrame:
    """
    Tối ưu phân bổ động 5 năm (2026–2030): tổng GDP đóng góp chiết khấu.
    Mỗi năm có budget riêng tăng dần 8%.

    Returns
    -------
    DataFrame: year × (6 vùng × 4 hạng mục) + GDP_contrib
    """
    rho = 0.97
    records = []

    for t in range(T):
        year = 2026 + t
        budget_t = annual_budget * (1.08 ** t)
        X_opt, gdp_val, info = optimize_static(scenario, budget_t)

        row = {"year": year, "scenario": scenario, "budget": round(budget_t, 1)}
        for r_idx, r_name in enumerate(REGIONS):
            for j_idx, j_name in enumerate(INVEST_TYPES):
                row[f"{r_name}_{j_name}"] = round(X_opt[r_idx, j_idx], 1)
        row["gdp_contribution"] = round(gdp_val * rho**t, 1)
        records.append(row)

    return pd.DataFrame(records)


def compare_scenarios_static(budget: float = BUDGET_TOTAL) -> pd.DataFrame:
    """So sánh GDP contribution tất cả 5 kịch bản."""
    rows = []
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        _, gdp_val, info = optimize_static(s, budget)
        rows.append({
            "scenario": s,
            "label": {"S1":"Truyền thống","S2":"Số hóa nhanh",
                      "S3":"AI dẫn dắt","S4":"Bao trùm số","S5":"Tối ưu cân bằng"}[s],
            "gdp_contribution": info["gdp_contribution"],
            "converged": info["converged"]
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== PHÂN BỔ NGÂN SÁCH TỐI ƯU (S5 - Cân bằng) ===")
    _, gdp, info = optimize_static("S5")
    print(info["alloc_table"].to_string())
    print(f"\nGDP contribution: {info['gdp_contribution']:,.1f} tỷ VND")

    print("\n=== SO SÁNH 5 KỊCH BẢN ===")
    print(compare_scenarios_static().to_string(index=False))
def run():
    st.title("💰 Bài 3: Tối ưu phân bổ ngân sách")
    st.markdown("Sử dụng giải thuật **SLSQP** để tìm phương án phân bổ ngân sách tối đa hóa GDP cho 6 vùng và 4 hạng mục đầu tư.")
    
    st.subheader("So sánh hiệu quả phân bổ tĩnh giữa các kịch bản")
    df_static = compare_scenarios_static()
    st.dataframe(df_static)

if __name__ == "__main__":
    run()