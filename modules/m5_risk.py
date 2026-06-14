"""
Module M5 — Đánh giá rủi ro đa chiều (Cyber, Môi trường, Phụ thuộc)
======================================================================
Kỹ thuật: Multi-criteria Risk Scoring + Stochastic threshold analysis.
Đầu vào : Tham số rủi ro từ Bài 7 (e, rho, sigma) + kịch bản đầu tư.
Đầu ra  : Chỉ số rủi ro tổng hợp + cảnh báo theo ngưỡng.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

DATA_DIR = Path(__file__).parent.parent / "data"

REGIONS = [
    "Trung du MN phía Bắc", "Đồng bằng sông Hồng",
    "BTB + DH Trung Bộ",    "Tây Nguyên",
    "Đông Nam Bộ",          "ĐB sông Cửu Long"
]

# Tham số rủi ro môi trường & an ninh (từ Bảng 7.3)
ENV_EMISSION   = np.array([0.42, 0.32, 0.45, 0.30, 0.12, 0.25])  # CO2/tỷ
DATA_RISK_AI   = np.array([0.18, 0.55, 0.28, 0.48, 0.28, 0.32])  # rủi ro/AI
MITIGATE_H     = np.array([0.32, 0.45, 0.30, 0.35, 0.62, 0.52])  # giảm rủi ro/H

# Ngưỡng cảnh báo rủi ro
RISK_THRESHOLDS = {
    "cyber":   {"low": 0.2, "medium": 0.4, "high": 0.6},
    "env":     {"low": 0.3, "medium": 0.5, "high": 0.7},
    "depend":  {"low": 0.25,"medium": 0.45,"high": 0.65},
    "total":   {"low": 0.3, "medium": 0.5, "high": 0.65},
}

SCENARIO_INVEST_PROFILE: Dict[str, Dict] = {
    "S1": {"ai_pct": 0.10, "h_pct": 0.10, "infra_pct": 0.70, "foreign_dep": 0.65},
    "S2": {"ai_pct": 0.15, "h_pct": 0.15, "infra_pct": 0.45, "foreign_dep": 0.55},
    "S3": {"ai_pct": 0.45, "h_pct": 0.15, "infra_pct": 0.20, "foreign_dep": 0.75},
    "S4": {"ai_pct": 0.10, "h_pct": 0.40, "infra_pct": 0.30, "foreign_dep": 0.40},
    "S5": {"ai_pct": 0.22, "h_pct": 0.25, "infra_pct": 0.28, "foreign_dep": 0.50},
}


def compute_risk_scores(scenario: str,
                        budget: float = 120_000.0) -> pd.DataFrame:
    """
    Tính điểm rủi ro theo 3 chiều: Cyber, Môi trường, Phụ thuộc công nghệ.

    Returns
    -------
    DataFrame: region × risk dimensions + composite_risk + alert_level
    """
    profile = SCENARIO_INVEST_PROFILE[scenario]

    ai_invest    = budget / 6 * profile["ai_pct"]   # trung bình mỗi vùng
    h_invest     = budget / 6 * profile["h_pct"]
    infra_invest = budget / 6 * profile["infra_pct"]
    foreign_dep  = profile["foreign_dep"]

    records = []
    for r, region in enumerate(REGIONS):
        # Rủi ro an ninh mạng / dữ liệu
        cyber_risk = (DATA_RISK_AI[r] * ai_invest / 10_000
                      - MITIGATE_H[r] * h_invest / 10_000)
        cyber_risk = np.clip(cyber_risk, 0, 1)

        # Rủi ro môi trường (phát thải từ hạ tầng + AI)
        env_risk = ENV_EMISSION[r] * (infra_invest + ai_invest) / 50_000
        env_risk = np.clip(env_risk, 0, 1)

        # Rủi ro phụ thuộc công nghệ nước ngoài
        depend_risk = foreign_dep * (profile["ai_pct"] + profile["infra_pct"]) / 2
        depend_risk = np.clip(depend_risk, 0, 1)

        # Rủi ro tổng hợp có trọng số
        composite = 0.45 * cyber_risk + 0.35 * env_risk + 0.20 * depend_risk

        # Phân loại cảnh báo
        t = RISK_THRESHOLDS["total"]
        if composite < t["low"]:       alert = "🟢 An toàn"
        elif composite < t["medium"]:  alert = "🟡 Theo dõi"
        elif composite < t["high"]:    alert = "🟠 Cảnh báo"
        else:                          alert = "🔴 Nguy hiểm"

        records.append({
            "region":       region,
            "scenario":     scenario,
            "cyber_risk":   round(float(cyber_risk), 4),
            "env_risk":     round(float(env_risk), 4),
            "depend_risk":  round(float(depend_risk), 4),
            "composite_risk": round(float(composite), 4),
            "alert_level":  alert,
            "ai_invest":    round(ai_invest, 1),
            "h_invest":     round(h_invest, 1),
        })

    return pd.DataFrame(records)


def monte_carlo_risk(scenario: str, n_sim: int = 5000,
                     budget: float = 120_000.0) -> Dict:
    """
    Monte Carlo simulation để phân tích phân phối rủi ro tổng hợp.
    Tham số ngẫu nhiên: dao động ±20% quanh giá trị trung tâm.

    Returns
    -------
    Dict: mean, std, VaR_95, CVaR_95, prob_high_risk (per region)
    """
    np.random.seed(42)
    profile = SCENARIO_INVEST_PROFILE[scenario]
    budget_per_region = budget / 6

    results = {r: [] for r in REGIONS}

    for _ in range(n_sim):
        # Shock ngẫu nhiên ±20%
        ai_shock    = np.random.uniform(0.8, 1.2, 6)
        h_shock     = np.random.uniform(0.8, 1.2, 6)
        infra_shock = np.random.uniform(0.8, 1.2, 6)

        for r, region in enumerate(REGIONS):
            ai_inv    = budget_per_region * profile["ai_pct"]    * ai_shock[r]
            h_inv     = budget_per_region * profile["h_pct"]     * h_shock[r]
            infra_inv = budget_per_region * profile["infra_pct"] * infra_shock[r]

            cyber = np.clip(DATA_RISK_AI[r]*ai_inv/10_000 - MITIGATE_H[r]*h_inv/10_000, 0, 1)
            env   = np.clip(ENV_EMISSION[r]*(infra_inv+ai_inv)/50_000, 0, 1)
            dep   = np.clip(profile["foreign_dep"]*(profile["ai_pct"]+profile["infra_pct"])/2
                            * np.random.uniform(0.9, 1.1), 0, 1)
            comp  = 0.45*cyber + 0.35*env + 0.20*dep
            results[region].append(comp)

    summary = {}
    for region in REGIONS:
        arr = np.array(results[region])
        var95  = float(np.percentile(arr, 95))
        cvar95 = float(arr[arr >= var95].mean())
        summary[region] = {
            "mean":          round(float(arr.mean()), 4),
            "std":           round(float(arr.std()),  4),
            "VaR_95":        round(var95,  4),
            "CVaR_95":       round(cvar95, 4),
            "prob_high_risk": round(float((arr > RISK_THRESHOLDS["total"]["high"]).mean()), 4),
        }
    return summary


def risk_radar_data(scenarios: list = None) -> pd.DataFrame:
    """Tổng hợp điểm rủi ro trung bình quốc gia theo 5 kịch bản."""
    if scenarios is None:
        scenarios = ["S1", "S2", "S3", "S4", "S5"]
    labels = {"S1":"Truyền thống","S2":"Số hóa nhanh","S3":"AI dẫn dắt",
              "S4":"Bao trùm số","S5":"Tối ưu cân bằng"}
    rows = []
    for s in scenarios:
        df = compute_risk_scores(s)
        rows.append({
            "scenario":    s,
            "label":       labels[s],
            "cyber_risk":  df["cyber_risk"].mean().round(4),
            "env_risk":    df["env_risk"].mean().round(4),
            "depend_risk": df["depend_risk"].mean().round(4),
            "composite":   df["composite_risk"].mean().round(4),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== ĐÁNH GIÁ RỦI RO (S5) ===")
    df = compute_risk_scores("S5")
    print(df[["region","cyber_risk","env_risk","depend_risk","composite_risk","alert_level"]].to_string(index=False))

    print("\n=== SO SÁNH RỦI RO 5 KỊCH BẢN ===")
    print(risk_radar_data().to_string(index=False))

    print("\n=== MONTE CARLO S3 (AI dẫn dắt - rủi ro cao nhất) ===")
    mc = monte_carlo_risk("S3", n_sim=1000)
    for r, v in mc.items():
        print(f"  {r:<28}: mean={v['mean']:.3f}, VaR95={v['VaR_95']:.3f}, P(high)={v['prob_high_risk']:.2%}")
def run():
    st.title("⚠️ Bài 5: Đánh giá Rủi ro Đa chiều")
    st.markdown("Phân tích rủi ro an ninh mạng, rủi ro môi trường và rủi ro phụ thuộc công nghệ để đưa ra cảnh báo.")
    
    st.subheader("Chỉ số rủi ro tổng hợp theo kịch bản")
    df_risk = risk_radar_data()
    st.dataframe(df_risk)

if __name__ == "__main__":
    run()