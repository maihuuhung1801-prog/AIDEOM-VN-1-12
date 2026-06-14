"""
Module M2 — Đánh giá sẵn sàng số (Digital Readiness)
======================================================
Kỹ thuật: TOPSIS + Entropy Weight Method
Đầu vào : vietnam_regions_2024.csv, vietnam_sectors_2024.csv
Đầu ra  : DataFrame xếp hạng Digital Index & AI Readiness theo vùng và ngành.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

DATA_DIR = Path(__file__).parent.parent / "data"

# Tiêu chí đánh giá vùng (benefit = True: cao tốt hơn)
REGION_CRITERIA = {
    "digital_index_0_100":      True,
    "ai_readiness_0_100":       True,
    "internet_penetration_pct": True,
    "trained_labor_pct":        True,
    "rd_intensity_pct":         True,
    "gini_coef":                False,   # thấp hơn = tốt hơn
}

SECTOR_CRITERIA = {
    "digital_index_0_100":  True,
    "ai_readiness_0_100":   True,
    "spillover_coef_0_1":   True,
    "rd_intensity_pct":     True,
    "automation_risk_pct":  False,
}


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Tính trọng số Entropy từ ma trận quyết định đã chuẩn hóa.
    Tiêu chí có phân tán lớn → trọng số cao hơn.
    """
    # Chuẩn hóa về [0,1]
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    norm = (matrix - col_min) / (col_max - col_min + 1e-9)
    norm = norm + 1e-9   # tránh log(0)

    # Tính xác suất
    p = norm / norm.sum(axis=0)

    # Entropy
    n = matrix.shape[0]
    e = -np.sum(p * np.log(p), axis=0) / np.log(n)

    # Trọng số
    d = 1 - e
    w = d / d.sum()
    return w


def topsis(matrix: np.ndarray, weights: np.ndarray,
           benefit_mask: np.ndarray) -> np.ndarray:
    """
    TOPSIS — Technique for Order of Preference by Similarity to Ideal Solution.

    Parameters
    ----------
    matrix       : (n, m) — ma trận quyết định gốc
    weights      : (m,)   — trọng số entropy
    benefit_mask : (m,)   — True = tiêu chí lợi ích

    Returns
    -------
    scores : (n,) — điểm TOPSIS [0, 1], cao hơn = tốt hơn
    """
    # Chuẩn hóa vector
    norms = np.sqrt((matrix**2).sum(axis=0))
    R = matrix / (norms + 1e-9)

    # Ma trận có trọng số
    V = R * weights

    # Nghiệm lý tưởng dương / âm
    ideal_pos = np.where(benefit_mask, V.max(axis=0), V.min(axis=0))
    ideal_neg = np.where(benefit_mask, V.min(axis=0), V.max(axis=0))

    d_pos = np.sqrt(((V - ideal_pos)**2).sum(axis=1))
    d_neg = np.sqrt(((V - ideal_neg)**2).sum(axis=1))

    return d_neg / (d_pos + d_neg + 1e-9)


def assess_regions() -> pd.DataFrame:
    """
    Đánh giá và xếp hạng 6 vùng kinh tế theo mức độ sẵn sàng số.

    Returns
    -------
    DataFrame gồm: region_name_en, các tiêu chí, entropy_weight (ví dụ),
    topsis_score, rank, digital_readiness_category
    """
    df = pd.read_csv(DATA_DIR / "vietnam_regions_2024.csv")
    criteria = list(REGION_CRITERIA.keys())
    benefit  = np.array(list(REGION_CRITERIA.values()))

    matrix  = df[criteria].values.astype(float)
    weights = entropy_weights(matrix)
    scores  = topsis(matrix, weights, benefit)

    df["topsis_score"] = scores
    df["rank"] = df["topsis_score"].rank(ascending=False).astype(int)
    df["digital_readiness_category"] = pd.cut(
        df["topsis_score"],
        bins=[0, 0.35, 0.55, 0.75, 1.01],
        labels=["Thấp", "Trung bình", "Cao", "Rất cao"]
    )

    # Trọng số entropy cho từng tiêu chí
    weight_dict = dict(zip(criteria, weights.round(4)))

    return df.sort_values("rank"), weight_dict


def assess_sectors() -> pd.DataFrame:
    """
    Đánh giá và xếp hạng 10 ngành kinh tế.
    """
    df = pd.read_csv(DATA_DIR / "vietnam_sectors_2024.csv")
    criteria = list(SECTOR_CRITERIA.keys())
    benefit  = np.array(list(SECTOR_CRITERIA.values()))

    matrix  = df[criteria].values.astype(float)
    weights = entropy_weights(matrix)
    scores  = topsis(matrix, weights, benefit)

    df["topsis_score"] = scores
    df["rank"] = df["topsis_score"].rank(ascending=False).astype(int)
    df["ai_priority"] = pd.cut(
        df["topsis_score"],
        bins=[0, 0.3, 0.5, 0.7, 1.01],
        labels=["Ưu tiên thấp", "Trung bình", "Ưu tiên cao", "Đầu tàu"]
    )

    return df.sort_values("rank")


def get_digital_gap_index() -> pd.DataFrame:
    """
    Tính Digital Gap Index = khoảng cách giữa vùng và mức tốt nhất quốc gia.
    """
    df_r, _ = assess_regions()
    best = df_r["topsis_score"].max()
    df_r["digital_gap"] = ((best - df_r["topsis_score"]) / best * 100).round(1)
    df_r["gap_category"] = pd.cut(
        df_r["digital_gap"],
        bins=[-1, 10, 30, 60, 101],
        labels=["Tốt", "Trung bình", "Cần cải thiện", "Ưu tiên đặc biệt"]
    )
    return df_r[["region_name_en", "topsis_score", "digital_gap", "gap_category", "rank"]]


if __name__ == "__main__":
    df_r, w = assess_regions()
    print("=== XẾP HẠNG VÙNG ===")
    print(df_r[["region_name_en", "topsis_score", "rank", "digital_readiness_category"]].to_string(index=False))
    print("\nTrọng số entropy:", w)

    df_s = assess_sectors()
    print("\n=== XẾP HẠNG NGÀNH ===")
    print(df_s[["sector_name_en", "topsis_score", "rank", "ai_priority"]].to_string(index=False))
def run():
    st.title("📈 Bài 2: Đánh giá sẵn sàng số (Digital Readiness)")
    st.subheader("Kết quả xếp hạng các vùng")
    
    result = assess_regions()
    
    if isinstance(result, tuple):
        df_regions = result[0]
    elif isinstance(result, dict):
        df_regions = pd.DataFrame(result)
    else:
        df_regions = result
        
    st.dataframe(df_regions)