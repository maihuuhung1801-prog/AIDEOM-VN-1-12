"""
modules/bai09.py
Bài 9 — Tác động AI tới Thị trường Lao động Việt Nam
Chuyển đổi từ Jupyter Notebook sang Streamlit module.
Giữ nguyên toàn bộ logic, công thức, dữ liệu gốc.
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# 1. DỮ LIỆU
# ──────────────────────────────────────────────────────────────────
def load_data():
    sectors = [
        "Nông-Lâm-Thủy sản",
        "CN chế biến chế tạo",
        "Xây dựng",
        "Bán buôn-bán lẻ",
        "Tài chính-Ngân hàng",
        "Logistics-Vận tải",
        "CNTT-Truyền thông",
        "Giáo dục-Đào tạo",
    ]
    L    = np.array([13.20, 5.2, 28.0, 18.5, 32.0, 72.5, 30.0, 32.5])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100.0
    a1   = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    a2   = np.array([12.0, 18.5, 4.80, 15.2, 0.55, 16.8, 0.62, 24.0])
    b1   = np.array([45.0, 50.0, 35.0, 42.0, 22.0, 26.0, 20.0, 55.0])
    c1   = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1   = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])
    return sectors, L, risk, a1, a2, b1, c1, d1


# ──────────────────────────────────────────────────────────────────
# 2. MÔ HÌNH
# ──────────────────────────────────────────────────────────────────
def run_model(budget, sectors, L, risk, a1, b1, c1, d1):
    """Câu 9.4.1 — Giải LP tối ưu phân bổ x_AI và x_H bằng CVXPY."""
    try:
        import cvxpy as cp
    except ImportError:
        return None, None, None, "cvxpy chưa được cài. Chạy: pip install cvxpy"

    N = len(sectors)
    x_AI = cp.Variable(N, nonneg=True)
    x_H  = cp.Variable(N, nonneg=True)

    NewJob       = a1 * x_AI
    UpgradeJob   = b1 * x_H
    DisplacedJob = c1 * risk * x_AI
    RetrainCap   = d1 * x_H
    NetJob       = NewJob + UpgradeJob - DisplacedJob

    constraints = [
        cp.sum(x_AI + x_H) <= budget,
        NetJob >= 0,
        DisplacedJob <= RetrainCap,
    ]
    prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None, prob.value, f"Solver trạng thái: {prob.status}"

    return x_AI.value.copy(), x_H.value.copy(), prob.value, None


def run_model_safe(budget, sectors, L, risk, a1, b1, c1, d1,
                   x_AI_opt, x_H_opt, Z_opt):
    """Câu 9.4.4 — Thêm ràng buộc an sinh (DisplacedJob ≤ 5% L)."""
    try:
        import cvxpy as cp
    except ImportError:
        return None, None, None, "cvxpy chưa được cài."

    N = len(sectors)
    x_AI_s = cp.Variable(N, nonneg=True)
    x_H_s  = cp.Variable(N, nonneg=True)

    Displaced_s = c1 * risk * x_AI_s
    Retrain_s   = d1 * x_H_s
    NetJob_s    = a1 * x_AI_s + b1 * x_H_s - Displaced_s

    constraints_safe = [
        cp.sum(x_AI_s + x_H_s) <= budget,
        NetJob_s >= 0,
        Displaced_s <= Retrain_s,
        Displaced_s <= 0.05 * L,
    ]
    prob_safe = cp.Problem(cp.Maximize(cp.sum(NetJob_s)), constraints_safe)
    prob_safe.solve(solver=cp.HIGHS, verbose=False)

    if prob_safe.status not in ("optimal", "optimal_inaccurate"):
        return None, None, prob_safe.value, f"Vô nghiệm ({prob_safe.status})"

    return x_AI_s.value.copy(), x_H_s.value.copy(), prob_safe.value, None


# ──────────────────────────────────────────────────────────────────
# 3. BIỂU ĐỒ
# ──────────────────────────────────────────────────────────────────
def chart_optimal_allocation(sectors, L, risk, a1, b1, c1, d1,
                              x_AI_opt, x_H_opt):
    """4 biểu đồ phân bổ tối ưu — Câu 9.4.1."""
    N = len(sectors)
    x_pos = np.arange(N)
    NewJob_v    = a1 * x_AI_opt
    UpgradeJob_v = b1 * x_H_opt
    DisplacedJob_v = c1 * risk * x_AI_opt
    NetJob_v    = NewJob_v + UpgradeJob_v - DisplacedJob_v

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Bài 9 — Phân bổ Tối ưu", fontsize=13, fontweight="bold")

    w = 0.35
    ax = axes[0, 0]
    ax.bar(x_pos - w/2, x_AI_opt, w, label="x_AI (AI)", color="#2196F3")
    ax.bar(x_pos + w/2, x_H_opt,  w, label="x_H (Đào tạo)", color="#4CAF50")
    ax.set_ylabel("Ngân sách (tỷ VND)")
    ax.set_title("Phân bổ tối ưu: AI vs Đào tạo", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N{i+1}" for i in range(N)], fontsize=8)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(x_pos, NewJob_v, label="Việc AI mới", color="#4CAF50")
    ax.bar(x_pos, UpgradeJob_v, bottom=NewJob_v, label="Nâng cấp", color="#2196F3")
    ax.bar(x_pos, -DisplacedJob_v, bottom=NewJob_v + UpgradeJob_v,
           label="Mất việc", color="#FF5722")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("NetJob (ngàn người)")
    ax.set_title("Thành phần NetJob", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N{i+1}" for i in range(N)], fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    colors = ["#4CAF50" if nj >= 0 else "#FF5722" for nj in NetJob_v]
    ax.bar(x_pos, NetJob_v, color=colors)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_ylabel("NetJob ròng (ngàn người)")
    ax.set_title("NetJob ròng mỗi ngành (phải ≥ 0)", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N{i+1}" for i in range(N)], fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for i, nj in enumerate(NetJob_v):
        ax.text(i, nj + 0.5, f"{nj:.1f}", ha="center", fontsize=8)

    ax = axes[1, 1]
    loss_ratio = (DisplacedJob_v / L) * 100
    c_ratio = ["#4CAF50" if lr <= 5 else ("#FF9800" if lr <= 10 else "#F44336")
               for lr in loss_ratio]
    ax.bar(x_pos, loss_ratio, color=c_ratio)
    ax.axhline(5, color="red", linestyle="--", lw=1.5, label="Ngưỡng 5%")
    ax.set_ylabel("Tỷ lệ mất việc (%)")
    ax.set_title("% Lao động bị dịch chuyển so với tổng", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N{i+1}" for i in range(N)], fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def chart_threshold_sector2(a1, b1, c1, risk, x_AI_opt, x_H_opt):
    """Câu 9.4.2 — Ngưỡng x_H tối thiểu cho Ngành 2."""
    i = 1
    a1_2, b1_2, c1_2, r_2 = a1[i], b1[i], c1[i], risk[i]
    x_AI_range = np.linspace(0, 10000, 50)

    coef = (c1_2 * r_2 - a1_2) / b1_2
    x_H_min_vals    = [max(0, coef * x_ai) for x_ai in x_AI_range]
    NetJob_scan      = [a1_2*x_ai + b1_2*max(0, coef*x_ai) - c1_2*r_2*x_ai
                        for x_ai in x_AI_range]

    x_ai_opt_2  = x_AI_opt[i]
    x_h_min_2   = max(0, coef * x_ai_opt_2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(x_AI_range, x_H_min_vals, "o-", color="#E91E63", lw=2.5, ms=3,
            label="x_H tối thiểu để NetJob≥0")
    ax.scatter([x_ai_opt_2], [x_h_min_2], color="#4CAF50", s=100, zorder=5,
               label=f"Tối ưu: ({x_ai_opt_2:.1f}, {x_h_min_2:.1f})")
    ax.fill_between(x_AI_range, 0, x_H_min_vals, alpha=0.1, color="#E91E63")
    ax.set_xlabel("Đầu tư AI x_AI (tỷ VND)")
    ax.set_ylabel("Đầu tư đào tạo x_H (tỷ VND)")
    ax.set_title("Ngành 2 (CN chế biến):\nNgưỡng x_H tối thiểu để NetJob≥0",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x_AI_range, NetJob_scan, "o-", color="#2196F3", lw=2.5, ms=3)
    ax.axhline(0, color="red", linestyle="--", lw=1.5, label="NetJob = 0")
    ax.fill_between(x_AI_range, 0, NetJob_scan, alpha=0.1, color="#4CAF50")
    ax.set_xlabel("Đầu tư AI x_AI (tỷ VND)")
    ax.set_ylabel("NetJob₂ (ngàn người)")
    ax.set_title("NetJob ròng Ngành 2\nkhi tuân thủ NetJob≥0", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, x_ai_opt_2, x_h_min_2


def chart_safety_comparison(sectors, L, risk, a1, b1, c1,
                             x_AI_opt, x_H_opt, Z_opt,
                             x_AI_safe, x_H_safe, Z_safe):
    """Câu 9.4.4 — So sánh với/không ràng buộc an sinh."""
    N = len(sectors)
    x_pos = np.arange(N)
    displaced_no  = c1 * risk * x_AI_opt
    displaced_yes = c1 * risk * x_AI_safe
    limit_5pct    = 0.05 * L

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    w = 0.25
    ax.bar(x_pos - w, displaced_no,  w, label="Không ràng buộc", color="#F44336")
    ax.bar(x_pos,     displaced_yes, w, label="Có ràng buộc 5%", color="#4CAF50")
    ax.bar(x_pos + w, limit_5pct,    w, label="Ngưỡng 5% L_i",  color="#FF9800", alpha=0.5)
    ax.set_ylabel("Lao động dịch chuyển (ngàn)")
    ax.set_title("Lao động bị thay thế: 2 kịch bản", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N{i+1}" for i in range(N)], fontsize=8)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    z_vals   = [Z_opt, Z_safe]
    labels_z = ["Không an sinh\n(Z*)", "Có an sinh\n(Z*)"]
    bars = ax.bar(labels_z, z_vals, color=["#2196F3", "#4CAF50"], width=0.5)
    ax.set_ylabel("Tổng NetJob (ngàn người)")
    ax.set_title(f"Tổng NetJob — Chi phí an sinh = {Z_opt-Z_safe:,.0f}K việc",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, z_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 10,
                f"{val:,.0f}", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────
# 4. HÀM RUN() CHÍNH
# ──────────────────────────────────────────────────────────────────
def run():
    st.title("📘 Bài 9 — Tác động AI tới Thị trường Lao động Việt Nam")
    st.markdown(
        """
        **Học phần:** Mô hình Ra Quyết Định | **Cấp độ:** Khá khó

        #### 🎯 Mục tiêu
        1. Xây dựng mô hình mô phỏng tác động AI + tự động hóa → lao động
        2. Tính **NetJob ròng** = Việc mới + Nâng cấp − Mất việc (Displaced)
        3. Tối ưu phân bổ ngân sách $x_{AI}$ và $x_H$ để **tối đa hóa tổng NetJob**
        4. Đo **ngưỡng đầu tư đào tạo tối thiểu** để bảo đảm không mất việc ròng
        5. Phân tích **lao động dễ bị tổn thương**

        #### 📐 Mô hình Toán học
        $$\\text{NetJob}_i = (a_{1i}\\cdot x^{AI}_i) + (b_{1i}\\cdot x^H_i) - (c_{1i}\\cdot r_i\\cdot x^{AI}_i)$$
        """
    )
    st.markdown("---")

    # ── Dữ liệu ────────────────────────────────────────────────────
    sectors, L, risk, a1, a2, b1, c1, d1 = load_data()
    N = len(sectors)

    st.subheader("📊 Bước 1 — Tham số 8 ngành (Bảng 9.3)")
    df_params = pd.DataFrame({
        "Lao động (M)": L,
        "Risk (%)": risk * 100,
        "a₁ (AI job)": a1,
        "a₂ (DX job)": a2,
        "b₁ (nâng)": b1,
        "c₁ (mất)": c1,
        "d₁ (đào tạo)": d1,
    }, index=[f"{i+1}. {s}" for i, s in enumerate(sectors)])
    st.dataframe(df_params.style.format("{:.2f}"), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng lao động 8 ngành", f"{L.sum():.1f} triệu")
    col2.metric("Ngành rủi ro cao nhất",
                f"{sectors[np.argmax(risk)]} ({np.max(risk)*100:.0f}%)")
    col3.metric("Ngành tạo việc AI cao nhất",
                f"{sectors[np.argmax(a1)]} ({np.max(a1):.1f} việc/tỷ)")

    st.markdown("---")

    # ── Tham số đầu vào ────────────────────────────────────────────
    st.subheader("⚙️ Tham số đầu vào")
    budget = st.slider(
        "Ngân sách tổng (tỷ VND)",
        min_value=10_000, max_value=80_000, value=30_000, step=1_000,
        format="%d",
    )

    # ── Nút chạy mô hình ───────────────────────────────────────────
    if st.button("🚀 Chạy mô hình tối ưu LP (CVXPY)", type="primary"):

        # === CÂU 9.4.1 ===
        with st.spinner("Đang giải LP (Câu 9.4.1)…"):
            x_AI_opt, x_H_opt, Z_opt, err = run_model(
                budget, sectors, L, risk, a1, b1, c1, d1
            )

        st.subheader("✅ Câu 9.4.1 — Kết quả tối ưu phân bổ x_AI và x_H")

        if err:
            st.error(err)
            return

        total_budget_used = x_AI_opt.sum() + x_H_opt.sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Z* (tổng NetJob)", f"{Z_opt:,.1f} ngàn việc")
        col2.metric("Ngân sách sử dụng", f"{total_budget_used:,.0f} tỷ")
        col3.metric("Tỷ lệ sử dụng ngân sách",
                    f"{total_budget_used/budget*100:.1f}%")

        # Bảng kết quả chi tiết
        NewJob_v     = a1 * x_AI_opt
        UpgradeJob_v = b1 * x_H_opt
        Displaced_v  = c1 * risk * x_AI_opt
        NetJob_v     = NewJob_v + UpgradeJob_v - Displaced_v

        df_result = pd.DataFrame({
            "Ngành": sectors,
            "x_AI (tỷ)": x_AI_opt.round(1),
            "x_H (tỷ)": x_H_opt.round(1),
            "NewJob (K)": NewJob_v.round(1),
            "Upgrade (K)": UpgradeJob_v.round(1),
            "Displaced (K)": Displaced_v.round(1),
            "NetJob (K)": NetJob_v.round(1),
        })
        # Tổng
        df_result.loc[len(df_result)] = {
            "Ngành": "**TỔNG**",
            "x_AI (tỷ)": x_AI_opt.sum().round(1),
            "x_H (tỷ)": x_H_opt.sum().round(1),
            "NewJob (K)": NewJob_v.sum().round(1),
            "Upgrade (K)": UpgradeJob_v.sum().round(1),
            "Displaced (K)": Displaced_v.sum().round(1),
            "NetJob (K)": NetJob_v.sum().round(1),
        }
        st.dataframe(df_result, use_container_width=True)

        # Biểu đồ 9.4.1
        fig1 = chart_optimal_allocation(
            sectors, L, risk, a1, b1, c1, d1, x_AI_opt, x_H_opt
        )
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("---")

        # === CÂU 9.4.2 ===
        st.subheader("📈 Câu 9.4.2 — Ngưỡng x_H tối thiểu cho Ngành 2 "
                     "(CN chế biến chế tạo)")
        st.markdown(
            "Điều kiện: $a_{1,2}\\cdot x^{AI}_{2} + b_{1,2}\\cdot x^H_{2,min}"
            "- c_{1,2}\\cdot r_2\\cdot x^{AI}_{2} \\geq 0$"
        )

        fig2, x_ai_opt_2, x_h_min_2 = chart_threshold_sector2(
            a1, b1, c1, risk, x_AI_opt, x_H_opt
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("x_AI tối ưu Ngành 2", f"{x_ai_opt_2:,.1f} tỷ")
        col2.metric("x_H tối thiểu cần thiết", f"{x_h_min_2:,.1f} tỷ")
        col3.metric("x_H mô hình chọn (an toàn hơn)",
                    f"{x_H_opt[1]:,.1f} tỷ (+{x_H_opt[1]-x_h_min_2:,.1f})")

        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("---")

        # === CÂU 9.4.3 ===
        st.subheader("🌊 Câu 9.4.3 — Lao động dễ bị tổn thương "
                     "(Nông lâm, Xây dựng, Bán buôn-bán lẻ)")

        vulnerable_idx   = [0, 2, 3]
        vulnerable_rows  = []
        for idx in vulnerable_idx:
            nj  = a1[idx] * x_AI_opt[idx]
            uj  = b1[idx] * x_H_opt[idx]
            dj  = c1[idx] * risk[idx] * x_AI_opt[idx]
            nj_net = nj + uj - dj
            retrained = min(dj, uj)
            lost      = max(0, dj - retrained)
            vulnerable_rows.append({
                "Ngành": sectors[idx],
                "L (triệu)": L[idx],
                "Risk (%)": risk[idx] * 100,
                "NewJob (K)": round(nj, 1),
                "Upgrade (K)": round(uj, 1),
                "Displaced (K)": round(dj, 1),
                "NetJob ròng (K)": round(nj_net, 1),
                "% Dịch chuyển": round(dj / L[idx] * 100, 2),
                "Được đào tạo lại (K)": round(retrained, 1),
                "Mất việc (K)": round(lost, 1),
            })
        df_vuln = pd.DataFrame(vulnerable_rows)
        st.dataframe(df_vuln, use_container_width=True)

        # Biểu đồ swimming-lane
        fig3, axes3 = plt.subplots(len(vulnerable_idx), 1,
                                   figsize=(13, 3 * len(vulnerable_idx)))
        fig3.suptitle(
            "Mô phỏng Luồng Lao động (3 ngành dễ bị tổn thương)",
            fontsize=12, fontweight="bold"
        )
        for ax_i, idx in enumerate(vulnerable_idx):
            ax = axes3[ax_i]
            displaced = c1[idx] * risk[idx] * x_AI_opt[idx]
            new_jobs  = a1[idx] * x_AI_opt[idx]
            upgrade   = b1[idx] * x_H_opt[idx]
            retrained = min(displaced, upgrade)
            lost      = max(0, displaced - retrained)

            ax.barh(4, L[idx], height=0.6, color="#4CAF50", label="Lao động")
            ax.text(L[idx]/2, 4, f"L={L[idx]:.1f}M", va="center",
                    ha="center", fontweight="bold", fontsize=9)
            ax.barh(3, L[idx] - displaced, height=0.6, color="#4CAF50")
            ax.barh(3, displaced, left=L[idx]-displaced, height=0.6, color="#FF9800")
            ax.text(L[idx]/2, 3,
                    f"Dịch chuyển: {displaced:.1f}K ({displaced/L[idx]*100:.1f}%)",
                    va="center", ha="center", fontweight="bold",
                    fontsize=8, color="white")
            ax.barh(2, retrained, height=0.6, color="#2196F3")
            ax.barh(2, lost, left=retrained, height=0.6, color="#F44336")
            ax.text(retrained/2 if retrained>0 else 0, 2,
                    f"Đào tạo: {retrained:.1f}K", va="center", ha="center",
                    fontweight="bold", fontsize=8)
            ax.set_ylim(0.5, 4.5)
            ax.set_xlim(0, max(L[idx]*1.3, L[idx] + new_jobs + 10))
            ax.set_ylabel(sectors[idx], fontweight="bold")
            ax.set_xlabel("Số người (ngàn)")
            ax.set_yticks([])
            ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.markdown("---")

        # === CÂU 9.4.4 ===
        st.subheader("⚖️ Câu 9.4.4 — Ràng buộc An sinh: "
                     "DisplacedJob ≤ 5% L_i")
        st.info(
            "Ràng buộc mới **C4**: $\\text{DisplacedJob}_i \\leq 0.05 \\times L_i$ ∀i  "
            "— Kiểm tra tính khả thi và chi phí an sinh."
        )

        with st.spinner("Đang giải LP với ràng buộc an sinh…"):
            x_AI_safe, x_H_safe, Z_safe, err_safe = run_model_safe(
                budget, sectors, L, risk, a1, b1, c1, d1,
                x_AI_opt, x_H_opt, Z_opt
            )

        if err_safe and x_AI_safe is None:
            st.error(f"❌ Vô nghiệm: {err_safe}")
            st.warning("Ràng buộc an sinh 5% quá chặt với mức ngân sách hiện tại.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Z* (Không an sinh)", f"{Z_opt:,.1f} K việc")
            col2.metric("Z* (Có an sinh 5%)", f"{Z_safe:,.1f} K việc")
            col3.metric("Chi phí an sinh", f"{Z_opt-Z_safe:,.1f} K việc "
                        f"({(Z_opt-Z_safe)/Z_opt*100:.2f}% Z*)")

            # Bảng kiểm tra tuân thủ
            limit_5pct = 0.05 * L
            safe_rows  = []
            for i in range(N):
                disp_val = c1[i] * risk[i] * x_AI_safe[i]
                safe_rows.append({
                    "Ngành": sectors[i],
                    "Displaced (K)": round(disp_val, 1),
                    "5% L_i (K)":   round(limit_5pct[i], 1),
                    "Tuân thủ": "✅" if disp_val <= limit_5pct[i] + 1e-6 else "❌",
                })
            st.dataframe(pd.DataFrame(safe_rows), use_container_width=True)

            fig4 = chart_safety_comparison(
                sectors, L, risk, a1, b1, c1,
                x_AI_opt, x_H_opt, Z_opt,
                x_AI_safe, x_H_safe, Z_safe,
            )
            st.pyplot(fig4)
            plt.close(fig4)

        st.markdown("---")

        # === TỔNG KẾT ===
        st.subheader("📋 Tổng kết Bài 9")
        st.markdown(f"""
        | Chỉ tiêu | Kết quả |
        |---|---|
        | **Z* (tổng NetJob)** | {Z_opt:,.1f} ngàn việc |
        | **Ngân sách sử dụng** | {total_budget_used:,.0f} / {budget:,} tỷ VND |
        | **Ngành cần x_H nhiều nhất** | {sectors[np.argmax(x_H_opt)]} ({x_H_opt.max():,.1f} tỷ) |
        | **Ngành cần x_AI nhiều nhất** | {sectors[np.argmax(x_AI_opt)]} ({x_AI_opt.max():,.1f} tỷ) |
        | **Ngưỡng x_H Ngành 2** | ≥ {x_h_min_2:,.1f} tỷ khi x_AI={x_ai_opt_2:,.1f} tỷ |
        | **Chi phí an sinh (5%)** | {Z_opt-Z_safe:,.1f} K việc | 
        """)

        st.success(
            "**Kết luận chính sách:** Mô hình thể hiện cân bằng giữa tự động hóa "
            "(tăng năng suất) và an sinh xã hội. Ngành CN chế biến, Bán buôn cần "
            "đầu tư đào tạo lại nhiều nhất. Ràng buộc `DisplacedJob ≤ RetrainCap` "
            "là biểu diễn toán học của 'tốc độ tự động hóa ≤ năng lực đào tạo'."
        )

        with st.expander("💬 Câu hỏi thảo luận chính sách"):
            st.markdown("""
            **a)** Ngành nào cần đầu tư đào tạo lại nhiều nhất theo kết quả tối ưu?  
            *Gợi ý: So sánh x_H_opt[i] giữa các ngành.*

            **b)** Ngành Tài chính-Ngân hàng: nguy cơ thay thế 52% nhưng hệ số tạo việc AI rất cao.  
            *Gợi ý: Ngành 5 có a₁=45.8 cao (tạo việc AI) nhưng risk=52% cao. Mô hình khuyến nghị gì?*

            **c)** Nông-Lâm-Thủy sản: hệ số tạo việc AI thấp (8.5) nhưng lao động dịch chuyển lớn.  
            *Gợi ý: x_AI tối ưu cho ngành 1 bằng bao nhiêu?*

            **d)** "Tốc độ tự động hóa ≤ năng lực đào tạo lại" được biểu diễn bằng công thức:  
            `c₁[i]·risk[i]·x_AI[i] ≤ d₁[i]·x_H[i]`
            """)
if __name__ == "__main__":
    run()