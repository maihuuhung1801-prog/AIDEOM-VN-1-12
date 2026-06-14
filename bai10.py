"""
modules/bai10.py
Bài 10 — Quy Hoạch Ngẫu Nhiên Hai Giai Đoạn
Hoạch Định Ngân Sách Đầu Tư Số Việt Nam 2026–2030
Chuyển đổi từ Jupyter Notebook sang Streamlit module.
Giữ nguyên toàn bộ logic, Pyomo model, dữ liệu gốc.
"""
import pulp
from pyomo.opt import SolverFactory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# 1. KHAI BÁO THAM SỐ & DỮ LIỆU
# ──────────────────────────────────────────────────────────────────
def load_data():
    J    = ["I", "D", "AI", "H"]
    J_vi = {
        "I":  "Hạ tầng số",
        "D":  "CĐS doanh nghiệp",
        "AI": "Trí tuệ nhân tạo",
        "H":  "Nhân lực số",
    }
    S    = ["s1", "s2", "s3", "s4"]
    S_vi = {
        "s1": "Lạc quan (TT=3.5%)",
        "s2": "Cơ sở   (TT=2.8%)",
        "s3": "Bi quan  (TT=1.5%)",
        "s4": "Khủng hoảng (TT=0.2%)",
    }
    p_prob = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
    beta   = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
    beta_s = {
        ("s1","I"):1.25, ("s1","D"):1.35, ("s1","AI"):1.55, ("s1","H"):1.05,
        ("s2","I"):1.00, ("s2","D"):1.10, ("s2","AI"):1.25, ("s2","H"):0.95,
        ("s3","I"):0.75, ("s3","D"):0.85, ("s3","AI"):0.90, ("s3","H"):1.00,
        ("s4","I"):0.40, ("s4","D"):0.50, ("s4","AI"):0.55, ("s4","H"):1.10,
    }
    BUDGET1  = 65_000
    BUDGET2  = 15_000
    floor1   = {"I":5_000,  "D":8_000,  "AI":5_000,  "H":8_000}
    ceil1    = {"I":20_000, "D":25_000, "AI":25_000, "H":25_000}
    target_H = {"s1":8_000, "s2":10_000, "s3":15_000, "s4":20_000}
    PENALTY  = 50
    return (J, J_vi, S, S_vi, p_prob, beta, beta_s,
            BUDGET1, BUDGET2, floor1, ceil1, target_H, PENALTY)


# ──────────────────────────────────────────────────────────────────
# 2. PYOMO MODELS
# ──────────────────────────────────────────────────────────────────
def check_pyomo():
    try:
        import pyomo.environ as pyo
        solver = SolverFactory("cbc", executable=pulp.pulp_cbc_path)
        if not solver.available():
            return False, "Pyomo có sẵn nhưng solver CBC chưa tìm thấy. Đảm bảo PuLP đã cài: pip install pulp"
        return True, None
    except ImportError:
        return False, "Pyomo chưa được cài. Chạy: pip install pyomo"


def build_and_solve_sp(J, S, p_prob, beta, beta_s,
                       BUDGET1, BUDGET2, floor1, ceil1,
                       target_H, PENALTY):
    """Câu 10.5.1 — Two-Stage SP với Pyomo + CBC."""
    import pyomo.environ as pyo

    m = pyo.ConcreteModel(name="VN_Digital_SP")
    m.J = pyo.Set(initialize=J)
    m.S = pyo.Set(initialize=S)
    m.p      = pyo.Param(m.S, initialize=p_prob)
    m.beta   = pyo.Param(m.J, initialize=beta)
    m.beta_s = pyo.Param(m.S, m.J, initialize=beta_s)
    m.tgt_H  = pyo.Param(m.S, initialize=target_H)

    m.x       = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y       = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
    m.slack_H = pyo.Var(m.S, within=pyo.NonNegativeReals)

    m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in J) <= BUDGET1)
    m.floor_c = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] >= floor1[j])
    m.ceil_c  = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] <= ceil1[j])
    m.budget2 = pyo.Constraint(m.S,
        rule=lambda m,s: sum(m.y[s,j] for j in J) <= BUDGET2)
    m.ai_cap  = pyo.Constraint(m.S,
        rule=lambda m,s: m.y[s,"AI"] <= 0.5 * m.x["H"])
    m.h_target = pyo.Constraint(m.S,
        rule=lambda m,s: m.x["H"] + m.y[s,"H"] + m.slack_H[s] >= m.tgt_H[s])

    def obj_rule(m):
        first  = sum(m.beta[j] * m.x[j] for j in J)
        second = sum(m.p[s] * sum(m.beta_s[s,j] * m.y[s,j] for j in J)
                     for s in S)
        penalty = PENALTY * sum(m.p[s] * m.slack_H[s] for s in S)
        return first + second - penalty
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    result = SolverFactory("cbc", executable=pulp.pulp_cbc_path).solve(m, tee=False)
    Z_sp = pyo.value(m.obj)
    x_sp = {j: pyo.value(m.x[j]) for j in J}
    y_sp = {(s,j): pyo.value(m.y[s,j]) for s in S for j in J}
    slk  = {s: pyo.value(m.slack_H[s]) for s in S}
    return Z_sp, x_sp, y_sp, slk, str(result.solver.termination_condition)


def solve_ev(J, S, p_prob, beta, beta_s, BUDGET1, BUDGET2,
             floor1, ceil1, target_H, PENALTY):
    """Giải EV (Expected Value scenario)."""
    import pyomo.environ as pyo
    beta_ev   = {j: sum(p_prob[s] * beta_s[(s,j)] for s in S) for j in J}
    target_ev = sum(p_prob[s] * target_H[s] for s in S)

    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=J)
    m.x   = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y   = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.slk = pyo.Var(within=pyo.NonNegativeReals)
    m.b1  = pyo.Constraint(expr=sum(m.x[j] for j in J) <= BUDGET1)
    m.fl  = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] >= floor1[j])
    m.cl  = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] <= ceil1[j])
    m.b2  = pyo.Constraint(expr=sum(m.y[j] for j in J) <= BUDGET2)
    m.aic = pyo.Constraint(expr=m.y["AI"] <= 0.5 * m.x["H"])
    m.htg = pyo.Constraint(expr=m.x["H"] + m.y["H"] + m.slk >= target_ev)
    m.obj = pyo.Objective(
        expr=sum(beta[j]*m.x[j] for j in J)
           + sum(beta_ev[j]*m.y[j] for j in J)
           - PENALTY*m.slk,
        sense=pyo.maximize)
    SolverFactory("cbc", executable=pulp.pulp_cbc_path).solve(m, tee=False)
    return (pyo.value(m.obj),
            {j: pyo.value(m.x[j]) for j in J},
            {j: pyo.value(m.y[j]) for j in J},
            {j: round(beta_ev[j],3) for j in J})


def solve_scenario_fix(s, J, beta, beta_s, BUDGET2, target_H,
                       PENALTY, x_fix):
    """Giải deterministic kịch bản s với first-stage cố định."""
    import pyomo.environ as pyo
    m = pyo.ConcreteModel()
    m.J   = pyo.Set(initialize=J)
    m.y   = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.slk = pyo.Var(within=pyo.NonNegativeReals)
    m.b2  = pyo.Constraint(expr=sum(m.y[j] for j in J) <= BUDGET2)
    m.aic = pyo.Constraint(expr=m.y["AI"] <= 0.5 * x_fix["H"])
    m.htg = pyo.Constraint(
        expr=x_fix["H"] + m.y["H"] + m.slk >= target_H[s])
    m.obj = pyo.Objective(
        expr=sum(beta[j]*x_fix[j] for j in J)
           + sum(beta_s[(s,j)]*m.y[j] for j in J)
           - PENALTY*m.slk,
        sense=pyo.maximize)
    SolverFactory("cbc", executable=pulp.pulp_cbc_path).solve(m, tee=False)
    return pyo.value(m.obj), {j: pyo.value(m.y[j]) for j in J}


def solve_scenario_free(s, J, beta, beta_s, BUDGET1, BUDGET2,
                        floor1, ceil1, target_H, PENALTY):
    """Giải deterministic kịch bản s (tự do first-stage)."""
    import pyomo.environ as pyo
    m = pyo.ConcreteModel()
    m.J   = pyo.Set(initialize=J)
    m.x   = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y   = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.slk = pyo.Var(within=pyo.NonNegativeReals)
    m.b1  = pyo.Constraint(expr=sum(m.x[j] for j in J) <= BUDGET1)
    m.fl  = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] >= floor1[j])
    m.cl  = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] <= ceil1[j])
    m.b2  = pyo.Constraint(expr=sum(m.y[j] for j in J) <= BUDGET2)
    m.aic = pyo.Constraint(expr=m.y["AI"] <= 0.5*m.x["H"])
    m.htg = pyo.Constraint(
        expr=m.x["H"] + m.y["H"] + m.slk >= target_H[s])
    m.obj = pyo.Objective(
        expr=sum(beta[j]*m.x[j] for j in J)
           + sum(beta_s[(s,j)]*m.y[j] for j in J)
           - PENALTY*m.slk,
        sense=pyo.maximize)
    SolverFactory("cbc", executable=pulp.pulp_cbc_path).solve(m, tee=False)
    return (pyo.value(m.obj),
            {j: pyo.value(m.x[j]) for j in J},
            {j: pyo.value(m.y[j]) for j in J})


def solve_robust_minimax(J, S, beta, beta_s, BUDGET1, BUDGET2,
                         floor1, ceil1, target_H, PENALTY, WS_by_s):
    """Câu 10.5.4 — Minimax Regret."""
    import pyomo.environ as pyo
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=J)
    m.S = pyo.Set(initialize=S)
    m.x     = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y     = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
    m.slk   = pyo.Var(m.S, within=pyo.NonNegativeReals)
    m.theta = pyo.Var(within=pyo.NonNegativeReals)

    m.b1  = pyo.Constraint(expr=sum(m.x[j] for j in J) <= BUDGET1)
    m.fl  = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] >= floor1[j])
    m.cl  = pyo.Constraint(m.J, rule=lambda m,j: m.x[j] <= ceil1[j])
    m.b2  = pyo.Constraint(m.S,
        rule=lambda m,s: sum(m.y[s,j] for j in J) <= BUDGET2)
    m.aic = pyo.Constraint(m.S,
        rule=lambda m,s: m.y[s,"AI"] <= 0.5*m.x["H"])
    m.htg = pyo.Constraint(m.S,
        rule=lambda m,s: m.x["H"]+m.y[s,"H"]+m.slk[s] >= target_H[s])

    def regret_rule(m, s):
        Z_s = (sum(beta[j]*m.x[j] for j in J)
             + sum(beta_s[(s,j)]*m.y[s,j] for j in J)
             - PENALTY*m.slk[s])
        return m.theta >= WS_by_s[s] - Z_s
    m.regret_c = pyo.Constraint(m.S, rule=regret_rule)
    m.obj = pyo.Objective(expr=m.theta, sense=pyo.minimize)
    solver = SolverFactory("cbc", executable=pulp.pulp_cbc_path)
    solver.solve(m, tee=False)
    return pyo.value(m.theta), {j: pyo.value(m.x[j]) for j in J}


# ──────────────────────────────────────────────────────────────────
# 3. BIỂU ĐỒ
# ──────────────────────────────────────────────────────────────────
def chart_vss_evpi(S, S_vi, p_prob, WS_by_s, Z_sp, EEV, WS, VSS, EVPI):
    """Câu 10.5.3 — Waterfall + WS per scenario."""
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    bar_colors = ["#2ca02c","#4878d0","#ff7f0e","#d62728"]

    ax = axes[0]
    bars = ax.bar([S_vi[s][:15] for s in S], [WS_by_s[s] for s in S],
                  color=bar_colors, edgecolor="white", width=0.6, alpha=0.85)
    ax.axhline(Z_sp, color="navy",  ls="--",  lw=2.0, label=f"SP = {Z_sp:,.0f}")
    ax.axhline(WS,   color="green", ls=":",   lw=2.0, label=f"WS = {WS:,.0f}")
    ax.axhline(EEV,  color="tomato",ls="-.",  lw=2.0, label=f"EEV = {EEV:,.0f}")
    for bar, val in zip(bars, [WS_by_s[s] for s in S]):
        ax.text(bar.get_x()+bar.get_width()/2, val+200,
                f"{val:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("GDP gain kỳ vọng (tỷ VND)")
    ax.set_title("GDP Gain: Wait-and-See từng Kịch bản vs SP vs EEV",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.tick_params(axis="x", rotation=15)

    ax2 = axes[1]
    categories = ["EEV", "+ VSS", "SP", "+ EVPI", "WS"]
    values      = [EEV, VSS, Z_sp, EVPI, WS]
    bottoms     = [0,   EEV, 0,    Z_sp, 0]
    colors_wf   = ["#4878d0","#2ca02c","#4878d0","#ff7f0e","#2ca02c"]
    for i, (cat, val, bot, clr) in enumerate(
            zip(categories, values, bottoms, colors_wf)):
        if cat in ("EEV","SP","WS"):
            ax2.bar(cat, val, color=clr, edgecolor="white", width=0.6, alpha=0.85)
            ax2.text(i, val+50, f"{val:,.0f}", ha="center",
                     fontsize=9, fontweight="bold")
        else:
            ax2.bar(cat, val, bottom=bot, color=clr,
                    edgecolor="white", width=0.6, alpha=0.85)
            ax2.text(i, bot+val/2, f"+{val:,.0f}", ha="center",
                     fontsize=9, color="white", fontweight="bold")
    ax2.set_ylabel("GDP gain kỳ vọng (tỷ VND)")
    ax2.set_title(f"Waterfall: EEV → SP → WS\nVSS={VSS:,.0f} | EVPI={EVPI:,.0f}",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.set_ylim(EEV*0.995, WS*1.005)
    plt.suptitle("VSS & EVPI — Giá trị của Tư duy Xác suất & Thông tin Hoàn hảo",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def chart_robust_comparison(J, J_vi, S, S_vi, p_prob, beta, beta_s,
                             BUDGET2, target_H, PENALTY, WS_by_s,
                             x_sp, x_ev, x_rob):
    """Câu 10.5.4 — Bar grouped + Regret."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    x_pos = np.arange(len(J))
    bw = 0.27
    axes[0].bar(x_pos-bw, [x_sp[j]  for j in J], bw,
                label="SP", color="#4878d0", alpha=0.85)
    axes[0].bar(x_pos,    [x_ev[j]  for j in J], bw,
                label="EV", color="#e8625a", alpha=0.85)
    axes[0].bar(x_pos+bw, [x_rob[j] for j in J], bw,
                label="Robust", color="#2ca02c", alpha=0.85)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([J_vi[j] for j in J], fontsize=10)
    axes[0].set_ylabel("Ngân sách (tỷ VND)")
    axes[0].set_title("Phân bổ First-stage: SP vs EV vs Robust",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    strategies = {"SP": x_sp, "EV": x_ev, "Robust": x_rob}
    colors_s   = ["#4878d0","#e8625a","#2ca02c"]
    x_s        = np.arange(len(S))
    bw2 = 0.27
    for i, (name, xd) in enumerate(strategies.items()):
        rgt = []
        for s in S:
            Za, _ = solve_scenario_fix(s, J, beta, beta_s, BUDGET2,
                                       target_H, PENALTY, xd)
            rgt.append(max(0, WS_by_s[s] - Za))
        axes[1].bar(x_s + (i-1)*bw2, rgt, bw2, label=name,
                    color=colors_s[i], alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x_s)
    axes[1].set_xticklabels([S_vi[s][:15] for s in S], rotation=15, fontsize=8.5)
    axes[1].set_ylabel("Regret (tỷ VND)")
    axes[1].set_title("Regret theo Kịch bản\n(Robust minimize max regret)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    plt.suptitle("Robust Optimization — Minimax Regret vs SP",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def chart_heatmap_strategies(J, J_vi, x_sp, x_ev, x_rob):
    """Heatmap phân bổ 3 chiến lược."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 4))
    data = pd.DataFrame({
        "SP"    : {J_vi[j]: round(x_sp[j])  for j in J},
        "EV"    : {J_vi[j]: round(x_ev[j])  for j in J},
        "Robust": {J_vi[j]: round(x_rob[j]) for j in J},
    })
    sns.heatmap(data.T, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label":"tỷ VND","shrink":0.8}, ax=ax)
    ax.set_title("Phân bổ First-stage: SP vs EV vs Robust (tỷ VND)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────
# 4. HÀM RUN() CHÍNH
# ──────────────────────────────────────────────────────────────────
def run():
    st.title("📘 Bài 10 — Quy Hoạch Ngẫu Nhiên Hai Giai Đoạn")
    st.subheader("Hoạch Định Ngân Sách Đầu Tư Số Việt Nam 2026–2030")
    st.markdown(
        """
        **Bài toán:** Phân bổ 80.000 tỷ VND ngân sách đầu tư số trong điều kiện bất định
        về tăng trưởng kinh tế toàn cầu, FDI, và xuất khẩu.

        **Cấu trúc mô hình:**
        $$\\max \\sum_j \\beta_j x_j + \\sum_{s \\in S} p_s
        \\left[ \\sum_j \\beta_j^s y_j^s - \\text{Penalty}(s) \\right]$$

        | Giai đoạn | Quyết định | Thời điểm |
        |---|---|---|
        | **First-stage** | $x_j$ — phân bổ ban đầu (≤65.000 tỷ) | Trước khi biết kịch bản |
        | **Second-stage** | $y_j^s$ — điều chỉnh bổ sung (≤15.000 tỷ/KB) | Sau khi kịch bản xảy ra |

        **Chỉ số quan trọng:** VSS = SP − EEV | EVPI = WS − SP
        """
    )
    st.markdown("---")

    # ── Dữ liệu ────────────────────────────────────────────────────
    (J, J_vi, S, S_vi, p_prob, beta, beta_s,
     BUDGET1, BUDGET2, floor1, ceil1, target_H, PENALTY) = load_data()

    # ── Hiển thị tham số ────────────────────────────────────────────
    st.subheader("📋 Bước 1 — Tham số Kịch bản & Hệ số β")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bảng Kịch bản Bất định**")
        df_scen = pd.DataFrame({
            "Kịch bản":       [S_vi[s] for s in S],
            "Xác suất":        [p_prob[s] for s in S],
            "TT Toàn cầu (%)": [3.5, 2.8, 1.5, 0.2],
            "FDI VN (tỷ USD)": [32, 27, 20, 12],
            "XK VN tăng (%)":  [12, 8, 3, -5],
            "Target H (tỷ)":   [target_H[s] for s in S],
        }).set_index("Kịch bản")
        st.dataframe(df_scen, use_container_width=True)

    with col2:
        st.markdown("**Hệ số β theo kịch bản**")
        df_beta = pd.DataFrame({
            "β cơ bản":   beta,
            **{S_vi[s]: {j: beta_s[(s,j)] for j in J} for s in S}
        }, index=J).rename(index=J_vi)
        st.dataframe(df_beta, use_container_width=True)

    st.markdown(f"Ngân sách tổng: **{BUDGET1+BUDGET2:,} tỷ VND** "
                f"(First: {BUDGET1:,} | Second: {BUDGET2:,}/kịch bản) | "
                f"Phạt shortfall nhân lực: {PENALTY}/tỷ")

    # ── Kiểm tra Pyomo ──────────────────────────────────────────────
    ok, pyomo_err = check_pyomo()
    if not ok:
        st.error(f"⚠️ {pyomo_err}")
        st.stop()

    st.markdown("---")
    st.subheader("⚙️ Tham số mô hình")
    penalty_input = st.slider("Hệ số phạt shortfall nhân lực (PENALTY)",
                               10, 200, PENALTY, step=10)
    PENALTY = penalty_input

    if st.button("🚀 Chạy mô hình Stochastic Programming (Pyomo + CBC)",
                 type="primary"):

        # === CÂU 10.5.1 — SP ===
        with st.spinner("Đang giải mô hình SP hai giai đoạn…"):
            Z_sp, x_sp, y_sp, slk, status = build_and_solve_sp(
                J, S, p_prob, beta, beta_s,
                BUDGET1, BUDGET2, floor1, ceil1, target_H, PENALTY
            )

        st.subheader("✅ Câu 10.5.1 — Kết quả Stochastic Programming")
        col1, col2, col3 = st.columns(3)
        col1.metric("Trạng thái solver", status)
        col2.metric("Z* (GDP gain kỳ vọng)", f"{Z_sp:,.2f} tỷ VND")
        col3.metric("Tổng first-stage",
                    f"{sum(x_sp.values()):,.0f} / {BUDGET1:,} tỷ")

        # Bảng first-stage
        df_x = pd.DataFrame({
            "Hạng mục":     [J_vi[j] for j in J],
            "Phân bổ (tỷ)": [round(x_sp[j]) for j in J],
            "Tỷ trọng (%)": [round(x_sp[j]/BUDGET1*100, 1) for j in J],
            "Sàn (tỷ)":     [floor1[j] for j in J],
            "Trần (tỷ)":    [ceil1[j]  for j in J],
        }).set_index("Hạng mục")
        st.markdown("**Quyết định First-stage x_j:**")
        st.dataframe(df_x, use_container_width=True)

        # Bảng second-stage
        df_y = pd.DataFrame(
            [[round(y_sp[(s,j)]) for j in J]
             + [round(sum(y_sp[(s,j)] for j in J)), round(slk[s])]
             for s in S],
            index=[S_vi[s] for s in S],
            columns=[J_vi[j] for j in J] + ["Tổng", "Slack_H"]
        )
        st.markdown("**Quyết định Second-stage y_j^s (tỷ VND):**")
        st.dataframe(df_y, use_container_width=True)

        st.markdown("---")

        # === CÂU 10.5.2 — SP vs EV ===
        st.subheader("📌 Câu 10.5.2 — So sánh SP vs EV (Expected Value)")
        with st.spinner("Đang giải EV…"):
            _, x_ev, y_ev, beta_ev_v = solve_ev(
                J, S, p_prob, beta, beta_s,
                BUDGET1, BUDGET2, floor1, ceil1, target_H, PENALTY
            )
            EEV = sum(
                p_prob[s] * solve_scenario_fix(
                    s, J, beta, beta_s, BUDGET2, target_H, PENALTY, x_ev
                )[0]
                for s in S
            )

        df_cmp = pd.DataFrame({
            "Hạng mục":    [J_vi[j] for j in J],
            "x_SP (tỷ)":   [round(x_sp[j]) for j in J],
            "x_EV (tỷ)":   [round(x_ev[j]) for j in J],
            "Chênh lệch":  [round(x_sp[j]-x_ev[j]) for j in J],
        }).set_index("Hạng mục")
        st.dataframe(df_cmp, use_container_width=True)

        h_diff = x_sp["H"] - x_ev["H"]
        st.info(
            f"SP đầu tư H = **{x_sp['H']:,.0f} tỷ** | EV = **{x_ev['H']:,.0f} tỷ**  \n"
            f"→ SP đầu tư H **{'nhiều hơn' if h_diff>0 else 'ít hơn'}** {abs(h_diff):,.0f} tỷ "
            f"(tính đến kịch bản s3/s4 cần nhân lực cao)."
        )

        # Quyết định theo từng kịch bản
        with st.spinner("Đang giải quyết định từng kịch bản…"):
            rows_det = []
            for s in S:
                _, xs, _ = solve_scenario_free(
                    s, J, beta, beta_s, BUDGET1, BUDGET2,
                    floor1, ceil1, target_H, PENALTY
                )
                rows_det.append(
                    [S_vi[s], p_prob[s]] + [round(xs[j]) for j in J]
                )
        df_det = pd.DataFrame(
            rows_det,
            columns=["Kịch bản","p"] + [J_vi[j] for j in J]
        ).set_index("Kịch bản")
        st.markdown("**Quyết định tối ưu theo từng kịch bản riêng lẻ:**")
        st.dataframe(df_det, use_container_width=True)
        st.caption(
            "💡 Kịch bản s3/s4 → đầu tư H nhiều hơn (nhân lực số là bảo hiểm khi khủng hoảng); "
            "s1 → AI nhiều hơn (tận dụng hệ số β_AI=1.55 cao)."
        )

        st.markdown("---")

        # === CÂU 10.5.3 — VSS & EVPI ===
        st.subheader("📌 Câu 10.5.3 — Tính VSS và EVPI")
        with st.spinner("Đang tính WS (Wait-and-See)…"):
            WS_by_s = {}
            for s in S:
                Z_ws, _, _ = solve_scenario_free(
                    s, J, beta, beta_s, BUDGET1, BUDGET2,
                    floor1, ceil1, target_H, PENALTY
                )
                WS_by_s[s] = Z_ws
            WS   = sum(p_prob[s]*WS_by_s[s] for s in S)
            VSS  = Z_sp - EEV
            EVPI = WS   - Z_sp

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("WS (Perfect Info)", f"{WS:,.2f} tỷ")
        col2.metric("SP (Stochastic)",   f"{Z_sp:,.2f} tỷ")
        col3.metric("EEV (EV applied)",  f"{EEV:,.2f} tỷ")
        col4.metric("VSS = SP − EEV",    f"{VSS:,.2f} tỷ",
                    f"{VSS/Z_sp*100:.3f}% Z*")
        col5.metric("EVPI = WS − SP",    f"{EVPI:,.2f} tỷ",
                    f"{EVPI/Z_sp*100:.3f}% Z*")

        df_sum = pd.DataFrame({
            "Chỉ số":        ["WS","SP","EEV","VSS","EVPI"],
            "Giá trị (tỷ)":  [WS, Z_sp, EEV, VSS, EVPI],
            "Ý nghĩa":       [
                "Biết trước kịch bản → tối ưu hoàn hảo",
                "Tối ưu xác suất → giải pháp SP",
                "Dùng quyết định EV đơn giản cho thực tế",
                "Lợi ích của tư duy xác suất",
                "Giá trị tối đa của thông tin hoàn hảo",
            ]
        }).set_index("Chỉ số")
        st.dataframe(df_sum, use_container_width=True)

        fig_vss = chart_vss_evpi(
            S, S_vi, p_prob, WS_by_s, Z_sp, EEV, WS, VSS, EVPI
        )
        st.pyplot(fig_vss)
        plt.close(fig_vss)

        if EEV <= Z_sp <= WS:
            st.success(f"✅ Bất biến SP thỏa mãn: EEV ({EEV:,.0f}) ≤ SP ({Z_sp:,.0f}) ≤ WS ({WS:,.0f})")

        st.markdown("---")

        # === CÂU 10.5.4 — ROBUST ===
        st.subheader("📌 Câu 10.5.4 (Mở rộng) — Robust Optimization "
                     "(Minimax Regret)")

        # Regret của SP
        st.markdown("**Regret của quyết định SP theo từng kịch bản:**")
        regret_rows = []
        for s in S:
            Z_actual, _ = solve_scenario_fix(
                s, J, beta, beta_s, BUDGET2, target_H, PENALTY, x_sp
            )
            regret_rows.append({
                "Kịch bản": S_vi[s],
                "WS[s] (tỷ)": round(WS_by_s[s]),
                "Z_actual (tỷ)": round(Z_actual),
                "Regret (tỷ)": round(WS_by_s[s]-Z_actual, 2),
            })
        st.dataframe(pd.DataFrame(regret_rows).set_index("Kịch bản"),
                     use_container_width=True)

        with st.spinner("Đang giải Minimax Regret…"):
            theta_rob, x_rob = solve_robust_minimax(
                J, S, beta, beta_s, BUDGET1, BUDGET2,
                floor1, ceil1, target_H, PENALTY, WS_by_s
            )

        # Bảng so sánh 3 chiến lược
        df_rob = pd.DataFrame({
            "Hạng mục":       [J_vi[j] for j in J],
            "x_SP (tỷ)":      [round(x_sp[j])  for j in J],
            "x_EV (tỷ)":      [round(x_ev[j])  for j in J],
            "x_Robust (tỷ)":  [round(x_rob[j]) for j in J],
        }).set_index("Hạng mục")
        st.markdown("**So sánh 3 chiến lược quyết định:**")
        st.dataframe(df_rob, use_container_width=True)

        max_regret_sp = max(
            WS_by_s[s] - solve_scenario_fix(
                s, J, beta, beta_s, BUDGET2, target_H, PENALTY, x_sp
            )[0]
            for s in S
        )
        col1, col2 = st.columns(2)
        col1.metric("Minimax Regret", f"{theta_rob:,.2f} tỷ VND")
        col2.metric("Max Regret (SP)", f"{max_regret_sp:,.2f} tỷ VND")
        st.info(
            f"Robust ưu tiên H = **{x_rob['H']:,.0f} tỷ** (bảo hiểm kịch bản xấu) "
            f"so với SP = **{x_sp['H']:,.0f} tỷ**."
        )

        fig_rob = chart_robust_comparison(
            J, J_vi, S, S_vi, p_prob, beta, beta_s,
            BUDGET2, target_H, PENALTY, WS_by_s,
            x_sp, x_ev, x_rob
        )
        st.pyplot(fig_rob)
        plt.close(fig_rob)

        fig_heat = chart_heatmap_strategies(J, J_vi, x_sp, x_ev, x_rob)
        st.pyplot(fig_heat)
        plt.close(fig_heat)

        st.markdown("---")

        # === TỔNG KẾT ===
        st.subheader("📋 Tổng kết Bài 10")
        st.markdown(f"""
        | Chỉ số | Giá trị (tỷ VND) | Ý nghĩa |
        |---|---|---|
        | WS | {WS:,.2f} | Biết trước kịch bản → hoàn hảo |
        | SP | {Z_sp:,.2f} | Quyết định xác suất tối ưu |
        | EEV | {EEV:,.2f} | Dùng quyết định EV đơn giản |
        | **VSS** | **{VSS:,.2f}** | Lợi ích tư duy xác suất |
        | **EVPI** | **{EVPI:,.2f}** | Giá trị thông tin hoàn hảo |
        """)

        with st.expander("💬 Thảo luận chính sách"):
            st.markdown(f"""
            **a) SP đầu tư H nhiều hơn hay ít hơn lời giải xác định?**
            SP đầu tư H **nhiều hơn** ({x_sp['H']:,.0f} > {x_ev['H']:,.0f} tỷ).
            Lý do: SP tính đủ tail risk kịch bản s3/s4 → nhân lực là "bảo hiểm kinh tế số".

            **b) VSS dương — Ý nghĩa đối với Việt Nam:**
            VSS = {VSS:,.2f} tỷ > 0 chứng minh tư duy xác suất tạo ra giá trị thực.
            Nền kinh tế Việt Nam độ mở ~180% GDP → cực kỳ nhạy cảm với cú sốc bên ngoài.

            **c) COVID-19 & Bão Yagi — Bài học về dưới đầu tư nhân lực số:**
            - COVID-19 (2020): GDP chỉ 2.91%, lao động số là "escape hatch" (remote work, e-commerce).
            - Bão Yagi (2024): Thiệt hại ~40.000 tỷ, hạ tầng số phục hồi nhanh hơn.
            - **Khuyến nghị:** Ngân sách đào tạo kỹ năng số nên là "ngân sách không thể cắt"
            trong kế hoạch 2026–2030.
            """)
if __name__ == "__main__":
    run()