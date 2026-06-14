# -*- coding: utf-8 -*-
"""
modules/bai10.py
Bai 10 - Quy Hoach Ngau Nhien Hai Giai Doan
Hoach Dinh Ngan Sach Dau Tu So Viet Nam 2026-2030
Chuyen doi tu Jupyter Notebook sang Streamlit module.
Giu nguyen toan bo logic, Pyomo model, du lieu goc.
"""
import pulp
from pyomo.opt import SolverFactory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# CBC PATH HELPER - compatible PuLP old (<=1.x) and new (>=2.x)
# ------------------------------------------------------------------
def _get_cbc_path() -> str:
    """CBC binary path, compatible with all PuLP versions."""
    # PuLP >= 2.x: use PULP_CBC_CMD().path
    if hasattr(pulp, "PULP_CBC_CMD"):
        try:
            return pulp.PULP_CBC_CMD().path
        except Exception:
            pass
    # PuLP <= 1.x: use pulp_cbc_path (old attribute)
    if hasattr(pulp, "pulp_cbc_path"):
        return pulp.pulp_cbc_path
    raise RuntimeError(
        "CBC solver not found. Run: pip install pulp"
    )


def _cbc_solver():
    """Create Pyomo SolverFactory pointing to PuLP CBC binary.
    Uses subprocess options to suppress CBC stdout (avoids UTF-8 decode errors
    on Windows where CBC may print non-UTF-8 bytes).
    """
    solver = SolverFactory("cbc", executable=_get_cbc_path())
    return solver


def _solve_silent(solver, model):
    """Solve model, suppressing all CBC output to avoid encoding errors on Windows."""
    import io, os, sys
    # Redirect stdout/stderr at OS level to devnull
    devnull = open(os.devnull, 'w', encoding='utf-8', errors='replace')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        result = solver.solve(model, tee=False, logfile=os.devnull)
    except Exception:
        # Try with logfile only
        result = solver.solve(model, tee=False)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
    return result

# ------------------------------------------------------------------
# 1. KHAI BAO THAM SO & DU LIEU
# ------------------------------------------------------------------
def load_data():
    J    = ["I", "D", "AI", "H"]
    J_vi = {
        "I":  "Ha tang so",
        "D":  "CDS doanh nghiep",
        "AI": "Tri tue nhan tao",
        "H":  "Nhan luc so",
    }
    S    = ["s1", "s2", "s3", "s4"]
    S_vi = {
        "s1": "Lac quan (TT=3.5%)",
        "s2": "Co so   (TT=2.8%)",
        "s3": "Bi quan  (TT=1.5%)",
        "s4": "Khung hoang (TT=0.2%)",
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


# ------------------------------------------------------------------
# 2. PYOMO MODELS
# ------------------------------------------------------------------
def check_pyomo():
    try:
        import pyomo.environ as pyo
        solver = _cbc_solver()
        try:
            avail = solver.available()
        except Exception:
            avail = False
        if not avail:
            return False, "Pyomo found but CBC solver not available. Install PuLP: pip install pulp"
        return True, None
    except ImportError:
        return False, "Pyomo not installed. Run: pip install pyomo"


def build_and_solve_sp(J, S, p_prob, beta, beta_s,
                       BUDGET1, BUDGET2, floor1, ceil1,
                       target_H, PENALTY):
    """Cau 10.5.1 - Two-Stage SP with Pyomo + CBC."""
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

    result = _solve_silent(_cbc_solver(), m)
    Z_sp = pyo.value(m.obj)
    x_sp = {j: pyo.value(m.x[j]) for j in J}
    y_sp = {(s,j): pyo.value(m.y[s,j]) for s in S for j in J}
    slk  = {s: pyo.value(m.slack_H[s]) for s in S}
    return Z_sp, x_sp, y_sp, slk, str(result.solver.termination_condition)


def solve_ev(J, S, p_prob, beta, beta_s, BUDGET1, BUDGET2,
             floor1, ceil1, target_H, PENALTY):
    """Solve EV (Expected Value scenario)."""
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
    _solve_silent(_cbc_solver(), m)
    return (pyo.value(m.obj),
            {j: pyo.value(m.x[j]) for j in J},
            {j: pyo.value(m.y[j]) for j in J},
            {j: round(beta_ev[j],3) for j in J})


def solve_scenario_fix(s, J, beta, beta_s, BUDGET2, target_H,
                       PENALTY, x_fix):
    """Solve deterministic scenario s with fixed first-stage."""
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
    _solve_silent(_cbc_solver(), m)
    return pyo.value(m.obj), {j: pyo.value(m.y[j]) for j in J}


def solve_scenario_free(s, J, beta, beta_s, BUDGET1, BUDGET2,
                        floor1, ceil1, target_H, PENALTY):
    """Solve deterministic scenario s (free first-stage)."""
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
    _solve_silent(_cbc_solver(), m)
    return (pyo.value(m.obj),
            {j: pyo.value(m.x[j]) for j in J},
            {j: pyo.value(m.y[j]) for j in J})


def solve_robust_minimax(J, S, beta, beta_s, BUDGET1, BUDGET2,
                         floor1, ceil1, target_H, PENALTY, WS_by_s):
    """Cau 10.5.4 - Minimax Regret."""
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
    _solve_silent(_cbc_solver(), m)
    return pyo.value(m.theta), {j: pyo.value(m.x[j]) for j in J}


# ------------------------------------------------------------------
# 3. BIEU DO
# ------------------------------------------------------------------
def chart_vss_evpi(S, S_vi, p_prob, WS_by_s, Z_sp, EEV, WS, VSS, EVPI):
    """Cau 10.5.3 - Waterfall + WS per scenario."""
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
    ax.set_ylabel("GDP gain ky vong (ty VND)")
    ax.set_title("GDP Gain: Wait-and-See by Scenario vs SP vs EEV",
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
    ax2.set_ylabel("GDP gain ky vong (ty VND)")
    ax2.set_title(f"Waterfall: EEV -> SP -> WS\nVSS={VSS:,.0f} | EVPI={EVPI:,.0f}",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.set_ylim(EEV*0.995, WS*1.005)
    plt.suptitle("VSS & EVPI - Gia tri Tu duy Xac suat & Thong tin Hoan hao",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def chart_robust_comparison(J, J_vi, S, S_vi, p_prob, beta, beta_s,
                             BUDGET2, target_H, PENALTY, WS_by_s,
                             x_sp, x_ev, x_rob):
    """Cau 10.5.4 - Bar grouped + Regret."""
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
    axes[0].set_ylabel("Ngan sach (ty VND)")
    axes[0].set_title("Phan bo First-stage: SP vs EV vs Robust",
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
    axes[1].set_ylabel("Regret (ty VND)")
    axes[1].set_title("Regret theo Kich ban\n(Robust minimize max regret)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    plt.suptitle("Robust Optimization - Minimax Regret vs SP",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def chart_heatmap_strategies(J, J_vi, x_sp, x_ev, x_rob):
    """Heatmap phan bo 3 chien luoc."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 4))
    data = pd.DataFrame({
        "SP"    : {J_vi[j]: round(x_sp[j])  for j in J},
        "EV"    : {J_vi[j]: round(x_ev[j])  for j in J},
        "Robust": {J_vi[j]: round(x_rob[j]) for j in J},
    })
    sns.heatmap(data.T, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label":"ty VND","shrink":0.8}, ax=ax)
    ax.set_title("Phan bo First-stage: SP vs EV vs Robust (ty VND)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# 4. HAM RUN() CHINH
# ------------------------------------------------------------------
def run():
    st.title(" Bai 10 - Quy Hoach Ngau Nhien Hai Giai Doan")
    st.subheader("Hoach Dinh Ngan Sach Dau Tu So Viet Nam 2026-2030")
    st.markdown(
        """
        **Bai toan:** Phan bo 80.000 ty VND ngan sach dau tu so trong dieu kien bat dinh
        ve tang truong kinh te toan cau, FDI, va xuat khau.

        **Cau truc mo hinh:**
        $$\\max \\sum_j \\beta_j x_j + \\sum_{s \\in S} p_s
        \\left[ \\sum_j \\beta_j^s y_j^s - \\text{Penalty}(s) \\right]$$

        | Giai doan | Quyet dinh | Thoi diem |
        |---|---|---|
        | **First-stage** | $x_j$ - phan bo ban dau (<=65.000 ty) | Truoc khi biet kich ban |
        | **Second-stage** | $y_j^s$ - dieu chinh bo sung (<=15.000 ty/KB) | Sau khi kich ban xay ra |

        **Chi so quan trong:** VSS = SP - EEV | EVPI = WS - SP
        """
    )
    st.markdown("---")

    # ------------------------------------------------------------------ Du lieu 
    (J, J_vi, S, S_vi, p_prob, beta, beta_s,
     BUDGET1, BUDGET2, floor1, ceil1, target_H, PENALTY) = load_data()

    # ------------------------------------------------------------------ Hien thi tham so 
    st.subheader(" Buoc 1 - Tham so Kich ban & He so Beta")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bang Kich ban Bat dinh**")
        df_scen = pd.DataFrame({
            "Kich ban":       [S_vi[s] for s in S],
            "Xac suat":        [p_prob[s] for s in S],
            "TT Toan cau (%)": [3.5, 2.8, 1.5, 0.2],
            "FDI VN (ty USD)": [32, 27, 20, 12],
            "XK VN tang (%)":  [12, 8, 3, -5],
            "Target H (ty)":   [target_H[s] for s in S],
        }).set_index("Kich ban")
        st.dataframe(df_scen, use_container_width=True)

    with col2:
        st.markdown("**He so beta theo kich ban**")
        df_beta = pd.DataFrame({
            "Beta co ban":   beta,
            **{S_vi[s]: {j: beta_s[(s,j)] for j in J} for s in S}
        }, index=J).rename(index=J_vi)
        st.dataframe(df_beta, use_container_width=True)

    st.markdown(f"Ngan sach tong: **{BUDGET1+BUDGET2:,} ty VND** "
                f"(First: {BUDGET1:,} | Second: {BUDGET2:,}/kich ban) | "
                f"Phat shortfall nhan luc: {PENALTY}/ty")

    # ------------------------------------------------------------------ Kiem tra Pyomo 
    ok, pyomo_err = check_pyomo()
    if not ok:
        st.error(f" {pyomo_err}")
        st.stop()

    st.markdown("---")
    st.subheader(" Tham so mo hinh")
    penalty_input = st.slider("He so phat shortfall nhan luc (PENALTY)",
                               10, 200, PENALTY, step=10)
    PENALTY = penalty_input

    if st.button(" Chay mo hinh Stochastic Programming (Pyomo + CBC)",
                 type="primary"):

        # === CAU 10.5.1 - SP ===
        with st.spinner("Dang giai mo hinh SP hai giai doan..."):
            Z_sp, x_sp, y_sp, slk, status = build_and_solve_sp(
                J, S, p_prob, beta, beta_s,
                BUDGET1, BUDGET2, floor1, ceil1, target_H, PENALTY
            )

        st.subheader(" Cau 10.5.1 - Ket qua Stochastic Programming")
        col1, col2, col3 = st.columns(3)
        col1.metric("Trang thai solver", status)
        col2.metric("Z* (GDP gain ky vong)", f"{Z_sp:,.2f} ty VND")
        col3.metric("Tong first-stage",
                    f"{sum(x_sp.values()):,.0f} / {BUDGET1:,} ty")

        # Bang first-stage
        df_x = pd.DataFrame({
            "Hang muc":     [J_vi[j] for j in J],
            "Phan bo (ty)": [round(x_sp[j]) for j in J],
            "Ty trong (%)": [round(x_sp[j]/BUDGET1*100, 1) for j in J],
            "San (ty)":     [floor1[j] for j in J],
            "Tran (ty)":    [ceil1[j]  for j in J],
        }).set_index("Hang muc")
        st.markdown("**Quyet dinh First-stage x_j:**")
        st.dataframe(df_x, use_container_width=True)

        # Bang second-stage
        df_y = pd.DataFrame(
            [[round(y_sp[(s,j)]) for j in J]
             + [round(sum(y_sp[(s,j)] for j in J)), round(slk[s])]
             for s in S],
            index=[S_vi[s] for s in S],
            columns=[J_vi[j] for j in J] + ["Tong", "Slack_H"]
        )
        st.markdown("**Quyet dinh Second-stage y_j^s (ty VND):**")
        st.dataframe(df_y, use_container_width=True)

        st.markdown("---")

        # === CAU 10.5.2 - SP vs EV ===
        st.subheader(" Cau 10.5.2 - So sanh SP vs EV (Expected Value)")
        with st.spinner("Dang giai EV..."):
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
            "Hang muc":    [J_vi[j] for j in J],
            "x_SP (ty)":   [round(x_sp[j]) for j in J],
            "x_EV (ty)":   [round(x_ev[j]) for j in J],
            "Chenh lech":  [round(x_sp[j]-x_ev[j]) for j in J],
        }).set_index("Hang muc")
        st.dataframe(df_cmp, use_container_width=True)

        h_diff = x_sp["H"] - x_ev["H"]
        st.info(
            f"SP dau tu H = **{x_sp['H']:,.0f} ty** | EV = **{x_ev['H']:,.0f} ty**  \n"
            f"-> SP dau tu H **{'nhieu hon' if h_diff>0 else 'it hon'}** {abs(h_diff):,.0f} ty "
            f"(tinh den kich ban s3/s4 can nhan luc cao)."
        )

        # Quyet dinh theo tung kich ban
        with st.spinner("Dang giai quyet dinh tung kich ban..."):
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
            columns=["Kich ban","p"] + [J_vi[j] for j in J]
        ).set_index("Kich ban")
        st.markdown("**Quyet dinh toi uu theo tung kich ban rieng le:**")
        st.dataframe(df_det, use_container_width=True)
        st.caption(
            " Kich ban s3/s4 -> dau tu H nhieu hon (nhan luc so la bao hiem khi khung hoang); "
            "s1 -> AI nhieu hon (tan dung he so beta_AI=1.55 cao)."
        )

        st.markdown("---")

        # === CAU 10.5.3 - VSS & EVPI ===
        st.subheader(" Cau 10.5.3 - Tinh VSS va EVPI")
        with st.spinner("Dang tinh WS (Wait-and-See)..."):
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
        col1.metric("WS (Perfect Info)", f"{WS:,.2f} ty")
        col2.metric("SP (Stochastic)",   f"{Z_sp:,.2f} ty")
        col3.metric("EEV (EV applied)",  f"{EEV:,.2f} ty")
        col4.metric("VSS = SP - EEV",    f"{VSS:,.2f} ty",
                    f"{VSS/Z_sp*100:.3f}% Z*")
        col5.metric("EVPI = WS - SP",    f"{EVPI:,.2f} ty",
                    f"{EVPI/Z_sp*100:.3f}% Z*")

        df_sum = pd.DataFrame({
            "Chi so":        ["WS","SP","EEV","VSS","EVPI"],
            "Gia tri (ty)":  [WS, Z_sp, EEV, VSS, EVPI],
            "Y nghia":       [
                "Biet truoc kich ban -> toi uu hoan hao",
                "Toi uu xac suat -> giai phap SP",
                "Dung quyet dinh EV don gian cho thuc te",
                "Loi ich cua tu duy xac suat",
                "Gia tri toi da cua thong tin hoan hao",
            ]
        }).set_index("Chi so")
        st.dataframe(df_sum, use_container_width=True)

        fig_vss = chart_vss_evpi(
            S, S_vi, p_prob, WS_by_s, Z_sp, EEV, WS, VSS, EVPI
        )
        st.pyplot(fig_vss)
        plt.close(fig_vss)

        if EEV <= Z_sp <= WS:
            st.success(f" Bat bien SP thoa man: EEV ({EEV:,.0f}) <= SP ({Z_sp:,.0f}) <= WS ({WS:,.0f})")

        st.markdown("---")

        # === CAU 10.5.4 - ROBUST ===
        st.subheader(" Cau 10.5.4 (Mo rong) - Robust Optimization "
                     "(Minimax Regret)")

        # Regret cua SP
        st.markdown("**Regret cua quyet dinh SP theo tung kich ban:**")
        regret_rows = []
        for s in S:
            Z_actual, _ = solve_scenario_fix(
                s, J, beta, beta_s, BUDGET2, target_H, PENALTY, x_sp
            )
            regret_rows.append({
                "Kich ban": S_vi[s],
                "WS[s] (ty)": round(WS_by_s[s]),
                "Z_actual (ty)": round(Z_actual),
                "Regret (ty)": round(WS_by_s[s]-Z_actual, 2),
            })
        st.dataframe(pd.DataFrame(regret_rows).set_index("Kich ban"),
                     use_container_width=True)

        with st.spinner("Dang giai Minimax Regret..."):
            theta_rob, x_rob = solve_robust_minimax(
                J, S, beta, beta_s, BUDGET1, BUDGET2,
                floor1, ceil1, target_H, PENALTY, WS_by_s
            )

        # Bang so sanh 3 chien luoc
        df_rob = pd.DataFrame({
            "Hang muc":       [J_vi[j] for j in J],
            "x_SP (ty)":      [round(x_sp[j])  for j in J],
            "x_EV (ty)":      [round(x_ev[j])  for j in J],
            "x_Robust (ty)":  [round(x_rob[j]) for j in J],
        }).set_index("Hang muc")
        st.markdown("**So sanh 3 chien luoc quyet dinh:**")
        st.dataframe(df_rob, use_container_width=True)

        max_regret_sp = max(
            WS_by_s[s] - solve_scenario_fix(
                s, J, beta, beta_s, BUDGET2, target_H, PENALTY, x_sp
            )[0]
            for s in S
        )
        col1, col2 = st.columns(2)
        col1.metric("Minimax Regret", f"{theta_rob:,.2f} ty VND")
        col2.metric("Max Regret (SP)", f"{max_regret_sp:,.2f} ty VND")
        st.info(
            f"Robust uu tien H = **{x_rob['H']:,.0f} ty** (bao hiem kich ban xau) "
            f"so voi SP = **{x_sp['H']:,.0f} ty**."
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

        # === TONG KET ===
        st.subheader(" Tong ket Bai 10")
        st.markdown(f"""
        | Chi so | Gia tri (ty VND) | Y nghia |
        |---|---|---|
        | WS | {WS:,.2f} | Biet truoc kich ban -> hoan hao |
        | SP | {Z_sp:,.2f} | Quyet dinh xac suat toi uu |
        | EEV | {EEV:,.2f} | Dung quyet dinh EV don gian |
        | **VSS** | **{VSS:,.2f}** | Loi ich tu duy xac suat |
        | **EVPI** | **{EVPI:,.2f}** | Gia tri thong tin hoan hao |
        """)

        with st.expander(" Thao luan chinh sach"):
            st.markdown(f"""
            **a) SP dau tu H nhieu hon hay it hon loi giai xac dinh?**
            SP dau tu H **nhieu hon** ({x_sp['H']:,.0f} > {x_ev['H']:,.0f} ty).
            Ly do: SP tinh du tail risk kich ban s3/s4 -> nhan luc la "bao hiem kinh te so".

            **b) VSS duong - Y nghia doi voi Viet Nam:**
            VSS = {VSS:,.2f} ty > 0 chung minh tu duy xac suat tao ra gia tri thuc.
            Nen kinh te Viet Nam do mo ~180% GDP -> cuc ky nhay cam voi cu soc ben ngoai.

            **c) COVID-19 & Bao Yagi - Bai hoc ve duoi dau tu nhan luc so:**
            - COVID-19 (2020): GDP chi 2.91%, lao dong so la "escape hatch" (remote work, e-commerce).
            - Bao Yagi (2024): Thiet hai ~40.000 ty, ha tang so phuc hoi nhanh hon.
            - **Khuyen nghi:** Ngan sach dao tao ky nang so nen la "ngan sach khong the cat"
            trong ke hoach 2026-2030.
            """)
if __name__ == "__main__":
    run()