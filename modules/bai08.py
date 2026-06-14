import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# CÁC HÀM XỬ LÝ (FUNCTIONS)
# ==========================================
def simulate(A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, g_D, g_AI, g_H, alpha, beta, gamma, delta, theta, years, start_year=2026):
    """Mô phỏng các biến số và tính GDP theo thời gian."""
    data = []
    for t in range(years):
        A = A0 * ((1 + g_A) ** t)
        K = K0 * ((1 + g_K) ** t)
        L = L0 * ((1 + g_L) ** t)
        D = D0 * ((1 + g_D) ** t)
        AI = AI0 * ((1 + g_AI) ** t)
        H = H0 * ((1 + g_H) ** t)
        Y = A * (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
        data.append({
            'Năm': start_year + t, 
            'Y (GDP)': Y, 
            'A (TFP)': A, 
            'K (Vốn)': K, 
            'L (Lao động)': L, 
            'D (Số hóa)': D, 
            'AI (Trí tuệ NT)': AI, 
            'H (Nhân lực)': H
        })
    return pd.DataFrame(data)

def optimize_scenario(alpha, beta, gamma, delta, theta, A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, g_D_base, g_AI_base, g_H_base, extra_budget, years):
    """Tối ưu hóa phân bổ ngân sách tăng trưởng phụ trợ vào D, AI, H để tối đa hóa GDP cuối kỳ."""
    def objective(x):
        # x[0]: extra g_D, x[1]: extra g_AI, x[2]: extra g_H
        df = simulate(A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, 
                      g_D_base + x[0], g_AI_base + x[1], g_H_base + x[2], 
                      alpha, beta, gamma, delta, theta, years)
        return -df['Y (GDP)'].iloc[-1] # Tối thiểu hóa giá trị âm của GDP = Tối đa hóa GDP

    # Ràng buộc: Tổng ngân sách = extra_budget
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - extra_budget})
    bounds = ((0, extra_budget), (0, extra_budget), (0, extra_budget))
    init_guess = [extra_budget/3, extra_budget/3, extra_budget/3]

    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def calculate_growth_contribution(alpha, beta, gamma, delta, theta, g_A, g_K, g_L, g_D, g_AI, g_H):
    """Tính toán phần trăm đóng góp của các yếu tố vào tốc độ tăng trưởng chung."""
    total_growth = g_A + alpha*g_K + beta*g_L + gamma*g_D + delta*g_AI + theta*g_H
    contributions = {
        'TFP (A)': (g_A / total_growth) * 100,
        'Vốn (K)': ((alpha * g_K) / total_growth) * 100,
        'Lao động (L)': ((beta * g_L) / total_growth) * 100,
        'Số hóa (D)': ((gamma * g_D) / total_growth) * 100,
        'AI': ((delta * g_AI) / total_growth) * 100,
        'Nhân lực (H)': ((theta * g_H) / total_growth) * 100,
    }
    return contributions

# ==========================================
# GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
def run():
    try:
        st.set_page_config(page_title="Tối ưu Động Phân Bổ", layout="wide")
    except:
        pass

    st.title("📊 Bài 8 - Tối Ưu Động Phân Bổ Liên Thời Gian 2026–2035")
    st.markdown("""
    **Mô hình Cobb-Douglas mở rộng:** $Y_t = A_t \\cdot K_t^{\\alpha} \\cdot L_t^{\\beta} \\cdot D_t^{\\gamma} \\cdot AI_t^{\\delta} \\cdot H_t^{\\theta}$
    """)

    # 1. Sidebar - Input Parameters
    with st.sidebar:
        st.header("⚙️ Tham số Đầu vào")
        
        st.subheader("Hệ số co giãn (Elasticities)")
        alpha = st.slider("α (Vốn - K)", 0.0, 1.0, 0.33, 0.01)
        beta = st.slider("β (Lao động - L)", 0.0, 1.0, 0.42, 0.01)
        gamma = st.slider("γ (Số hóa - D)", 0.0, 1.0, 0.10, 0.01)
        delta = st.slider("δ (AI)", 0.0, 1.0, 0.08, 0.01)
        theta = st.slider("θ (Nhân lực - H)", 0.0, 1.0, 0.07, 0.01)
        
        sum_elasticity = alpha + beta + gamma + delta + theta
        st.caption(f"*Tổng hệ số hiện tại: {sum_elasticity:.2f}*")
        
        st.subheader("Tốc độ tăng trưởng hàng năm (%)")
        g_A = st.number_input("TFP (A)", value=2.0, step=0.1) / 100
        g_K = st.number_input("Vốn (K)", value=6.0, step=0.1) / 100
        g_L = st.number_input("Lao động (L)", value=1.5, step=0.1) / 100
        g_D = st.number_input("Số hóa (D)", value=10.0, step=0.1) / 100
        g_AI = st.number_input("AI", value=15.0, step=0.1) / 100
        g_H = st.number_input("Nhân lực (H)", value=5.0, step=0.1) / 100
        
        st.subheader("Khung thời gian")
        start_year = st.number_input("Năm bắt đầu", value=2026, step=1)
        end_year = st.number_input("Năm kết thúc", value=2035, step=1)
        years = end_year - start_year + 1

    # Base values (Giả định quy mô ban đầu = 100)
    A0, K0, L0, D0, AI0, H0 = 1.0, 100, 100, 100, 100, 100

    # 2. Dynamic Simulation Engine (Chạy các kịch bản)
    df_base = simulate(A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, g_D, g_AI, g_H, alpha, beta, gamma, delta, theta, years, start_year)
    df_high_ai = simulate(A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, g_D, g_AI + 0.05, g_H, alpha, beta, gamma, delta, theta, years, start_year)
    df_high_digital = simulate(A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, g_D + 0.05, g_AI, g_H, alpha, beta, gamma, delta, theta, years, start_year)

    df_compare = pd.DataFrame({
        'Năm': df_base['Năm'],
        'Baseline': df_base['Y (GDP)'],
        'High AI (+5%)': df_high_ai['Y (GDP)'],
        'High Digital (+5%)': df_high_digital['Y (GDP)']
    })

    # ==========================================
    # 3. TỐI ƯU HÓA (Optimization Module)
    # ==========================================
    st.header("🎯 Kịch Bản Tối Ưu Hóa Chính Sách")
    st.markdown("Giả sử Chính phủ có **dư địa ngân sách để tăng thêm 5% tốc độ tăng trưởng** và cần phân bổ vào **Số hóa (D)**, **AI**, hoặc **Nhân lực (H)**. Mô hình tìm tỷ lệ phân bổ giúp **tối đa hóa GDP năm kết thúc**.")

    extra_budget = 0.05 
    opt_alloc = optimize_scenario(alpha, beta, gamma, delta, theta, A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, g_D, g_AI, g_H, extra_budget, years)

    df_opt = simulate(A0, K0, L0, D0, AI0, H0, g_A, g_K, g_L, 
                      g_D + opt_alloc[0], g_AI + opt_alloc[1], g_H + opt_alloc[2], 
                      alpha, beta, gamma, delta, theta, years, start_year)
    df_compare['Optimized Allocation'] = df_opt['Y (GDP)']

    col1, col2, col3 = st.columns(3)
    col1.metric("Bổ sung Tăng trưởng Số hóa (D)", f"+{opt_alloc[0]*100:.2f}%")
    col2.metric("Bổ sung Tăng trưởng AI", f"+{opt_alloc[1]*100:.2f}%")
    col3.metric("Bổ sung Tăng trưởng Nhân lực (H)", f"+{opt_alloc[2]*100:.2f}%")

    # ==========================================
    # 4. VISUALIZATION (Bằng Matplotlib)
    # ==========================================
    st.markdown("### 📈 Trực quan hóa Dữ liệu")
    tab1, tab2, tab3 = st.tabs(["So sánh Kịch bản GDP", "Xu hướng Các Yếu tố", "Phân rã Đóng góp"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, col in enumerate(df_compare.columns[1:]):
            ax1.plot(df_compare['Năm'], df_compare[col], marker='o', linewidth=2, label=col, color=colors[idx])
        ax1.set_title("Dự phóng GDP (Chỉ số) theo các Kịch bản", fontweight='bold')
        ax1.set_xlabel("Năm")
        ax1.set_ylabel("Quy mô GDP (Base=100)")
        ax1.set_xticks(df_compare['Năm'])
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        st.pyplot(fig1)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        factors = ['K (Vốn)', 'L (Lao động)', 'D (Số hóa)', 'AI (Trí tuệ NT)', 'H (Nhân lực)']
        for f in factors:
            ax2.plot(df_base['Năm'], df_base[f], marker='s', label=f)
        ax2.set_title("Sự phát triển của các Yếu tố Đầu vào (Kịch bản Baseline)", fontweight='bold')
        ax2.set_xlabel("Năm")
        ax2.set_ylabel("Giá trị Chỉ số")
        ax2.set_xticks(df_base['Năm'])
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

    with tab3:
        contrib = calculate_growth_contribution(alpha, beta, gamma, delta, theta, g_A, g_K, g_L, g_D, g_AI, g_H)
        df_contrib = pd.DataFrame(list(contrib.items()), columns=['Yếu tố', 'Đóng góp (%)']).sort_values('Đóng góp (%)', ascending=False)
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bars = ax3.bar(df_contrib['Yếu tố'], df_contrib['Đóng góp (%)'], color='teal', edgecolor='black', alpha=0.8)
        ax3.set_title("Tỷ trọng Đóng góp vào Tăng trưởng GDP (Baseline)", fontweight='bold')
        ax3.set_ylabel("Đóng góp (%)")
        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', fontweight='bold')
            
        st.pyplot(fig3)

    # ==========================================
    # 5. POLICY INSIGHTS PANEL
    # ==========================================
    st.header("💡 Policy Insights & Nhận Xét")

    best_scenario = df_compare.columns[1:][np.argmax(df_compare.iloc[-1, 1:].values)]
    max_contrib_factor = df_contrib.iloc[0]['Yếu tố']
    ai_roi = df_high_ai['Y (GDP)'].iloc[-1] - df_base['Y (GDP)'].iloc[-1]
    dig_roi = df_high_digital['Y (GDP)'].iloc[-1] - df_base['Y (GDP)'].iloc[-1]

    st.info(f"""
    **Báo cáo Tóm tắt Chính sách (Policy Brief):**
    1. **Động lực chính:** Dựa trên các tham số hiện tại, yếu tố đóng góp lớn nhất vào tăng trưởng chung là **{max_contrib_factor}**. Chính phủ cần ưu tiên duy trì tốc độ phát triển của động lực lõi này.
    2. **Kịch bản Tối ưu:** Trong số các kịch bản được mô phỏng, **{best_scenario}** mang lại quy mô kinh tế lớn nhất vào năm kết thúc (đạt mức {df_compare[best_scenario].iloc[-1]:.2f} so với mức cơ sở).
    3. **Hiệu quả biên (Marginal Return):** - Tăng thêm 5% nguồn lực vào AI mang lại thêm **{ai_roi:.2f}** điểm chỉ số GDP.
       - Tăng thêm 5% nguồn lực vào Số hóa mang lại thêm **{dig_roi:.2f}** điểm chỉ số GDP.
    4. **Khuyến nghị Phân bổ:** Module phân bổ động khuyến nghị dồn ngân sách phụ trợ theo tỷ lệ `{opt_alloc[0]*100:.1f}% Số hóa` : `{opt_alloc[1]*100:.1f}% AI` : `{opt_alloc[2]*100:.1f}% Nhân lực`. Do hàm Cobb-Douglas có đặc tính sinh lời biên giảm dần, thuật toán sẽ tự động nghiêng về yếu tố có hệ số độ co giãn cao mà hiện tại đang bị "bỏ đói" nguồn lực.
    """)

if __name__ == "__main__":
    run()