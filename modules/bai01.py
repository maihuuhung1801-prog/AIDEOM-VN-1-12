import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.title("📘 Bài 1: Hàm Sản Xuất Cobb-Douglas Mở Rộng với AI và Số Hóa")
    
    st.markdown("""
    **Mục tiêu:** Ước lượng Năng suất Nhân tố Tổng hợp (TFP) từ dữ liệu mô phỏng kinh tế vĩ mô Việt Nam giai đoạn 2020-2025 và đánh giá tác động của các yếu tố đầu vào (Vốn, Lao động, Số hóa, AI, Nhân lực số) đến tổng sản phẩm quốc nội (GDP).
    
    **Mô hình Toán học:**
    $$Y_t = A_t \\cdot K_t^{\\alpha} \\cdot L_t^{\\beta} \\cdot D_t^{\\gamma} \\cdot AI_t^{\\delta} \\cdot H_t^{\\theta}$$
    
    *Trong đó:*
    - **Y:** GDP (sản lượng)
    - **A:** TFP (Năng suất nhân tố tổng hợp)
    - **K:** Vốn vật chất | **L:** Lao động | **D:** Tỷ trọng kinh tế số/GDP 
    - **AI:** Số DN công nghệ số | **H:** Tỷ lệ LĐ qua đào tạo
    
    *Điều kiện lợi suất không đổi theo quy mô (CRS):* $\\alpha + \\beta + \\gamma + \\delta + \\theta = 1$
    """)
    
    st.subheader("⚙️ Tham số Đầu vào (Hệ số độ co giãn Cobb-Douglas)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        alpha = st.number_input("α (Vốn K)", value=0.33, step=0.01, format="%.2f")
    with col2:
        beta = st.number_input("β (Lao động L)", value=0.42, step=0.01, format="%.2f")
    with col3:
        gamma = st.number_input("γ (Số hóa D)", value=0.10, step=0.01, format="%.2f")
    with col4:
        delta = st.number_input("δ (AI)", value=0.08, step=0.01, format="%.2f")
    with col5:
        theta = st.number_input("θ (Nhân lực H)", value=0.07, step=0.01, format="%.2f")

    # Kiểm tra điều kiện CRS
    total_coef = alpha + beta + gamma + delta + theta
    if not np.isclose(total_coef, 1.0):
        st.warning(f"⚠️ Tổng các hệ số hiện tại là {total_coef:.2f}. Theo lý thuyết Cobb-Douglas chuẩn, khuyến nghị tổng = 1 (Điều kiện CRS).")
    else:
        st.success("✅ Tổng các hệ số = 1 (Thỏa mãn điều kiện lợi suất không đổi theo quy mô).")

    if st.button("🚀 Chạy mô hình ước lượng TFP", type="primary"):
        # Dữ liệu mô phỏng dựa trên cấu trúc bảng 1.3 của notebook
        data = {
            'Năm': [2020, 2021, 2022, 2023, 2024, 2025],
            'Y (GDP)': [8000, 8400, 9500, 10200, 11000, 11800],
            'K (Vốn)': [16500, 17800, 19600, 21300, 23500, 25900],
            'L (Lao động)': [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
            'D (Số hóa %)': [9.6, 11.9, 14.2, 16.5, 18.0, 20.0],
            'AI (DN nghìn)': [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
            'H (Nhân lực %)': [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
        }
        df = pd.DataFrame(data)
        
        # Tính toán TFP (A_t) bằng cách giải ngược phương trình
        K_alpha = df['K (Vốn)'] ** alpha
        L_beta = df['L (Lao động)'] ** beta
        D_gamma = df['D (Số hóa %)'] ** gamma
        AI_delta = df['AI (DN nghìn)'] ** delta
        H_theta = df['H (Nhân lực %)'] ** theta
        
        factor = K_alpha * L_beta * D_gamma * AI_delta * H_theta
        df['TFP (A_t)'] = df['Y (GDP)'] / factor
        
        # 1. Bảng dữ liệu kết quả
        st.subheader("📊 Bảng dữ liệu và Kết quả TFP ($A_t$)")
        st.dataframe(df.style.format({
            "Y (GDP)": "{:,.0f}",
            "K (Vốn)": "{:,.0f}",
            "L (Lao động)": "{:.1f}",
            "D (Số hóa %)": "{:.1f}",
            "AI (DN nghìn)": "{:.1f}",
            "H (Nhân lực %)": "{:.1f}",
            "TFP (A_t)": "{:.4f}"
        }), use_container_width=True)
        
        # 2. Biểu đồ trực quan
        st.subheader("📈 Trực quan hóa Xu hướng GDP và TFP")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Trục y thứ nhất (GDP)
        color1 = '#1f77b4'
        ax1.set_xlabel('Năm', fontweight='bold')
        ax1.set_ylabel('GDP (Nghìn tỷ VND)', color=color1, fontweight='bold')
        ax1.plot(df['Năm'], df['Y (GDP)'], marker='o', color=color1, linewidth=2, label='GDP Thực tế')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(df['Năm'])
        
        # Trục y thứ hai (TFP)
        ax2 = ax1.twinx()
        color2 = '#d62728'
        ax2.set_ylabel('TFP ($A_t$)', color=color2, fontweight='bold')
        ax2.plot(df['Năm'], df['TFP (A_t)'], marker='s', color=color2, linewidth=2, linestyle='--', label='TFP (A_t)')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # 3. Nhận xét phân tích
        tfp_growth = (df['TFP (A_t)'].iloc[-1] / df['TFP (A_t)'].iloc[0] - 1) * 100
        mean_tfp = df['TFP (A_t)'].mean()
        
        st.subheader("💡 Nhận xét chính sách & Ý nghĩa kinh tế")
        st.info(f"""
        - **Xu hướng TFP:** Giá trị TFP trung bình đạt **{mean_tfp:.4f}**, với mức thay đổi **{tfp_growth:+.2f}%** trong giai đoạn 2020-2025.
        - **Ý nghĩa chỉ số:** TFP (Năng suất nhân tố tổng hợp) phản ánh phần tăng trưởng GDP không đến từ sự gia tăng đơn thuần của số lượng các yếu tố đầu vào vật chất (Vốn, Lao động). Nó đại diện cho sự hiệu quả, tiến bộ công nghệ, năng lực quản lý và tác động của môi trường vĩ mô (như độ trễ do Covid-19 vào năm 2021).
        - **Khuyến nghị chiến lược:** Nếu TFP có dấu hiệu chững lại, việc tăng trưởng dựa vào thâm dụng vốn (K) sẽ gặp hiệu ứng năng suất biên giảm dần. Các nhà hoạch định chính sách cần tập trung giải ngân vào nâng cao chỉ số số hóa (D) và thúc đẩy ứng dụng AI để tối ưu hoá $A_t$ thay vì chỉ bơm tiền vào thị trường.
        """)

if __name__ == "__main__":
    run()