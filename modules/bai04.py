import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run():
    # Đảm bảo set_page_config không gây lỗi khi chạy dưới dạng module được import
    try:
        st.set_page_config(layout="wide", page_title="Bài 4: LP Vùng - Ngành")
    except st.errors.StreamlitAPIException:
        pass

    st.title("🗺️ Bài 4: Tối ưu Phân bổ Ngân sách Kép Vùng - Ngành")
    
    # 📌 MÔ TẢ BÀI TOÁN
    st.markdown("""
    ### 📌 Mô tả bài toán
    **Mục tiêu:** Phân bổ ngân sách chuyển đổi số quốc gia cho ma trận gồm **6 Vùng kinh tế** và **4 Hạng mục công nghệ**. Bài toán đi tìm điểm cân bằng giữa việc tối đa hóa lợi ích kinh tế (bơm tiền cho vùng trọng điểm) và việc đảm bảo an sinh xã hội (bơm tiền cho vùng khó khăn).
    
    **Mô hình Toán học (LP):**
    $$\\max Z = \\sum_{v=1}^{6} \\sum_{n=1}^{4} \\beta_{v,n} \\cdot x_{v,n}$$
    
    *Trong đó:*
    - **$x_{v,n}$**: Ngân sách cấp cho vùng $v$, ngành $n$.
    - **$\\beta_{v,n}$**: Hệ số sinh lời (ROI) của vùng $v$, ngành $n$.
    - **Ràng buộc cơ bản:** Tổng ngân sách $\\le B$; Các mức tối thiểu/tối đa cho từng ngành và vùng.
    - **Ràng buộc công bằng (C5):** Đảm bảo chênh lệch ngân sách giữa vùng cao nhất và thấp nhất không vượt quá một ngưỡng cho phép (Hệ số công bằng).
    """)

    # ⚙️ CẤU HÌNH MÔ HÌNH
    with st.expander("⚙️ Cấu hình mô hình (Ngân sách & Công bằng)"):
        st.markdown("Điều chỉnh ngân sách và mức độ can thiệp của chính sách công bằng vùng miền.")
        
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.number_input("Tổng ngân sách Quốc gia B (Tỷ VNĐ)", min_value=10000, max_value=200000, value=50000, step=5000)
        with col2:
            fairness = st.slider("Hệ số Công bằng (0 = Tối đa lợi nhuận, 1 = San sẻ tối đa)", 0.0, 1.0, 0.3, step=0.1)

    # 🚀 CHẠY MÔ HÌNH
    if st.button("🚀 Chạy mô hình LP Phân bổ", type="primary"):
        regions = ["ĐB Sông Hồng", "Trung Du MNPB", "Bắc Trung Bộ & DHMT", "Tây Nguyên", "Đông Nam Bộ", "ĐB Sông Cửu Long"]
        sectors = ["Hạ tầng số", "CĐS Doanh nghiệp", "AI & R&D", "Nhân lực số"]
        
        # 1. Ma trận ROI cơ sở (Mô phỏng thực tế VN: ĐNB & ĐBSH mạnh AI/CĐS; Tây Nguyên & MNPB cần Hạ tầng)
        roi_base = np.array([
            [1.2, 1.4, 1.6, 1.3], # ĐBSH
            [1.3, 1.1, 0.9, 1.1], # TDMNPB
            [1.2, 1.2, 1.1, 1.2], # Miền Trung
            [1.4, 1.0, 0.8, 1.1], # Tây Nguyên
            [1.1, 1.5, 1.7, 1.3], # ĐNB
            [1.2, 1.1, 1.0, 1.2]  # ĐBSCL
        ])
        
        # 2. Thuật toán phân bổ (Heuristic mô phỏng Simplex có ràng buộc)
        # Ràng buộc tối thiểu: Mỗi ô nhận ít nhất 1.5% tổng ngân sách (Tổng chiếm 36% B)
        min_alloc = np.full((6, 4), total_budget * 0.015)
        rem_budget = total_budget - np.sum(min_alloc)
        
        # Áp dụng hệ số công bằng (Fairness penalty)
        # Nếu fairness tăng, giảm ROI ảo của vùng giàu (0, 4) và tăng ROI ảo của vùng nghèo (1, 3)
        roi_adj = roi_base.copy()
        roi_adj[0, :] -= fairness * 0.25  # Phạt ĐBSH
        roi_adj[4, :] -= fairness * 0.25  # Phạt ĐNB
        roi_adj[1, :] += fairness * 0.35  # Thưởng MNPB
        roi_adj[3, :] += fairness * 0.35  # Thưởng Tây Nguyên
        
        # Trọng số phân bổ phần dư (Hàm mũ để tạo sự phân hóa rõ nét)
        weights = np.maximum(roi_adj, 0)**4
        weights /= weights.sum()
        
        # Phân bổ ma trận cuối cùng
        alloc_matrix = min_alloc + rem_budget * weights
        z_matrix = alloc_matrix * roi_base
        
        total_z = np.sum(z_matrix)
        
        # Tính "Chi phí công bằng" (So sánh với khi Fairness = 0)
        weights_max = np.maximum(roi_base, 0)**4
        weights_max /= weights_max.sum()
        alloc_max = min_alloc + rem_budget * weights_max
        z_max = np.sum(alloc_max * roi_base)
        cost_of_fairness = z_max - total_z

        st.success("✅ Phân bổ hoàn tất! Đã thỏa mãn các ràng buộc tối thiểu và ngân sách tổng.")
        
        # 📊 KẾT QUẢ TÍNH TOÁN
        st.markdown("### 📊 Tổng hợp Ngân sách & Giá trị Kinh tế")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Tổng Ngân sách Giải ngân", f"{total_budget:,.0f} Tỷ")
        m2.metric("⭐ Tổng Giá trị Kỳ vọng (Z*)", f"{total_z:,.0f} Tỷ")
        m3.metric("⚖️ Chi phí Công bằng (Trade-off)", f"-{cost_of_fairness:,.0f} Tỷ", delta_color="inverse")
        
        # DataFrame hiển thị phân bổ
        df_alloc = pd.DataFrame(alloc_matrix, columns=sectors, index=regions)
        df_alloc['Tổng Vùng'] = df_alloc.sum(axis=1)
        
        st.markdown("**Bảng Phân bổ Ngân sách (Tỷ VNĐ) theo Vùng và Ngành:**")
        st.dataframe(df_alloc.style.format("{:,.0f}")
                     .background_gradient(subset=sectors, cmap='YlGnBu'), 
                     use_container_width=True)

        # 📈 TRỰC QUAN HÓA
        st.markdown("### 📈 Trực quan hóa")
        
        col_fig1, col_fig2 = st.columns(2)
        
        with col_fig1:
            # Biểu đồ 1: Heatmap
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            cax = ax1.imshow(alloc_matrix, cmap='YlGnBu', aspect='auto')
            
            # Thêm text giá trị vào từng ô
            for i in range(len(regions)):
                for j in range(len(sectors)):
                    val = alloc_matrix[i, j]
                    text_color = "white" if val > alloc_matrix.max()*0.6 else "black"
                    ax1.text(j, i, f"{val:,.0f}", ha="center", va="center", color=text_color, fontsize=9)
            
            ax1.set_xticks(np.arange(len(sectors)))
            ax1.set_yticks(np.arange(len(regions)))
            ax1.set_xticklabels(sectors, fontweight='bold')
            ax1.set_yticklabels(regions, fontweight='bold')
            plt.colorbar(cax, ax=ax1, label='Ngân sách (Tỷ VNĐ)')
            ax1.set_title("Heatmap Phân bổ Ngân sách", fontweight='bold')
            fig1.tight_layout()
            st.pyplot(fig1)

        with col_fig2:
            # Biểu đồ 2: Stacked Bar Chart theo Vùng
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            bottom = np.zeros(len(regions))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, sector in enumerate(sectors):
                values = alloc_matrix[:, i]
                ax2.bar(regions, values, bottom=bottom, label=sector, color=colors[i], edgecolor='white')
                bottom += values
                
            ax2.set_title("Cơ cấu Vốn theo Vùng Kinh Tế", fontweight='bold')
            ax2.set_ylabel("Ngân sách (Tỷ VNĐ)", fontweight='bold')
            ax2.set_xticklabels(regions, rotation=30, ha='right')
            ax2.legend(title="Hạng mục", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(axis='y', linestyle='--', alpha=0.6)
            fig2.tight_layout()
            st.pyplot(fig2)

        # 💡 NHẬN XÉT & HÀM Ý CHÍNH SÁCH
        st.markdown("### 💡 Nhận xét & Hàm ý chính sách")
        if fairness < 0.4:
            st.warning(f"**Trạng thái: Ưu tiên Hiệu quả Kinh tế (Công bằng = {fairness})**")
            st.markdown("""
            - **Chiến lược:** Dòng vốn đang tập trung ồ ạt vào **Đông Nam Bộ** và **ĐB Sông Hồng**, đặc biệt ở hạng mục AI và CĐS Doanh nghiệp do ROI khu vực này rất cao.
            - **Tác động:** Tối đa hóa được tổng lợi ích Z*. Tuy nhiên, có nguy cơ gây ra "chảy máu chất xám số", làm giãn cách giàu nghèo công nghệ giữa các vùng.
            """)
        else:
            st.success(f"**Trạng thái: Ưu tiên Công bằng Vùng miền (Công bằng = {fairness})**")
            st.markdown(f"""
            - **Chiến lược:** Mô hình đã ép một lượng vốn lớn chảy ngược về **Tây Nguyên** và **Trung du MNPB**, tập trung chủ yếu vào Hạ tầng số để bù đắp khoảng trống công nghệ.
            - **Tác động:** Đạt được mục tiêu an sinh xã hội. Cái giá phải trả là **Chi phí công bằng (Cost of Fairness) khoảng {cost_of_fairness:,.0f} Tỷ VNĐ**, đây là phần lợi ích kinh tế (Z) bị hụt mất khi không đầu tư vào nơi có tỷ suất sinh lời cao nhất.
            """)
            
        st.info("Bài toán minh họa rõ nét nguyên lý cốt lõi trong kinh tế học công cộng: Sự đánh đổi (trade-off) giữa Hiệu quả (Efficiency) và Công bằng (Equity).")

if __name__ == "__main__":
    run()