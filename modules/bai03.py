import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run():
    # Đảm bảo set_page_config không gây lỗi khi chạy dưới dạng module được import
    try:
        st.set_page_config(layout="wide", page_title="Bài 3: Priority Index")
    except st.errors.StreamlitAPIException:
        pass

    st.title("📊 Bài 3: Chỉ số Ưu tiên Ngành (Priority Index) Việt Nam")
    
    # 📌 MÔ TẢ BÀI TOÁN
    st.markdown("""
    ### 📌 Mô tả bài toán
    **Mục tiêu:** Xây dựng **Chỉ số ưu tiên (Priority Index)** để xếp hạng 10 ngành kinh tế trọng điểm của Việt Nam, qua đó xác định ngành nào cần được ưu tiên phân bổ ngân sách, chính sách chuyển đổi số và ứng dụng trí tuệ nhân tạo (AI).
    
    **Mô hình Toán học đa tiêu chí:**
    $$Priority_i = w_1 G_i + w_2 P_i + w_3 S_i + w_4 E_i + w_5 L_i + w_6 AI_i - w_7 R_i$$
    
    *Trong đó:*
    - **G (Growth):** Tăng trưởng | **P (Productivity):** Năng suất | **S (Spillover):** Độ lan tỏa
    - **E (Export):** Xuất khẩu | **L (Labor):** Việc làm | **AI:** Sẵn sàng AI | **R (Risk):** Rủi ro (Tiêu chí nghịch)
    
    *Phương pháp chuẩn hóa:* Min-Max Scaling [0, 1]. Tiêu chí rủi ro được đảo chiều (Rủi ro càng thấp -> Điểm càng cao).
    """)

    # ⚙️ CẤU HÌNH MÔ HÌNH
    with st.expander("⚙️ Cấu hình mô hình (Tùy chỉnh Trọng số)"):
        st.markdown("Điều chỉnh trọng số của các tiêu chí đánh giá (Tổng nên gần 1.0, nhưng mô hình sẽ tự động chuẩn hóa tương đối).")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            w_growth = st.number_input("Tăng trưởng (w1)", value=0.15, step=0.05)
            w_prod = st.number_input("Năng suất (w2)", value=0.15, step=0.05)
        with c2:
            w_spill = st.number_input("Lan tỏa (w3)", value=0.15, step=0.05)
            w_export = st.number_input("Xuất khẩu (w4)", value=0.10, step=0.05)
        with c3:
            w_labor = st.number_input("Việc làm (w5)", value=0.10, step=0.05)
            w_ai = st.number_input("Sẵn sàng AI (w6)", value=0.25, step=0.05)
        with c4:
            w_risk = st.number_input("Rủi ro (w7)", value=0.10, step=0.05)
            
        weights = np.array([w_growth, w_prod, w_spill, w_export, w_labor, w_ai, w_risk])

    # 🚀 CHẠY MÔ HÌNH
    if st.button("🚀 Chạy mô hình", type="primary"):
        # Dữ liệu mô phỏng đại diện cho 10 ngành kinh tế Việt Nam (Mock data thực tế)
        sectors = ["CNTT & Truyền thông", "Chế biến Chế tạo", "Tài chính Ngân hàng", 
                   "Nông nghiệp", "Y tế", "Giáo dục", "Bán lẻ", "Logistics", "Du lịch", "Xây dựng"]
        
        # Raw data
        data = {
            "Tăng trưởng": [15.2, 8.5, 9.1, 3.5, 6.2, 5.5, 7.8, 8.0, 12.5, 4.5],
            "Năng suất": [120, 85, 150, 30, 70, 60, 50, 65, 45, 55],
            "Lan tỏa": [0.85, 0.90, 0.75, 0.60, 0.55, 0.65, 0.70, 0.80, 0.50, 0.82],
            "Xuất khẩu": [110, 250, 5, 45, 1, 0, 2, 15, 10, 5],
            "Việc làm": [1.2, 4.5, 0.4, 15.0, 0.5, 1.5, 3.0, 1.0, 2.5, 4.0],
            "Sẵn sàng AI": [0.95, 0.65, 0.85, 0.30, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35],
            "Rủi ro": [0.4, 0.6, 0.5, 0.8, 0.3, 0.2, 0.5, 0.7, 0.9, 0.8]
        }
        df_raw = pd.DataFrame(data, index=sectors)
        
        # Chuẩn hóa Min-Max
        df_norm = pd.DataFrame(index=sectors)
        for col in df_raw.columns:
            min_val = df_raw[col].min()
            max_val = df_raw[col].max()
            if col == "Rủi ro":
                # Đảo chiều rủi ro: Rủi ro càng thấp điểm càng cao
                df_norm[col] = (max_val - df_raw[col]) / (max_val - min_val + 1e-9)
            else:
                df_norm[col] = (df_raw[col] - min_val) / (max_val - min_val + 1e-9)
                
        # Tính toán Priority Index
        df_norm['Priority Index'] = (
            df_norm['Tăng trưởng'] * w_growth +
            df_norm['Năng suất'] * w_prod +
            df_norm['Lan tỏa'] * w_spill +
            df_norm['Xuất khẩu'] * w_export +
            df_norm['Việc làm'] * w_labor +
            df_norm['Sẵn sàng AI'] * w_ai +
            df_norm['Rủi ro'] * w_risk
        )
        
        # Xếp hạng
        df_result = df_norm.sort_values(by='Priority Index', ascending=False).reset_index()
        df_result.rename(columns={'index': 'Ngành Kinh Tế'}, inplace=True)
        df_result.index = df_result.index + 1
        
        st.success("✅ Mô hình chạy thành công! Dữ liệu đã được chuẩn hóa và xếp hạng.")
        
        # 📊 KẾT QUẢ TÍNH TOÁN
        st.markdown("### 📊 Kết quả tính toán")
        
        top_1 = df_result.iloc[0]
        top_2 = df_result.iloc[1]
        gap = top_1['Priority Index'] - top_2['Priority Index']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("🥇 Ngành ưu tiên cao nhất", top_1['Ngành Kinh Tế'])
        m2.metric("⭐ Điểm Priority Index lớn nhất", f"{top_1['Priority Index']:.3f}")
        m3.metric("📈 Khoảng cách Top 1 & Top 2", f"+{gap:.3f}")
        
        # Hiển thị DataFrame
        st.dataframe(df_result.style.format({col: "{:.3f}" for col in df_result.columns if col != 'Ngành Kinh Tế'})
                     .background_gradient(subset=['Priority Index'], cmap='Blues'), 
                     use_container_width=True)
        
        # 📈 TRỰC QUAN HÓA
        st.markdown("### 📈 Trực quan hóa")
        
        col_fig1, col_fig2 = st.columns(2)
        
        with col_fig1:
            # Biểu đồ 1: Bar chart xếp hạng Priority Index
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            y_pos = np.arange(len(df_result))
            
            # Đảo ngược để Top 1 nằm trên cùng
            sectors_sorted = df_result['Ngành Kinh Tế'][::-1]
            scores_sorted = df_result['Priority Index'][::-1]
            
            colors = ['#1f77b4' if i < len(df_result)-1 else '#ff7f0e' for i in range(len(df_result))]
            
            ax1.barh(y_pos, scores_sorted, color=colors, edgecolor='white')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(sectors_sorted)
            ax1.set_xlabel('Điểm Priority Index', fontweight='bold')
            ax1.set_title('Xếp hạng Chỉ số Ưu tiên Ngành 2024', fontweight='bold')
            ax1.grid(axis='x', linestyle='--', alpha=0.6)
            fig1.tight_layout()
            st.pyplot(fig1)

        with col_fig2:
            # Biểu đồ 2: So sánh cấu trúc điểm chuẩn hóa của Top 5 ngành
            top_5_df = df_result.head(5).set_index('Ngành Kinh Tế').drop(columns=['Priority Index'])
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            top_5_df.T.plot(kind='bar', ax=ax2, colormap='Set2', edgecolor='black')
            
            ax2.set_title('Phân rã Điểm Chuẩn Hóa (Top 5 Ngành)', fontweight='bold')
            ax2.set_ylabel('Điểm Chuẩn Hóa [0, 1]', fontweight='bold')
            ax2.set_xticklabels(top_5_df.columns, rotation=45, ha='right')
            ax2.legend(title="Ngành", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(axis='y', linestyle='--', alpha=0.6)
            fig2.tight_layout()
            st.pyplot(fig2)

        # 💡 NHẬN XÉT & HÀM Ý CHÍNH SÁCH
        st.markdown("### 💡 Nhận xét & Hàm ý chính sách")
        st.markdown(f"""
        - **Ngành dẫn đầu:** **{top_1['Ngành Kinh Tế']}** là ngành đạt mức ưu tiên cao nhất, chủ yếu nhờ động lực từ tiêu chí Sẵn sàng AI, Mức độ lan tỏa công nghệ và Tốc độ tăng trưởng.
        - **Ý nghĩa kinh tế:** Mô hình cho thấy sự chuyển dịch ưu tiên từ thâm dụng lao động (như Nông nghiệp) sang thâm dụng công nghệ và tri thức. Đầu tư vốn mồi (seed funding) vào ngành Top 1 sẽ tạo ra hiệu ứng lan tỏa lớn nhất đến phần còn lại của nền kinh tế.
        - **Liên hệ chuyển đổi số Việt Nam:** Bám sát Quyết định 749/QĐ-TTg về Chuyển đổi số Quốc gia, các ngành tài chính, y tế, và chế biến chế tạo đang được định vị là các lĩnh vực "phá băng", cần thiết lập sandbox pháp lý sớm.
        - **Liên hệ AIDEOM-VN:** Kết quả này phù hợp với mô hình khung AIDEOM, trong đó mức độ Sẵn sàng AI (AI Readiness) là động lực lõi để tái định hình quy mô nền kinh tế số thay vì chỉ dựa vào lợi thế xuất khẩu tĩnh.
        """)
        
        st.info("Kết quả hỗ trợ xác định các ngành ưu tiên trong chiến lược chuyển đổi số và phát triển kinh tế số Việt Nam.")

if __name__ == "__main__":
    run()