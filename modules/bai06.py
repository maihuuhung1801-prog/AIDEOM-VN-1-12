import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. HÀM TẢI DỮ LIỆU (Load Data)
# ==========================================
@st.cache_data
def load_data():
    """
    Tái tạo bộ dữ liệu đánh giá 6 Vùng kinh tế trọng điểm dựa trên cấu trúc bài học.
    Ghi chú:
    - C1 (Hạ tầng số), C2 (Nhân lực công nghệ), C3 (Năng lực R&D) là tiêu chí thuận (càng lớn càng tốt).
    - C4 (Rào cản/Rủi ro) là tiêu chí nghịch (càng nhỏ càng tốt).
    """
    data = {
        "Vùng Kinh Tế": [
            "Đông Nam Bộ", 
            "Đồng Bằng Sông Hồng", 
            "Bắc Trung Bộ & DHMT", 
            "Đồng Bằng Sông Cửu Long", 
            "Trung Du MNPB", 
            "Tây Nguyên"
        ],
        "Hạ tầng số (C1)": [95.0, 90.0, 65.0, 60.0, 45.0, 40.0],
        "Nhân lực số (C2)": [90.0, 92.0, 55.0, 50.0, 35.0, 30.0],
        "Năng lực R&D (C3)": [85.0, 88.0, 45.0, 35.0, 25.0, 20.0],
        "Rào cản (C4)": [20.0, 25.0, 50.0, 60.0, 75.0, 80.0]
    }
    df = pd.DataFrame(data)
    
    # Xác định loại tiêu chí: 1 là Thuận (Benefit), -1 là Nghịch (Cost)
    criteria_types = [1, 1, 1, -1] 
    return df, criteria_types

# ==========================================
# 2. HÀM CHẠY MÔ HÌNH TOPSIS (Run Model)
# ==========================================
def run_model(df, weights, criteria_types):
    # Trích xuất ma trận giá trị X (bỏ cột đầu tiên là Tên Vùng)
    X = df.iloc[:, 1:].values.astype(float)
    
    # Bước 1: Chuẩn hóa vector (Vector Normalization)
    norm_X = X / np.sqrt((X**2).sum(axis=0))
    
    # Bước 2: Ma trận có trọng số V
    weights = np.array(weights) / np.sum(weights) # Đảm bảo tổng trọng số = 1
    V = norm_X * weights
    
    # Bước 3: Xác định giải pháp lý tưởng dương (A*) và lý tưởng âm (A-)
    A_star = np.zeros(X.shape[1])
    A_minus = np.zeros(X.shape[1])
    
    for j in range(X.shape[1]):
        if criteria_types[j] == 1: # Tiêu chí Thuận
            A_star[j] = np.max(V[:, j])
            A_minus[j] = np.min(V[:, j])
        else:                      # Tiêu chí Nghịch
            A_star[j] = np.min(V[:, j])
            A_minus[j] = np.max(V[:, j])
            
    # Bước 4: Tính khoảng cách Euclide tới A* và A-
    S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))
    
    # Bước 5: Tính Hệ số gần gũi (C*)
    C_star = S_minus / (S_star + S_minus)
    
    # Gắn kết quả trở lại DataFrame
    df_result = df.copy()
    df_result["Khoảng cách S*"] = S_star
    df_result["Khoảng cách S-"] = S_minus
    df_result["Điểm TOPSIS (C*)"] = C_star
    
    # Xếp hạng giảm dần theo C*
    df_result = df_result.sort_values(by="Điểm TOPSIS (C*)", ascending=False).reset_index(drop=True)
    df_result["Thứ hạng"] = df_result.index + 1
    
    return df_result

# ==========================================
# 3. HÀM TẠO BIỂU ĐỒ (Create Charts)
# ==========================================
def create_charts(df_result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biểu đồ 1: Bar chart xếp hạng Điểm TOPSIS (C*)
    regions = df_result["Vùng Kinh Tế"]
    scores = df_result["Điểm TOPSIS (C*)"]
    
    # Phân biệt màu sắc: Top 1 màu đỏ, còn lại màu xanh
    colors = ['#d62728' if i == 0 else '#1f77b4' for i in range(len(regions))]
    
    bars = ax1.bar(regions, scores, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_title("Xếp hạng Điểm TOPSIS (C*) 6 Vùng Kinh Tế", fontweight='bold')
    ax1.set_ylabel("Hệ số gần gũi lý tưởng (C*)")
    ax1.set_ylim(0, 1.1) # Điểm tối đa luôn <= 1
    ax1.set_xticklabels(regions, rotation=30, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Thêm giá trị trực tiếp trên đầu mỗi cột
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', fontweight='bold')

    # Biểu đồ 2: So sánh thành phần tiêu chí (Đã chuẩn hóa Min-Max để trực quan hóa)
    raw_criteria = df_result.iloc[:, 1:5]
    # Chuẩn hóa Min-Max để đưa tất cả các thang đo về [0, 1] trên đồ thị
    norm_criteria = (raw_criteria - raw_criteria.min()) / (raw_criteria.max() - raw_criteria.min())
    norm_criteria.index = df_result["Vùng Kinh Tế"]
    
    norm_criteria.plot(kind='bar', ax=ax2, colormap='Spectral', edgecolor='black', alpha=0.9)
    
    ax2.set_title("So sánh Cấu trúc Các Tiêu Chí (Min-Max Scaled)", fontweight='bold')
    ax2.set_ylabel("Mức độ tương đối (0-1)")
    ax2.set_xticklabels(norm_criteria.index, rotation=30, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    return fig

# ==========================================
# 4. HÀM GIAO DIỆN CHÍNH (Run)
# ==========================================
def run():
    try:
        st.set_page_config(layout="wide", page_title="Bài 6: TOPSIS Vùng Kinh Tế")
    except:
        pass

    st.title("🏆 Bài 6: TOPSIS - Xếp hạng Ưu tiên Đầu tư AI theo Vùng")
    
    # 📌 MÔ TẢ BÀI TOÁN
    st.markdown("""
    ### 📌 Mô tả bài toán
    **Mục tiêu:** Ứng dụng phương pháp **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** để đánh giá đa tiêu chí và xếp hạng năng lực chuyển đổi số/AI của 6 Vùng Kinh tế Việt Nam. Qua đó giúp nhà nước ưu tiên phân bổ nguồn lực.
    
    **Quy trình Toán học (5 Bước):**
    1. Chuẩn hóa vector ma trận quyết định ($r_{ij}$).
    2. Nhân ma trận với bộ trọng số ($v_{ij}$).
    3. Tìm Giải pháp lý tưởng dương ($A^*$) và lý tưởng âm ($A^-$).
    4. Đo khoảng cách Euclide từ mỗi vùng tới $A^*$ và $A^-$.
    5. Tính Hệ số gần gũi ($C^*$). Vùng có $C^*$ gần 1 nhất sẽ đứng đầu.
    """)

    # Tải dữ liệu
    df_raw, criteria_types = load_data()

    # ⚙️ CẤU HÌNH MÔ HÌNH
    with st.expander("⚙️ Cấu hình mô hình (Tùy chỉnh Trọng số)", expanded=True):
        st.markdown("Điều chỉnh trọng số của 4 tiêu chí. Hệ thống sẽ tự động chuẩn hóa để tổng trọng số = 100%.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            w1 = st.number_input("Hạ tầng số (C1) [+]", value=0.35, step=0.05)
        with c2:
            w2 = st.number_input("Nhân lực số (C2) [+]", value=0.25, step=0.05)
        with c3:
            w3 = st.number_input("Năng lực R&D (C3) [+]", value=0.20, step=0.05)
        with c4:
            w4 = st.number_input("Rào cản (C4) [-]", value=0.20, step=0.05)
            
        weights = [w1, w2, w3, w4]

    # 🚀 CHẠY MÔ HÌNH
    if st.button("🚀 Chạy mô hình TOPSIS", type="primary"):
        with st.spinner("Đang tính toán khoảng cách Euclide..."):
            df_result = run_model(df_raw, weights, criteria_types)
        
        st.success("✅ Mô hình TOPSIS chạy thành công!")
        
        # 📊 KẾT QUẢ TÍNH TOÁN
        st.markdown("### 📊 Kết quả Xếp hạng TOPSIS")
        
        top1_region = df_result.iloc[0]["Vùng Kinh Tế"]
        top1_score = df_result.iloc[0]["Điểm TOPSIS (C*)"]
        top2_score = df_result.iloc[1]["Điểm TOPSIS (C*)"]
        bottom_region = df_result.iloc[-1]["Vùng Kinh Tế"]
        gap = top1_score - top2_score
        
        # Hiển thị KPI bằng st.metric()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🥇 Vùng xếp hạng Số 1", top1_region)
        m2.metric("⭐ Điểm TOPSIS cao nhất", f"{top1_score:.4f}")
        m3.metric("📉 Vùng xếp hạng cuối", bottom_region)
        m4.metric("📈 Khoảng cách Top 1 & Top 2", f"+{gap:.4f}")
        
        # 📋 HIỂN THỊ BẢNG (Làm tròn float thay vì dùng .style để tránh lỗi Streamlit)
        st.markdown("**Bảng Tổng hợp Xếp hạng & Tiêu chí Đánh giá:**")
        df_display = df_result[["Thứ hạng", "Vùng Kinh Tế", "Điểm TOPSIS (C*)", 
                                "Hạ tầng số (C1)", "Nhân lực số (C2)", 
                                "Năng lực R&D (C3)", "Rào cản (C4)"]].copy()
        
        # Dùng .round() để format hiển thị, KHÔNG DÙNG .style.format()
        df_display = df_display.round(4)
        st.dataframe(df_display, use_container_width=True)
        
        # 📈 TRỰC QUAN HÓA
        st.markdown("### 📈 Trực quan hóa Khoảng cách & Cấu trúc")
        fig = create_charts(df_result)
        st.pyplot(fig)
        
        # 💡 NHẬN XÉT & HÀM Ý CHÍNH SÁCH
        st.markdown("### 💡 Nhận xét & Hàm ý chính sách")
        st.markdown(f"""
        - **Vùng đứng đầu:** **{top1_region}** đạt điểm TOPSIS cao nhất. Nguyên nhân là vì khu vực này có khoảng cách gần nhất với Giải pháp lý tưởng ($A^*$) và xa nhất với Giải pháp phi lý tưởng ($A^-$). Đây là hệ sinh thái mạnh nhất về Hạ tầng số (C1) và có Rào cản (C4) thấp nhất.
        - **Khuyến nghị đầu tư:** Các siêu dự án về trung tâm dữ liệu (Data Center), công viên phần mềm, và trung tâm R&D Trí tuệ nhân tạo (AI) nên được ưu tiên giải ngân vào **{top1_region}** để tạo cực tăng trưởng "đầu tàu".
        - **Chiến lược cho vùng khó khăn:** Vùng **{bottom_region}** đứng chót bảng do thiếu vắng Năng lực R&D và Rào cản còn quá cao. Đối với vùng này, nhà nước chưa nên triển khai AI ngay, mà phải ưu tiên vốn giải quyết các bài toán cơ bản như xóa lõm sóng viễn thông và đào tạo kỹ năng số cho người lao động.
        """)
        
        st.info("Kết quả TOPSIS hỗ trợ ưu tiên phân bổ nguồn lực và hoạch định chính sách phát triển vùng trong khuôn khổ AIDEOM-VN.")

if __name__ == "__main__":
    run()