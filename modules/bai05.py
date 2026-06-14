import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp

# ==========================================
# 1. HÀM TẢI DỮ LIỆU (Load Data)
# ==========================================
@st.cache_data
def load_data():
    """
    Tái tạo chính xác bộ dữ liệu 15 dự án từ Bài 05.
    P14: An ninh mạng (bắt buộc)
    P15: Open Data (B/C cao, chi phí 1.500)
    P8 & P13: Có khả năng cộng hưởng (AI & Bán dẫn)
    """
    data = {
        "ID": [f"P{i}" for i in range(1, 16)],
        "Tên Dự án": [
            "Hạ tầng Đám mây (G-Cloud)", "CSDL Quốc gia về Dân cư", "Hệ thống định danh điện tử",
            "Trục liên thông văn bản quốc gia", "Cổng Dịch vụ công", "Nền tảng thanh toán số",
            "Hệ thống giám sát (IOC)", "Nền tảng AI quốc gia", "CĐS Y tế (Hồ sơ SK)",
            "CĐS Giáo dục (Học bạ ĐT)", "Nền tảng Nông nghiệp số", "Logistics thông minh",
            "Công nghiệp Bán dẫn", "An ninh mạng lõi", "Dữ liệu mở (Open Data)"
        ],
        "Chi phí (Tỷ VNĐ)": [5000, 4500, 2000, 1800, 2500, 1500, 3000, 3500, 2800, 2200, 1900, 3200, 6000, 2500, 1500],
        "Lợi ích kỳ vọng":  [9000, 8500, 3800, 3200, 4800, 3000, 5200, 7500, 5000, 4200, 3500, 5800, 11000, 3500, 4500],
        "Lĩnh vực": [
            "Hạ tầng", "Dữ liệu", "Dữ liệu", "Chính phủ số", "Chính phủ số", "Kinh tế số",
            "Chính phủ số", "Công nghệ lõi", "Xã hội số", "Xã hội số", "Kinh tế số",
            "Kinh tế số", "Công nghệ lõi", "An ninh mạng", "Dữ liệu"
        ]
    }
    df = pd.DataFrame(data)
    df["Tỷ suất B/C"] = df["Lợi ích kỳ vọng"] / df["Chi phí (Tỷ VNĐ)"]
    return df

# ==========================================
# 2. HÀM CHẠY MÔ HÌNH (Run Model - MIP bằng PuLP)
# ==========================================
def run_model(df, budget, force_p14, apply_synergy, mutex_p4_p5):
    # Khởi tạo mô hình
    model = pulp.LpProblem("Danh_Muc_Du_An_CDSO", pulp.LpMaximize)
    
    # Biến quyết định y_i thuộc {0, 1}
    project_ids = df["ID"].tolist()
    y = pulp.LpVariable.dicts("Project", project_ids, cat='Binary')
    
    # Biến cộng hưởng (synergy) giữa P8 và P13
    z_8_13 = pulp.LpVariable("Synergy_P8_P13", cat='Binary')
    synergy_bonus = 3000 if apply_synergy else 0

    # Hàm mục tiêu: Tổng lợi ích + Lợi ích cộng hưởng
    costs = dict(zip(df["ID"], df["Chi phí (Tỷ VNĐ)"]))
    benefits = dict(zip(df["ID"], df["Lợi ích kỳ vọng"]))
    
    model += pulp.lpSum([benefits[i] * y[i] for i in project_ids]) + synergy_bonus * z_8_13, "Total_Benefit"
    
    # Ràng buộc 1: Giới hạn ngân sách
    model += pulp.lpSum([costs[i] * y[i] for i in project_ids]) <= budget, "Budget_Constraint"
    
    # Ràng buộc 2: Bắt buộc chọn P14 (An ninh mạng)
    if force_p14:
        model += y["P14"] == 1, "Force_CyberSecurity"
        
    # Ràng buộc 3: Tiên quyết (Precedence) - P8 (AI) cần P1 (Hạ tầng đám mây)
    model += y["P8"] <= y["P1"], "Precedence_P8_requires_P1"
    
    # Ràng buộc 4: Loại trừ lẫn nhau (Mutually Exclusive) giữa P4 và P5
    if mutex_p4_p5:
        model += y["P4"] + y["P5"] <= 1, "Mutex_P4_P5"
        
    # Ràng buộc 5: Logic của biến cộng hưởng z_8_13 = y_8 AND y_13
    model += z_8_13 <= y["P8"], "Synergy_Logic_1"
    model += z_8_13 <= y["P13"], "Synergy_Logic_2"
    model += z_8_13 >= y["P8"] + y["P13"] - 1, "Synergy_Logic_3"

    # Giải bài toán
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Xử lý kết quả
    status = pulp.LpStatus[model.status]
    selected_projects = [i for i in project_ids if y[i].varValue == 1.0]
    total_cost = sum([costs[i] for i in selected_projects])
    total_benefit = pulp.value(model.objective)
    
    return status, selected_projects, total_cost, total_benefit

# ==========================================
# 3. HÀM TẠO BIỂU ĐỒ (Create Charts)
# ==========================================
def create_charts(df, selected_projects):
    df_selected = df[df["ID"].isin(selected_projects)].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Biểu đồ 1: Bar chart - Chi phí và Lợi ích của các dự án được chọn
    x = np.arange(len(df_selected))
    width = 0.35
    
    ax1.bar(x - width/2, df_selected["Chi phí (Tỷ VNĐ)"], width, label='Chi phí', color='#d62728')
    ax1.bar(x + width/2, df_selected["Lợi ích kỳ vọng"], width, label='Lợi ích', color='#2ca02c')
    
    ax1.set_ylabel('Tỷ VNĐ', fontweight='bold')
    ax1.set_title('Chi phí vs Lợi ích của các Dự án được chọn', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_selected["ID"], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Biểu đồ 2: Pie chart - Cơ cấu phân bổ ngân sách theo Lĩnh vực
    sector_costs = df_selected.groupby("Lĩnh vực")["Chi phí (Tỷ VNĐ)"].sum()
    
    colors = plt.cm.Paired(np.linspace(0, 1, len(sector_costs)))
    explode = [0.05] * len(sector_costs)
    
    ax2.pie(sector_costs, labels=sector_costs.index, autopct='%1.1f%%', 
            startangle=90, colors=colors, explode=explode, textprops={'fontsize': 10})
    ax2.set_title('Cơ cấu phân bổ Ngân sách theo Lĩnh vực', fontweight='bold')
    
    fig.tight_layout()
    return fig

# ==========================================
# 4. HÀM GIAO DIỆN CHÍNH (Run)
# ==========================================
def run():
    try:
        st.set_page_config(layout="wide", page_title="Bài 5: MIP Lựa chọn Dự án")
    except:
        pass

    st.title("💼 Bài 5: Quy hoạch Nguyên Hỗn hợp (MIP) - Lựa chọn Dự án CĐS")
    
    # 📌 MÔ TẢ BÀI TOÁN
    st.markdown("""
    ### 📌 Mô tả bài toán
    **Mục tiêu:** Tối ưu hóa danh mục đầu tư từ **15 dự án chuyển đổi số quốc gia** với nguồn ngân sách hữu hạn.
    Đây là bài toán cái túi (Knapsack Problem) mở rộng sử dụng **Quy hoạch Nguyên Hỗn hợp (Mixed Integer Programming)**.
    
    **Mô hình Toán học:**
    $$\\max Z = \\sum_{i=1}^{15} B_i \\cdot y_i + \\Delta B_{synergy} \\cdot z_{8,13}$$
    
    *Trong đó:*
    - **$y_i \\in \\{0, 1\\}$**: Quyết định chọn (1) hoặc không chọn (0) dự án $i$.
    - **$z_{8,13}$**: Biến cộng hưởng nếu chọn cả P8 (AI) và P13 (Bán dẫn).
    - **Ràng buộc:** Ngân sách ($C_i y_i \\le B$), Tiên quyết ($y_8 \\le y_1$), Bắt buộc ($y_{14} = 1$).
    """)

    df = load_data()

    # ⚙️ CẤU HÌNH MÔ HÌNH
    with st.expander("⚙️ Cấu hình mô hình (Tham số MIP)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Ngân sách tối đa (Tỷ VNĐ)", min_value=5000, max_value=50000, value=25000, step=1000)
            apply_synergy = st.checkbox("Kích hoạt cộng hưởng P8 & P13 (+3.000 Tỷ lợi ích)", value=True)
        with col2:
            force_p14 = st.checkbox("Bắt buộc chọn P14 (An ninh mạng)", value=True)
            mutex_p4_p5 = st.checkbox("Ràng buộc loại trừ (Chỉ chọn 1 trong 2: P4 hoặc P5)", value=True)

    # 🚀 CHẠY MÔ HÌNH
    if st.button("🚀 Chạy mô hình Tối ưu", type="primary"):
        status, selected_projects, total_cost, total_benefit = run_model(df, budget, force_p14, apply_synergy, mutex_p4_p5)
        
        if status == "Optimal":
            st.success("✅ Mô hình tối ưu chạy thành công! Đã tìm ra danh mục dự án tốt nhất.")
            
            # 📊 KẾT QUẢ TỐI ƯU
            st.markdown("### 📊 Kết quả Tối ưu")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💰 Tổng ngân sách sử dụng", f"{total_cost:,.0f} Tỷ", f"{(total_cost/budget)*100:.1f}%")
            m2.metric("⭐ Tổng giá trị lợi ích (Z*)", f"{total_benefit:,.0f} Tỷ")
            m3.metric("📈 Tỷ suất sinh lời (B/C)", f"{total_benefit/total_cost:.2f}")
            m4.metric("📁 Số dự án được chọn", f"{len(selected_projects)} / 15")
            
            # Đánh dấu dữ liệu hiển thị
            df_result = df.copy()
            df_result["Trạng thái"] = df_result["ID"].apply(lambda x: "✅ Được chọn" if x in selected_projects else "❌ Bị loại")
            
            st.dataframe(df_result.style.format({
                "Chi phí (Tỷ VNĐ)": "{:,.0f}", 
                "Lợi ích kỳ vọng": "{:,.0f}", 
                "Tỷ suất B/C": "{:.2f}"
            }).map(lambda x: "background-color: #d4edda; color: #155724" if "Được chọn" in str(x) else "color: #721c24", subset=["Trạng thái"]), 
            use_container_width=True)
            
            # 📈 TRỰC QUAN HÓA
            st.markdown("### 📈 Trực quan hóa")
            fig = create_charts(df, selected_projects)
            st.pyplot(fig)
            
            # 💡 NHẬN XÉT & HÀM Ý CHÍNH SÁCH
            st.markdown("### 💡 Nhận xét & Hàm ý chính sách")
            
            # Phân tích P15 (Open Data)
            p15_selected = "P15" in selected_projects
            p14_selected = "P14" in selected_projects
            
            st.markdown(f"""
            - **Tính bắt buộc & Chi phí cơ hội:** Ràng buộc `y14=1` (An ninh mạng) tiêu tốn 2.500 tỷ dù lợi ích sinh ra trực tiếp không cao. Việc bắt buộc này làm giảm tổng $Z^*$ nhưng là điều kiện tiên quyết (sine qua non) để bảo vệ hệ sinh thái số theo **Quyết định 749/QĐ-TTg**.
            - **Hiện tượng "Lọt lưới" (Knapsack problem):** Dự án **P15 (Open Data)** có tỷ suất B/C rất cao ({df[df['ID']=='P15']['Tỷ suất B/C'].values[0]:.2f}), với chi phí chỉ 1.500 tỷ. Tuy nhiên, nó {"**được chọn**" if p15_selected else "**bị loại**"} trong kịch bản này. Nguyên nhân do thuật toán MIP ưu tiên lấp đầy ngân sách bằng các dự án lớn mang lại giá trị tuyệt đối cao hơn hoặc bị chèn ép bởi các ràng buộc cứng.
            - **Hiệu ứng cộng hưởng (Synergy):** Nhờ thiết lập biến tương tác $z_{8,13}$, mô hình hiểu rằng phát triển **AI (P8)** và tự chủ **Bán dẫn (P13)** cùng lúc sẽ tạo ra bước nhảy vọt (+3000 Tỷ). Đây chính là cốt lõi của **mô hình AIDEOM-VN**, chuyển từ lợi ích tuyến tính sang lợi ích theo cấp số nhân.
            """)
            
            st.info("Kết quả hỗ trợ lựa chọn danh mục dự án chuyển đổi số tối ưu trong điều kiện ngân sách hữu hạn, tránh rủi ro đầu tư dàn trải.")
            
        else:
            st.error(f"Mô hình không tìm được điểm tối ưu (Trạng thái: {status}). Hãy thử tăng ngân sách hoặc nới lỏng ràng buộc.")

if __name__ == "__main__":
    run()