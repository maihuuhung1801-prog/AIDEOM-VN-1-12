import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.title("📘 Bài 2: Quy hoạch Tuyến tính (LP) - Phân bổ Ngân sách Đầu tư Số")
    
    st.markdown("""
    **Mục tiêu:** Tối ưu hóa việc phân bổ ngân sách nhà nước cho 4 hạng mục chuyển đổi số cốt lõi nhằm tối đa hóa tổng giá trị kinh tế mang lại (Z), đồng thời thỏa mãn các ràng buộc về tỷ lệ vốn tối thiểu để đảm bảo phát triển đồng đều.
    
    **Mô hình Toán học:**
    $$\\max Z = c_1 x_1 + c_2 x_2 + c_3 x_3 + c_4 x_4$$
    
    *Trong đó (Các biến quyết định & Hệ số sinh lời ROI):*
    - **$x_1$**: Hạ tầng số ($c_1$)
    - **$x_2$**: CĐS Doanh nghiệp ($c_2$)
    - **$x_3$**: AI & R&D ($c_3$)
    - **$x_4$**: Nhân lực số ($c_4$)
    
    *Các ràng buộc hệ sinh thái (Chính sách):*
    - Tổng ngân sách: $x_1 + x_2 + x_3 + x_4 \\le B$
    - Ràng buộc tối thiểu: $x_1 \\ge 20\\%B$, $x_2 \\ge 15\\%B$, $x_3 \\ge 10\\%B$, $x_4 \\ge 15\\%B$
    - Ràng buộc rủi ro: $x_3 \\le 25\\%B$ (AI & R&D không vượt quá 25% do rủi ro cao)
    """)
    
    st.subheader("⚙️ Tham số Đầu vào (Ngân sách & Hệ số ROI)")
    
    # Khu vực tham số đầu vào
    total_budget = st.number_input("Tổng ngân sách B (Tỷ VNĐ)", min_value=1000, max_value=100000, value=10000, step=1000)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        c1 = st.number_input("ROI Hạ tầng số (c1)", value=1.15, step=0.05, format="%.2f")
    with col2:
        c2 = st.number_input("ROI CĐS DN (c2)", value=1.25, step=0.05, format="%.2f")
    with col3:
        c3 = st.number_input("ROI AI & R&D (c3)", value=1.35, step=0.05, format="%.2f")
    with col4:
        c4 = st.number_input("ROI Nhân lực (c4)", value=1.10, step=0.05, format="%.2f")

    if st.button("🚀 Chạy mô hình tối ưu phân bổ (LP)", type="primary"):
        # 1. Thuật toán mô phỏng LP (Linear Programming)
        # Khởi tạo mảng phân bổ tối thiểu theo ràng buộc
        min_alloc = {
            "Hạ tầng số": 0.20 * total_budget,
            "CĐS Doanh nghiệp": 0.15 * total_budget,
            "AI & R&D": 0.10 * total_budget,
            "Nhân lực số": 0.15 * total_budget
        }
        
        # Max ràng buộc cho AI
        max_ai = 0.25 * total_budget
        
        # Ngân sách còn lại sau khi rải đều mức tối thiểu
        allocated = sum(min_alloc.values())
        remaining = total_budget - allocated
        
        # Lưu ROI để xếp hạng
        rois = {
            "Hạ tầng số": c1,
            "CĐS Doanh nghiệp": c2,
            "AI & R&D": c3,
            "Nhân lực số": c4
        }
        
        # Phân bổ phần ngân sách thặng dư (greedy approach to simulate Simplex)
        # Ưu tiên dồn tiền vào nơi có ROI cao nhất, nhưng phải tuân thủ max bounds
        final_alloc = min_alloc.copy()
        sorted_categories = sorted(rois.items(), key=lambda item: item[1], reverse=True)
        
        for cat, roi in sorted_categories:
            if remaining <= 0:
                break
                
            if cat == "AI & R&D":
                # Chỉ được bơm thêm cho đến khi đạt max_ai
                allowance = max_ai - final_alloc[cat]
                add_amount = min(remaining, allowance)
                final_alloc[cat] += add_amount
                remaining -= add_amount
            else:
                # Các hạng mục khác không có trần giới hạn
                final_alloc[cat] += remaining
                remaining -= remaining

        # Tính tổng giá trị kinh tế (Z)
        z_value = sum(final_alloc[cat] * rois[cat] for cat in rois)
        
        # 2. Tạo DataFrame kết quả
        df_result = pd.DataFrame({
            "Hạng mục": list(final_alloc.keys()),
            "Tỷ lệ yêu cầu": ["≥ 20%", "≥ 15%", "≥ 10% (Max 25%)", "≥ 15%"],
            "Hệ số sinh lời (ROI)": list(rois.values()),
            "Phân bổ tối ưu (Tỷ VNĐ)": list(final_alloc.values()),
            "Giá trị mang lại (Tỷ VNĐ)": [final_alloc[cat] * rois[cat] for cat in final_alloc]
        })
        
        # 3. Hiển thị bảng kết quả
        st.subheader(f"📊 Bảng Phương án Tối ưu (Tổng lợi ích Z* = {z_value:,.0f} Tỷ VNĐ)")
        st.dataframe(df_result.style.format({
            "Hệ số sinh lời (ROI)": "{:.2f}",
            "Phân bổ tối ưu (Tỷ VNĐ)": "{:,.0f}",
            "Giá trị mang lại (Tỷ VNĐ)": "{:,.0f}"
        }), use_container_width=True)
        
        # 4. Trực quan hóa Biểu đồ
        st.subheader("📈 Trực quan hóa Cơ cấu Phân bổ")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Biểu đồ tròn (Pie chart)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = df_result["Hạng mục"]
        sizes = df_result["Phân bổ tối ưu (Tỷ VNĐ)"]
        explode = [0.05 if x == max(sizes) else 0 for x in sizes]
        
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax1.set_title("Tỷ trọng Phân bổ Ngân sách", fontweight='bold')
        
        # Biểu đồ cột (Bar chart) hiển thị sự chuyển hóa từ Vốn -> Lợi ích
        x = np.arange(len(labels))
        width = 0.35
        
        ax2.bar(x - width/2, df_result["Phân bổ tối ưu (Tỷ VNĐ)"], width, label='Vốn đầu tư', color='#8c564b')
        ax2.bar(x + width/2, df_result["Giá trị mang lại (Tỷ VNĐ)"], width, label='Lợi ích (ROI)', color='#9467bd')
        
        ax2.set_ylabel('Giá trị (Tỷ VNĐ)', fontweight='bold')
        ax2.set_title('So sánh Vốn Phân bổ và Giá trị Lợi ích', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # 5. Nhận xét phân tích
        max_alloc_cat = df_result.loc[df_result["Phân bổ tối ưu (Tỷ VNĐ)"].idxmax(), "Hạng mục"]
        
        st.subheader("💡 Nhận xét chính sách & Ý nghĩa kinh tế")
        st.success(f"""
        - **Chiến lược tối ưu:** Sau khi đáp ứng đủ mức duy trì hệ sinh thái cơ bản (các mức tối thiểu 10-20%), mô hình đã dồn toàn bộ nguồn lực thặng dư vào **{max_alloc_cat}** để tối đa hóa điểm Z.
        - **Giá trị đối ngẫu (Shadow Price):** Việc AI & R&D bị chặn ở ngưỡng 25% (Ràng buộc rủi ro) cho thấy: Dù AI có ROI cao nhất ({c3}), chính phủ vẫn không thể tất tay (all-in). Phần vốn dư thừa buộc phải chảy sang hạng mục có ROI cao thứ 2 (thường là CĐS Doanh nghiệp). 
        - **Khuyến nghị:** Nếu muốn tăng giới hạn đầu tư cho AI, nhà nước cần có chính sách kiểm soát rủi ro công nghệ (nới lỏng trần 25%). Ngược lại, nếu ROI của Nhân lực số tăng lên, thuật toán sẽ tự động điều chỉnh dòng vốn chuyển dịch về giáo dục và đào tạo.
        """)

if __name__ == "__main__":
    run()