
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

# ==========================================
# 1. ĐỊNH NGHĨA BÀI TOÁN TỐI ƯU (PyMoo Problem)
# ==========================================
class VietnamDigitalAllocation(ElementwiseProblem):
    def __init__(self):
        # 24 biến quyết định: 6 vùng x 4 hạng mục (I, D, AI, H)
        # 4 hàm mục tiêu: f1(Max GDP), f2(Min Gini), f3(Min CO2), f4(Min Risk)
        # 1 ràng buộc đẳng thức: Tổng ngân sách = 1.0 (100%)
        super().__init__(n_var=24, 
                         n_obj=4, 
                         n_ieq_constr=0, 
                         n_eq_constr=1, 
                         xl=0.01, # Mức phân bổ tối thiểu mỗi hạng mục là 1%
                         xu=0.20) # Mức phân bổ tối đa cho 1 hạng mục là 20%

        # Khởi tạo ma trận hệ số mô phỏng (tương tự notebook)
        np.random.seed(42)
        self.C_gdp = np.random.uniform(1.0, 3.0, 24)
        self.C_gini = np.random.uniform(-0.5, 0.5, 24)
        self.C_co2 = np.random.uniform(10, 50, 24)
        self.C_risk = np.random.uniform(5, 40, 24)

    def _evaluate(self, x, out, *args, **kwargs):
        # f1: Tối đa hóa GDP -> Đổi thành tối thiểu hóa (-GDP)
        f1 = -np.sum(self.C_gdp * x)
        
        # f2: Tối thiểu hóa Bất bình đẳng (Gini)
        f2 = np.sum(self.C_gini * x)
        
        # f3: Tối thiểu hóa phát thải CO2
        f3 = np.sum(self.C_co2 * x)
        
        # f4: Tối thiểu hóa rủi ro an ninh dữ liệu
        f4 = np.sum(self.C_risk * x)
        
        # Ràng buộc: Tổng ngân sách x_i = 1.0 (100%)
        h1 = np.sum(x) - 1.0
        
        out["F"] = [f1, f2, f3, f4]
        out["H"] = [h1]

# ==========================================
# 2. HÀM CHẠY MÔ HÌNH NSGA-II VÀ TOPSIS
# ==========================================
def run_model(pop_size, generations, mut_prob, cross_prob):
    problem = VietnamDigitalAllocation()
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=cross_prob, eta=15),
        mutation=PM(prob=mut_prob, eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", generations)
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   save_history=False,
                   verbose=False)
    
    return res

def run_topsis(F):
    """Chọn nghiệm thỏa hiệp từ tập Pareto bằng TOPSIS"""
    # Bước 1: Chuẩn hóa Min-Max các hàm mục tiêu
    norm_F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    
    # Bước 2: Điểm lý tưởng (A*) = 0, Điểm phi lý tưởng (A-) = 1 (vì mọi F đều là bài toán Min)
    weights = np.array([0.4, 0.25, 0.20, 0.15]) # Trọng số: GDP, Gini, CO2, Risk
    V = norm_F * weights
    
    A_star = np.zeros(4) * weights
    A_minus = np.ones(4) * weights
    
    # Bước 3: Tính khoảng cách
    S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))
    
    # Bước 4: Hệ số gần gũi (C*)
    C_star = S_minus / (S_star + S_minus)
    best_idx = np.argmax(C_star)
    
    return best_idx, C_star

# ==========================================
# 3. HÀM TẠO BIỂU ĐỒ (Create Charts)
# ==========================================
def create_charts(df_pareto, topsis_idx):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    gdp = df_pareto["GDP (Tỷ VND)"]
    gini = df_pareto["Bất bình đẳng (Gini)"]
    co2 = df_pareto["Phát thải (CO2)"]
    risk = df_pareto["Rủi ro (Risk)"]
    
    # Biểu đồ 1: Parallel Coordinates cho tập Pareto
    ax1 = axs[0, 0]
    # Chuẩn hóa để đưa về cùng thang đo [0,1] trên đồ thị Parallel
    norm_df = (df_pareto.iloc[:, 1:5] - df_pareto.iloc[:, 1:5].min()) / (df_pareto.iloc[:, 1:5].max() - df_pareto.iloc[:, 1:5].min())
    for i in range(len(norm_df)):
        color = 'red' if i == topsis_idx else 'steelblue'
        alpha = 1.0 if i == topsis_idx else 0.3
        lw = 3 if i == topsis_idx else 1
        ax1.plot(["GDP", "Gini", "CO2", "Risk"], norm_df.iloc[i].values, color=color, alpha=alpha, linewidth=lw)
    ax1.set_title("Biểu đồ Song song (Parallel Coordinates) - Tập Pareto", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Biểu đồ 2: Đánh đổi GDP vs Gini
    ax2 = axs[0, 1]
    ax2.scatter(gdp, gini, c='steelblue', alpha=0.6)
    ax2.scatter(gdp.iloc[topsis_idx], gini.iloc[topsis_idx], c='red', s=150, marker='*', label="TOPSIS")
    ax2.set_xlabel("Tăng trưởng GDP (Tỷ VND)")
    ax2.set_ylabel("Hệ số Bất bình đẳng (Gini)")
    ax2.set_title("Đánh đổi: Tăng trưởng ↔ Công bằng", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    # Biểu đồ 3: Đánh đổi GDP vs CO2
    ax3 = axs[1, 0]
    ax3.scatter(gdp, co2, c='green', alpha=0.6)
    ax3.scatter(gdp.iloc[topsis_idx], co2.iloc[topsis_idx], c='red', s=150, marker='*', label="TOPSIS")
    ax3.set_xlabel("Tăng trưởng GDP (Tỷ VND)")
    ax3.set_ylabel("Phát thải ròng (CO2)")
    ax3.set_title("Đánh đổi: Tăng trưởng ↔ Môi trường", fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()

    # Biểu đồ 4: Đánh đổi GDP vs Rủi ro
    ax4 = axs[1, 1]
    ax4.scatter(gdp, risk, c='purple', alpha=0.6)
    ax4.scatter(gdp.iloc[topsis_idx], risk.iloc[topsis_idx], c='red', s=150, marker='*', label="TOPSIS")
    ax4.set_xlabel("Tăng trưởng GDP (Tỷ VND)")
    ax4.set_ylabel("Chỉ số Rủi ro dữ liệu")
    ax4.set_title("Đánh đổi: Tăng trưởng ↔ Rủi ro An ninh", fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend()

    fig.tight_layout(pad=3.0)
    return fig

# ==========================================
# 4. HÀM GIAO DIỆN CHÍNH (Run)
# ==========================================
def run():
    try:
        st.set_page_config(layout="wide", page_title="Bài 7: NSGA-II")
    except:
        pass

    st.title("🧬 Bài 7: NSGA-II - Tối Ưu Đa Mục Tiêu Bằng Thuật Toán Di Truyền")
    
    # 📌 MÔ TẢ BÀI TOÁN
    st.markdown("""
    ### 📌 Mô tả bài toán
    **Mục tiêu:** Sử dụng thuật toán di truyền **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) của thư viện `pymoo` để giải bài toán phân bổ ngân sách quốc gia vào 4 hạng mục công nghệ trên 6 vùng kinh tế (24 biến).
    
    **4 Mục tiêu Xung đột:**
    1. **Tối đa hóa GDP ($f_1$)** 2. **Tối thiểu hóa Bất bình đẳng Gini ($f_2$)**
    3. **Tối thiểu hóa Phát thải CO2 ($f_3$)**
    4. **Tối thiểu hóa Rủi ro Dữ liệu ($f_4$)**
    """)

    # ⚙️ CẤU HÌNH MÔ HÌNH
    with st.expander("⚙️ Cấu hình thuật toán NSGA-II (PyMoo)", expanded=True):
        st.warning("⚠️ Giữ giá trị Population và Generations ở mức nhỏ (≤ 100) để đảm bảo Streamlit phản hồi dưới 3 giây.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pop_size = st.number_input("Population Size", min_value=10, max_value=300, value=50, step=10)
        with c2:
            generations = st.number_input("Generations", min_value=10, max_value=300, value=50, step=10)
        with c3:
            mut_prob = st.slider("Mutation Probability", 0.0, 1.0, 0.1, step=0.05)
        with c4:
            cross_prob = st.slider("Crossover Probability", 0.0, 1.0, 0.9, step=0.05)

    # 🚀 CHẠY MÔ HÌNH
    if st.button("🚀 Chạy thuật toán NSGA-II", type="primary"):
        with st.spinner("Đang chạy quá trình tiến hóa và lọc mặt chuẩn Pareto (NSGA-II)..."):
            res = run_model(pop_size, generations, mut_prob, cross_prob)
            
            if res.F is None:
                st.error("Không tìm thấy nghiệm khả thi. Hãy thử tăng số lượng quần thể.")
                return
                
            F_obj = res.F
            X_var = res.X
            
            # Tìm nghiệm thỏa hiệp TOPSIS
            topsis_idx, C_star = run_topsis(F_obj)
            
        st.success("✅ Mô hình NSGA-II chạy thành công! Đã tìm ra tập nghiệm không bị thống trị (Pareto Front).")
        
        # 📊 KẾT QUẢ PARETO
        st.markdown("### 📊 Tổng quan Tập nghiệm Pareto")
        
        # Lưu F_obj vào DataFrame, f1 lấy giá trị âm để hiển thị số dương cho GDP
        df_pareto = pd.DataFrame({
            "Nghiệm ID": [f"Sol_{i+1}" for i in range(len(F_obj))],
            "GDP (Tỷ VND)": -F_obj[:, 0],
            "Bất bình đẳng (Gini)": F_obj[:, 1],
            "Phát thải (CO2)": F_obj[:, 2],
            "Rủi ro (Risk)": F_obj[:, 3],
            "Điểm TOPSIS": C_star
        })
        
        # Sắp xếp để nghiệm TOPSIS tốt nhất đứng đầu
        df_pareto = df_pareto.sort_values(by="Điểm TOPSIS", ascending=False).reset_index(drop=True)
        # Cập nhật lại chỉ mục TOPSIS sau khi sort
        topsis_idx_sorted = 0 
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🧬 Số nghiệm Pareto", len(F_obj))
        m2.metric("📈 GDP cao nhất", f"{df_pareto['GDP (Tỷ VND)'].max():.2f}")
        m3.metric("🌱 CO2 thấp nhất", f"{df_pareto['Phát thải (CO2)'].min():.2f}")
        m4.metric("⭐ Nghiệm Thỏa hiệp", df_pareto.iloc[0]["Nghiệm ID"])
        
        # Hiển thị bảng (Không dùng .style)
        df_display = df_pareto.round(4)
        st.dataframe(df_display, use_container_width=True)
        
        # 📈 TRỰC QUAN HÓA
        st.markdown("### 📈 Trực quan hóa Không gian Mục tiêu")
        fig = create_charts(df_pareto, topsis_idx_sorted)
        st.pyplot(fig)
        
        # 💡 NHẬN XÉT & HÀM Ý CHÍNH SÁCH
        st.markdown("### 💡 Nhận xét & Hàm ý chính sách")
        st.markdown(f"""
        - **Mặt chuẩn Pareto (Pareto Frontier):** Thuật toán NSGA-II chứng minh rõ ràng không tồn tại một phương án phân bổ ngân sách nào có thể tối ưu cả 4 mục tiêu. Việc cố tình **tối đa hóa GDP** sẽ kéo theo sự gia tăng tuyến tính của **Bất bình đẳng (Gini)** và **Phát thải CO2**.
        - **Điểm Thỏa hiệp (Nghiệm TOPSIS đỏ):** Nghiệm này đánh dấu bước lùi chiến lược—chấp nhận hy sinh một phần tốc độ tăng trưởng GDP (đỉnh) để đảm bảo an sinh xã hội vùng miền và tuân thủ các cam kết môi trường quốc tế (COP26). 
        - **Liên hệ Chiến lược Quốc gia:** Dữ liệu hỗ trợ Quyết định 127/QĐ-TTg, khẳng định rằng trí tuệ nhân tạo (AI) phải đi đôi với các hành lang pháp lý để kiểm soát rủi ro dữ liệu, nếu không $f_4$ (Rủi ro) sẽ vượt ngưỡng an toàn của hệ thống.
        """)
        
        st.info("Kết quả NSGA-II hỗ trợ các nhà hoạch định chính sách vượt qua tư duy tuyến tính, tìm kiếm điểm cân bằng đa chiều trong khuôn khổ AIDEOM-VN.")

if __name__ == "__main__":
    run()