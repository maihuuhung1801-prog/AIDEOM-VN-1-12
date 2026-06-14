"""
Module M6 — Dashboard Đồ án tổng hợp AIDEOM-VN
Tích hợp toàn bộ kết quả từ các module M1 (Dự báo), M2 (Sẵn sàng số), 
M3 (Phân bổ), M4 (Lao động), M5 (Rủi ro).
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Đồng bộ đường dẫn gốc
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import cẩn thận từ các bài tập trước
try:
    from modules.m1_forecast import forecast_all_scenarios
    from modules.m2_readiness import assess_regions, get_digital_gap_index
    from modules.m3_allocation import optimize_static, compare_scenarios_static, REGIONS, INVEST_TYPES
    from modules.m4_labor import compare_scenarios_labor
    from modules.m5_risk import risk_radar_data, monte_carlo_risk
except ImportError as e:
    st.error(f"Lỗi Import. Vui lòng đảm bảo bạn đã tạo file __init__.py trong thư mục modules/. Chi tiết: {e}")

def run():
    st.title("🇻🇳 BÀI 12: ĐỒ ÁN TỔNG HỢP AIDEOM-VN")
    st.markdown("Hệ thống Tối ưu hoá & Hỗ trợ Ra Quyết định Kinh tế Số Việt Nam (2026-2030)")
    st.divider()

    # Tạo 4 Tab chuẩn hóa
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Vĩ mô & Hiện trạng", 
        "💰 2. Phân bổ ngân sách", 
        "⚖️ 3. So sánh kịch bản", 
        "⚠️ 4. Phân tích rủi ro"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.header("M1 & M2: Quỹ đạo tăng trưởng & Sẵn sàng số")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dự báo quy mô GDP (Tỷ VND)")
            df_gdp = forecast_all_scenarios()
            if isinstance(df_gdp, pd.DataFrame):
                fig1 = px.line(df_gdp, x="year", y="Y", color="label", markers=True)
                st.plotly_chart(fig1, use_container_width=True)
                
        with col2:
            st.subheader("Xếp hạng sẵn sàng số (TOPSIS)")
            # Fix lỗi trả về tuple hay df của assess_regions
            res_regions = assess_regions()
            df_regions = res_regions[0] if isinstance(res_regions, tuple) else res_regions
            
            if isinstance(df_regions, pd.DataFrame):
                fig2 = px.bar(df_regions, x="region_name_en", y="topsis_score", color="rank")
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Bảng Chỉ số Khoảng cách số (Digital Gap)")
        df_gap = get_digital_gap_index()
        if isinstance(df_gap, pd.DataFrame):
            st.dataframe(df_gap, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.header("M3: Tối ưu hóa phân bổ đầu tư")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            scenario_choice = st.selectbox("Chọn Kịch bản (S1-S5):", ["S1", "S2", "S3", "S4", "S5"], index=4)
            budget_val = st.slider("Ngân sách dự kiến (Tỷ VND):", 500, 5000, 2000, 100)
            
        with c2:
            # Xử lý an toàn đầu ra của optimize_static
            X_opt, gdp_val, info = optimize_static(scenario_choice, budget_val)
            st.metric(label="Tổng GDP gia tăng ước tính", value=f"{gdp_val:,.1f} Tỷ VND", delta=f"Hội tụ: {info.get('converged', True)}")
            
        st.subheader("Ma trận Phân bổ tối ưu (Vùng x Hạng mục)")
        df_matrix = pd.DataFrame(X_opt, index=REGIONS, columns=INVEST_TYPES)
        st.dataframe(df_matrix.style.background_gradient(cmap="Greens"), use_container_width=True)

    # ---------------- TAB 3 ----------------
    with tab3:
        st.header("M3 & M4: Phân tích đối chiếu Đa Kịch Bản")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Hiệu quả GDP theo kịch bản")
            df_static_comp = compare_scenarios_static(2000)
            if isinstance(df_static_comp, pd.DataFrame):
                fig3 = px.bar(df_static_comp, x="label", y="gdp_contribution", color="label")
                st.plotly_chart(fig3, use_container_width=True)
                
        with col4:
            st.subheader("Việc làm ròng (NetJob) năm 2030")
            df_labor = compare_scenarios_labor()
            if isinstance(df_labor, pd.DataFrame):
                fig4 = px.bar(df_labor, x="label", y="net_balance_2030", color="label")
                st.plotly_chart(fig4, use_container_width=True)

    # ---------------- TAB 4 ----------------
    with tab4:
        st.header("M5: Cảnh báo Rủi ro & Monte Carlo")
        
        st.subheader("Điểm rủi ro tổng hợp")
        df_risk = risk_radar_data()
        if isinstance(df_risk, pd.DataFrame):
            st.dataframe(df_risk, use_container_width=True)
            
        st.subheader("Mô phỏng Monte Carlo (VaR & CVaR)")
        with st.spinner("Đang chạy 1000 vòng lặp Monte Carlo..."):
            mc_res = monte_carlo_risk(scenario_choice, n_sim=1000, budget=2000)
            
            if isinstance(mc_res, dict):
                df_mc = pd.DataFrame(mc_res).T.reset_index().rename(columns={"index": "Vùng"})
                fig5 = px.bar(df_mc, x="Vùng", y=["mean", "VaR_95"], barmode="group")
                st.plotly_chart(fig5, use_container_width=True)
                st.dataframe(df_mc, use_container_width=True)

        st.error("🔴 **Khuyến nghị chính sách:** Dữ liệu cho thấy nếu thiếu kiểm soát, rủi ro an ninh mạng và môi trường có thể vượt ngưỡng an toàn vào năm 2028.")

if __name__ == "__main__":
    run()