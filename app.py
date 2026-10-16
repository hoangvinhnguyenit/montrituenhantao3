# -*- coding: utf-8 -*-
"""Streamlit page router (giao diện đồng bộ với index.html)."""

import streamlit as st

from theme import apply_theme

st.set_page_config(
    page_title="Dự báo doanh số & quản trị tồn kho",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS toàn cục: áp dụng cho mọi trang vì app.py chạy trước mỗi trang.
apply_theme()

pages = [
    st.Page("app_pages/8_Huan_luyen_lstm_gru.py", title="Mô hình huấn luyện", icon=":material/neurology:"),
    st.Page("app_pages/5_Du_bao.py", title="Dự báo doanh số", icon=":material/trending_up:"),
    st.Page("app_pages/9_Feature_Engineering.py", title="Tạo đặc trưng dự báo", icon=":material/auto_awesome:"),
    st.Page("app_pages/2_Tien_xu_ly.py", title="Tiền xử lý dữ liệu", icon=":material/tune:"),
    st.Page("app_pages/3_Phan_tich_EDA.py", title="Phân tích dữ liệu", icon=":material/analytics:"),
    st.Page("app_pages/10_Phan_tich_xu_huong_ban_hang.py", title="Phân tích xu hướng bán hàng", icon=":material/show_chart:"),
    st.Page("app_pages/4_Huan_luyen.py", title="Đánh giá mô hình", icon=":material/model_training:"),
    st.Page("app_pages/0_Dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("app_pages/6_Quan_tri_ton_kho.py", title="Tồn kho an toàn & Đặt hàng lại", icon=":material/inventory_2:"),
    st.Page("app_pages/11_Canh_bao_ton_kho.py", title="Cảnh báo tồn kho", icon=":material/notifications_active:"),
    st.Page("app_pages/1_Du_lieu.py", title="Dữ liệu bán hàng", icon=":material/table_view:"),
    st.Page("app_pages/7_Bao_cao.py", title="Báo cáo tổng hợp", icon=":material/description:"),
]

st.navigation(pages, position="sidebar").run()
