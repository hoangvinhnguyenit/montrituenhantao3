# -*- coding: utf-8 -*-
"""Create, inspect and export forecasting features."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import (
    configure_page,
    get_active_sales,
    initialize_session_state,
    show_data_required,
)
from theme import page_intro
from utils import engineer_features


configure_page("Tạo đặc trưng dự báo")
initialize_session_state()

page_intro(
    "Tạo đặc trưng dự báo",
    "Tạo đặc trưng lịch, khuyến mãi, độ trễ và trung bình trượt từ dữ liệu bán hàng.",
)

sales = get_active_sales()
if show_data_required(sales):
    st.stop()

feature_columns = [
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "onpromotion",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]
existing_features = [column for column in feature_columns if column in sales.columns]
if not existing_features:
    st.info(
        "Chưa có bộ đặc trưng trong phiên. Bạn có thể tạo từ dữ liệu hiện hành "
        "hoặc làm sạch dữ liệu trước ở trang **Tiền xử lý dữ liệu**."
    )

if st.button("Tạo / cập nhật đặc trưng", type="primary"):
    try:
        featured = engineer_features(sales)
        st.session_state.processed_df = featured
        st.success(
            f"Đã tạo {len(feature_columns):,} đặc trưng cho "
            f"{len(featured):,} dòng dữ liệu."
        )
        st.rerun()
    except ValueError as exc:
        st.error(f"Không thể tạo đặc trưng: {exc}")

featured = get_active_sales()
if featured is None or featured.empty:
    st.stop()
existing_features = [column for column in feature_columns if column in featured.columns]
if not existing_features:
    st.info("Bấm **Tạo / cập nhật đặc trưng** để bắt đầu.")
    st.stop()

calendar_features = [
    column
    for column in ("year", "month", "day", "day_of_week", "day_of_year", "is_weekend")
    if column in existing_features
]
lag_features = [column for column in ("lag_1", "lag_7", "lag_28") if column in existing_features]
rolling_features = [
    column
    for column in ("rolling_mean_7", "rolling_mean_28")
    if column in existing_features
]
metric_columns = st.columns(4)
metric_columns[0].metric("Số dòng", f"{len(featured):,}")
metric_columns[1].metric("Đặc trưng lịch", f"{len(calendar_features):,}")
metric_columns[2].metric("Đặc trưng độ trễ", f"{len(lag_features):,}")
metric_columns[3].metric("Đặc trưng trung bình trượt", f"{len(rolling_features):,}")

with st.container(border=True):
    st.subheader("Các nhóm đặc trưng")
    definitions = pd.DataFrame(
        [
            ("Lịch", ", ".join(calendar_features), "Mùa vụ, ngày trong tuần và cuối tuần."),
            ("Khuyến mãi", "onpromotion" if "onpromotion" in existing_features else "—", "Số mặt hàng được khuyến mãi."),
            ("Độ trễ", ", ".join(lag_features), "Doanh số ở các ngày trước đó."),
            ("Trung bình trượt", ", ".join(rolling_features), "Trung bình lịch sử, không bao gồm ngày hiện tại."),
        ],
        columns=["Nhóm", "Cột", "Ý nghĩa"],
    )
    st.dataframe(definitions, hide_index=True, width="stretch")
    st.caption(
        "Các cột lag và rolling chỉ dùng dữ liệu trước ngày hiện tại để tránh rò rỉ "
        "doanh số tương lai."
    )

with st.container(border=True):
    st.subheader("Đặc trưng doanh số theo thời gian")
    selected_features = st.multiselect(
        "Chọn đặc trưng để so sánh",
        [column for column in ("sales", *existing_features) if column in featured.columns],
        default=[
            column
            for column in ("sales", "rolling_mean_7", "rolling_mean_28")
            if column in featured.columns
        ],
        key="feature_engineering_chart_columns",
    )
    if selected_features and "date" in featured.columns:
        chart_aggregations = {
            column: "sum"
            if column in {"sales", "rolling_mean_7", "rolling_mean_28"}
            else "mean"
            for column in selected_features
        }
        chart_data = (
            featured.dropna(subset=["date"])
            .groupby("date", as_index=False)
            .agg(chart_aggregations)
            .sort_values("date")
        )
        figure = go.Figure()
        for column in selected_features:
            if column in chart_data:
                figure.add_trace(
                    go.Scatter(
                        x=chart_data["date"],
                        y=chart_data[column],
                        name=column,
                        mode="lines",
                    )
                )
        figure.update_layout(
            xaxis_title="Ngày",
            yaxis_title="Doanh số / giá trị đặc trưng",
            hovermode="x unified",
        )
        st.plotly_chart(figure, width="stretch")

with st.container(border=True):
    st.subheader("Bản xem trước và tải xuống")
    preview_columns = list(dict.fromkeys(["date", "store_nbr", "family", "sales", *existing_features]))
    preview_columns = [column for column in preview_columns if column in featured.columns]
    st.dataframe(featured[preview_columns].head(100), hide_index=True, width="stretch")
    st.download_button(
        "Tải dữ liệu kèm đặc trưng (CSV)",
        featured.to_csv(index=False).encode("utf-8-sig"),
        "du_lieu_feature_engineering.csv",
        "text/csv",
    )
