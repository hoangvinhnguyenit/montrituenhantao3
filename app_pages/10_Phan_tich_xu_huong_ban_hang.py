# -*- coding: utf-8 -*-
"""Dedicated sales-trend and seasonality analysis."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import (
    configure_page,
    filter_controls,
    get_active_sales,
    initialize_session_state,
    show_data_required,
)
from theme import page_intro
from utils import aggregate_daily


WEEKDAY_LABELS = {
    0: "Thứ 2",
    1: "Thứ 3",
    2: "Thứ 4",
    3: "Thứ 5",
    4: "Thứ 6",
    5: "Thứ 7",
    6: "Chủ nhật",
}

configure_page("Phân tích xu hướng bán hàng")
initialize_session_state()

page_intro(
    "Phân tích xu hướng bán hàng",
    "Theo dõi nhịp bán theo thời gian, mùa vụ trong tuần và nhóm hàng nổi bật.",
)

sales = get_active_sales()
if show_data_required(sales):
    st.stop()

filtered = filter_controls(sales, key_prefix="sales_trend")
filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
filtered["sales"] = pd.to_numeric(filtered["sales"], errors="coerce")
filtered = filtered.dropna(subset=["date", "sales"])
if filtered.empty:
    st.warning("Không có dữ liệu bán hàng hợp lệ sau khi áp dụng bộ lọc.")
    st.stop()

minimum_date = filtered["date"].min().date()
maximum_date = filtered["date"].max().date()
date_range = st.date_input(
    "Khoảng thời gian",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
    key="sales_trend_date_range",
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] < end)]

if filtered.empty:
    st.info("Không có dữ liệu trong khoảng thời gian đã chọn.")
    st.stop()

daily = aggregate_daily(filtered)
period_options = {"Ngày": "D", "Tuần": "W", "Tháng": "MS"}
frequency = st.radio(
    "Gom doanh số theo",
    list(period_options),
    horizontal=True,
    key="sales_trend_frequency",
)
trend = aggregate_daily(filtered, freq=period_options[frequency])
trend["Trung bình trượt"] = trend["sales"].rolling(7, min_periods=1).mean()

metrics = st.columns(3)
metrics[0].metric("Doanh số trong kỳ", f"{filtered['sales'].sum():,.0f}")
metrics[1].metric("Doanh số trung bình/ngày", f"{daily['sales'].mean():,.1f}")
peak = daily.loc[daily["sales"].idxmax()]
metrics[2].metric("Ngày bán cao nhất", f"{peak['date']:%d/%m/%Y}")

with st.container(border=True):
    st.subheader(f"Xu hướng doanh số theo {frequency.casefold()}")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=trend["date"],
            y=trend["sales"],
            name="Doanh số",
            mode="lines",
            line={"width": 2.5, "color": "#0d9f6e"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trend["date"],
            y=trend["Trung bình trượt"],
            name="Trung bình trượt 7 kỳ",
            mode="lines",
            line={"width": 2, "dash": "dash", "color": "#f59e0b"},
        )
    )
    figure.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Doanh số",
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")

weekday_data = filtered.assign(
    weekday=filtered["date"].dt.dayofweek.map(WEEKDAY_LABELS)
).groupby("weekday", as_index=False)["sales"].mean()
weekday_data["weekday"] = pd.Categorical(
    weekday_data["weekday"],
    categories=list(WEEKDAY_LABELS.values()),
    ordered=True,
)
weekday_data = weekday_data.sort_values("weekday")

season_col, family_col = st.columns(2)
with season_col:
    with st.container(border=True):
        st.subheader("Mùa vụ theo ngày trong tuần")
        st.plotly_chart(
            px.bar(
                weekday_data,
                x="weekday",
                y="sales",
                labels={"weekday": "Ngày", "sales": "Doanh số trung bình"},
                color="sales",
                color_continuous_scale="Tealgrn",
            ),
            width="stretch",
        )

with family_col:
    with st.container(border=True):
        st.subheader("Nhóm hàng đóng góp nhiều nhất")
        if "family" in filtered:
            family_sales = (
                filtered.groupby("family", as_index=False)["sales"]
                .sum()
                .sort_values("sales", ascending=False)
                .head(10)
            )
            st.plotly_chart(
                px.bar(
                    family_sales.sort_values("sales"),
                    x="sales",
                    y="family",
                    orientation="h",
                    labels={"family": "Nhóm hàng", "sales": "Doanh số"},
                    color="sales",
                    color_continuous_scale="Tealgrn",
                ),
                width="stretch",
            )
        else:
            st.info("Dữ liệu hiện tại chưa có cột nhóm hàng.")
