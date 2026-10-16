# -*- coding: utf-8 -*-
"""Sales and inventory overview dashboard."""

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import configure_page, get_active_sales, initialize_session_state
from model import forecast_future
from theme import page_intro
from utils import aggregate_daily


configure_page("Dashboard")
initialize_session_state()

page_intro(
    "Dashboard",
    "Tổng quan doanh số, kết quả dự báo và tình hình bán hàng.",
)

sales = get_active_sales()
if sales is None or sales.empty:
    st.info("Mở trang **Dữ liệu** ở thanh bên để tải dữ liệu hoặc tạo dữ liệu mẫu.")
    st.stop()

training_series = st.session_state.get("training_series")
if training_series is not None and not training_series.empty:
    history = training_series[["date", "sales"]].copy()
else:
    history = aggregate_daily(sales)

history["date"] = pd.to_datetime(history["date"], errors="coerce")
history["sales"] = pd.to_numeric(history["sales"], errors="coerce")
history = (
    history.dropna(subset=["date", "sales"])
    .sort_values("date")
    .tail(60)
    .reset_index(drop=True)
)
if history.empty:
    st.warning("Không có dữ liệu doanh số hợp lệ để hiển thị.")
    st.stop()

forecast = st.session_state.get("forecast_result")
if forecast is not None and not forecast.empty:
    forecast = forecast.copy()
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    forecast["forecast"] = pd.to_numeric(forecast["forecast"], errors="coerce")
    forecast = (
        forecast.dropna(subset=["date", "forecast"])
        .sort_values("date")
        .reset_index(drop=True)
    )

metrics = st.session_state.get("test_metrics")
prediction = st.session_state.get("test_pred_data")
families_count = (
    int(sales["family"].nunique()) if "family" in sales.columns else 0
)
recent_sales = float(history["sales"].tail(60).sum())
daily_average = recent_sales / len(history)

sales_with_families = sales.copy()
if "family" in sales_with_families.columns:
    sales_with_families["date"] = pd.to_datetime(
        sales_with_families["date"], errors="coerce"
    )
    sales_with_families["sales"] = pd.to_numeric(
        sales_with_families["sales"], errors="coerce"
    )
    last_sales_date = sales_with_families["date"].max()
    recent_start = last_sales_date - pd.Timedelta(days=15)
    previous_start = recent_start - pd.Timedelta(days=16)
    family_periods = sales_with_families.dropna(
        subset=["date", "family", "sales"]
    )
    recent_by_family = (
        family_periods.loc[family_periods["date"].between(recent_start, last_sales_date)]
        .groupby("family")["sales"]
        .sum()
    )
    previous_by_family = (
        family_periods.loc[
            family_periods["date"].between(previous_start, recent_start, inclusive="left")
        ]
        .groupby("family")["sales"]
        .sum()
    )
    comparable_families = previous_by_family[previous_by_family > 0]
    growing_families = int(
        (
            recent_by_family.reindex(comparable_families.index, fill_value=0)
            / comparable_families
            - 1
            > 0.10
        ).sum()
    )
else:
    growing_families = 0

forecast_total = (
    float(forecast["forecast"].sum())
    if forecast is not None and not forecast.empty
    else None
)
forecast_days = len(forecast) if forecast_total is not None else 16
previous_sales = float(history["sales"].tail(forecast_days).sum())
forecast_change = (
    (forecast_total / previous_sales - 1) * 100
    if forecast_total is not None and previous_sales > 0
    else None
)
mape = metrics.get("MAPE (%)") if metrics else None
accuracy = 100 - float(mape) if mape is not None and np.isfinite(mape) else None

cards = [
    (
        "Doanh số 60 ngày qua",
        f"{recent_sales / 1_000_000:.2f} triệu",
        f"Trung bình {daily_average:,.0f} đơn vị/ngày",
        "🪙",
    ),
    (
        f"Dự báo {forecast_days} ngày tới",
        f"{forecast_total / 1_000_000:.2f} triệu"
        if forecast_total is not None
        else "Chưa có",
        f"{forecast_change:+.1f}% so với kỳ trước"
        if forecast_change is not None
        else "Tạo dự báo từ mô hình đã huấn luyện",
        "📈",
    ),
    (
        "Độ chính xác kiểm định",
        f"{accuracy:.1f}%" if accuracy is not None else "Chưa có",
        f"100% − MAPE {float(mape):.1f}%" if accuracy is not None else "Huấn luyện để xem chỉ số",
        "🎯",
    ),
    (
        "Nhóm hàng cần kiểm tra tồn",
        f"{growing_families} / {families_count}",
        "Nhu cầu tăng trên 10%" if growing_families else "Nhu cầu tương đối ổn định",
        "🔔",
    ),
]

card_html = []
for index, (label, value, note, icon) in enumerate(cards, start=1):
    card_html.append(
        f'<div class="dashboard-kpi dashboard-kpi-{index}">'
        f'<span class="dashboard-kpi-label">{html.escape(label)}</span>'
        f'<strong class="dashboard-kpi-value">{html.escape(value)}</strong>'
        f'<span class="dashboard-kpi-note">{html.escape(note)}</span>'
        f'<span class="dashboard-kpi-icon" aria-hidden="true">{icon}</span>'
        "</div>"
    )

st.markdown(
    """
    <style>
    .dashboard-kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
        gap:16px; margin:8px 0 20px; }
    .dashboard-kpi { position:relative; overflow:hidden; display:flex; flex-direction:column;
        justify-content:center; min-height:122px; padding:18px 20px; border-radius:22px;
        color:#fff; box-shadow:0 12px 24px -10px rgba(15,23,42,.34); }
    .dashboard-kpi-1 { background:linear-gradient(135deg,#10b981,#047857); }
    .dashboard-kpi-2 { background:linear-gradient(135deg,#38bdf8,#0369a1); }
    .dashboard-kpi-3 { background:linear-gradient(135deg,#a78bfa,#6d28d9); }
    .dashboard-kpi-4 { background:linear-gradient(135deg,#fb7185,#be123c); }
    .dashboard-kpi-label { z-index:1; font-size:13px; font-weight:700; opacity:.96; }
    .dashboard-kpi-value { z-index:1; margin-top:4px; font-size:clamp(24px,2.2vw,34px);
        font-weight:800; line-height:1.15; letter-spacing:-.03em; }
    .dashboard-kpi-note { z-index:1; margin-top:4px; font-size:12px; opacity:.95; }
    .dashboard-kpi-icon { position:absolute; right:4px; bottom:-13px; font-size:78px;
        opacity:.18; }
    .dashboard-chart-title { margin:0!important; }
    @media (max-width:1000px) {
        .dashboard-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    @media (max-width:540px) {
        .dashboard-kpi-grid { grid-template-columns:1fr; gap:10px; }
        .dashboard-kpi { min-height:102px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="dashboard-kpi-grid">{"".join(card_html)}</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Doanh số theo ngày: 60 ngày gần nhất và dự báo 16 ngày")
    if forecast is None or forecast.empty:
        model = st.session_state.get("model")
        scaler = st.session_state.get("scaler")
        scaled = st.session_state.get("full_scaled")
        if model is not None and scaler is not None and scaled is not None:
            st.caption(
                "Đường xanh thể hiện doanh số lịch sử. Tạo dự báo 16 ngày từ mô hình "
                "LSTM/GRU đã huấn luyện để thêm đường dự báo màu cam."
            )
            if st.button("Tạo dự báo 16 ngày", type="primary"):
                try:
                    values = forecast_future(
                        model,
                        scaled[-st.session_state.look_back :],
                        scaler,
                        16,
                    )
                    forecast = pd.DataFrame(
                        {
                            "date": pd.date_range(
                                history["date"].max() + pd.Timedelta(days=1),
                                periods=16,
                                freq="D",
                            ),
                            "forecast": values,
                        }
                    )
                    st.session_state.forecast_result = forecast
                    st.rerun()
                except ValueError as exc:
                    st.error(f"Không thể tạo dự báo: {exc}")
        else:
            st.caption(
                "Đường xanh thể hiện doanh số lịch sử. Huấn luyện mô hình LSTM/GRU "
                "để tạo dự báo cho các ngày tiếp theo."
            )
    else:
        st.caption(
            f"Dữ liệu mô hình {st.session_state.model_config['type']}; "
            "dải màu cam biểu thị độ biến động sai số trên tập kiểm định."
        )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["sales"],
            mode="lines",
            name="Thực tế",
            line={"color": "#0d9f6e", "width": 3, "shape": "spline", "smoothing": 0.5},
            hovertemplate="%{x|%d/%m}<br>Thực tế: %{y:,.0f}<extra></extra>",
        )
    )
    if forecast is not None and not forecast.empty:
        forecast_x = pd.concat(
            [history["date"].tail(1), forecast["date"]],
            ignore_index=True,
        )
        forecast_y = pd.concat(
            [history["sales"].tail(1), forecast["forecast"]],
            ignore_index=True,
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines",
                name="Dự báo",
                line={
                    "color": "#f97316",
                    "width": 3,
                    "dash": "dash",
                    "shape": "spline",
                    "smoothing": 0.5,
                },
                hovertemplate="%{x|%d/%m}<br>Dự báo: %{y:,.0f}<extra></extra>",
            )
        )

        if prediction is not None:
            residuals = (
                np.asarray(prediction["actual"], dtype=float)
                - np.asarray(prediction["predicted"], dtype=float)
            )
            finite_residuals = residuals[np.isfinite(residuals)]
            if finite_residuals.size > 1:
                error_band = float(np.std(finite_residuals, ddof=1))
                lower_bound = np.maximum(forecast["forecast"] - error_band, 0)
                upper_bound = forecast["forecast"] + error_band
                figure.add_trace(
                    go.Scatter(
                        x=pd.concat([forecast["date"], forecast["date"][::-1]], ignore_index=True),
                        y=pd.concat([upper_bound, lower_bound[::-1]], ignore_index=True),
                        fill="toself",
                        fillcolor="rgba(249,115,22,.16)",
                        line={"color": "rgba(249,115,22,0)"},
                        hoverinfo="skip",
                        name="Độ biến động sai số",
                        showlegend=False,
                    )
                )
                figure.data = (figure.data[0], figure.data[2], figure.data[1])
        figure.add_vline(
            x=forecast["date"].min(),
            line={"color": "#cbd5e1", "width": 1, "dash": "dot"},
        )

    figure.update_layout(
        height=365,
        margin={"l": 56, "r": 18, "t": 26, "b": 42},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.12,
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={
            "gridcolor": "#e2e8f0",
            "tickformat": "%d/%m",
            "nticks": 10,
            "tickfont": {"size": 11},
        },
        yaxis={
            "gridcolor": "#e2e8f0",
            "tickformat": ",.0f",
            "tickfont": {"size": 11},
            "rangemode": "tozero",
        },
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
        key="dashboard_sales_forecast",
    )
