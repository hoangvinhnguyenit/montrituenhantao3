# -*- coding: utf-8 -*-
"""Generate, inspect and export multi-step forecasts."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import configure_page, get_active_sales, initialize_session_state
from model import forecast_future
from theme import page_intro


configure_page("Dự báo doanh số")
initialize_session_state()

page_intro(
    "Dự báo doanh số",
    "So sánh doanh số thực tế với dự báo cho 16 ngày tiếp theo.",
)

model = st.session_state.model
scaler = st.session_state.scaler
scaled = st.session_state.get("full_scaled")
series = st.session_state.get("training_series")
if model is None or scaler is None or scaled is None or series is None:
    with st.container(border=True):
        st.warning("Hãy huấn luyện mô hình ở trang **Mô hình huấn luyện** trước.")
    st.stop()

config = st.session_state.model_config
model_info = (
    f"Mô hình {config['type']} · cửa sổ nhìn lại {config['look_back']} ngày · "
    f"dữ liệu đến {series['date'].max():%d/%m/%Y}"
)
st.caption(model_info)

controls_col, status_col = st.columns([1, 3], vertical_alignment="center")
with controls_col:
    horizon = st.select_slider(
        "Số ngày dự báo",
        options=[7, 14, 16, 30, 60, 90],
        value=16,
    )
    create_forecast = st.button(
        "Tạo dự báo",
        type="primary",
        width="stretch",
    )
with status_col:
    st.caption(
        "Mô hình dự báo chuỗi đã chọn. Biểu đồ nhóm hàng bên dưới phân bổ tổng "
        "dự báo theo tỷ trọng bán gần nhất, không phải dự báo độc lập cho từng nhóm."
    )

if create_forecast:
    try:
        values = forecast_future(
            model,
            scaled[-st.session_state.look_back :],
            scaler,
            horizon,
        )
        dates = pd.date_range(
            series["date"].max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        st.session_state.forecast_result = pd.DataFrame(
            {"date": dates, "forecast": values}
        )
        st.session_state.inventory_plan = None
    except ValueError as exc:
        st.error(f"Không thể tạo dự báo: {exc}")

forecast = st.session_state.forecast_result
if forecast is not None and not forecast.empty:
    forecast = forecast.copy()
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    forecast["forecast"] = pd.to_numeric(forecast["forecast"], errors="coerce")
    forecast = forecast.dropna(subset=["date", "forecast"]).sort_values("date")

if forecast is None or forecast.empty:
    with st.container(border=True):
        st.info("Chọn số ngày dự báo rồi bấm **Tạo dự báo** để xem kết quả.")
    st.stop()

forecast_days = len(forecast)
recent_history = series.copy()
recent_history["date"] = pd.to_datetime(recent_history["date"], errors="coerce")
recent_history["sales"] = pd.to_numeric(recent_history["sales"], errors="coerce")
recent_history = (
    recent_history.dropna(subset=["date", "sales"])
    .sort_values("date")
    .tail(60)
)
if recent_history.empty:
    st.error("Không có dữ liệu lịch sử hợp lệ để hiển thị biểu đồ.")
    st.stop()

history_total = float(recent_history.tail(forecast_days)["sales"].sum())
forecast_total = float(forecast["forecast"].sum())
change = (
    (forecast_total / history_total - 1) * 100
    if history_total > 0
    else None
)
metric_cols = st.columns(3)
metric_cols[0].metric(
    f"Doanh số {forecast_days} ngày gần nhất",
    f"{history_total:,.0f}",
)
metric_cols[1].metric(
    f"Dự báo {forecast_days} ngày tới",
    f"{forecast_total:,.0f}",
)
metric_cols[2].metric(
    "Thay đổi so với kỳ trước",
    f"{change:+.1f}%" if change is not None else "—",
)

with st.container(border=True):
    st.subheader("Doanh số theo ngày: 60 ngày gần nhất và dự báo")
    st.caption(
        f"{model_info}. Đường liền là doanh số thực tế; đường nét đứt là dự báo."
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent_history["date"],
            y=recent_history["sales"],
            mode="lines",
            name="Thực tế",
            line={"color": "#0d9f6e", "width": 3, "shape": "spline", "smoothing": 0.5},
            hovertemplate="%{x|%d/%m/%Y}<br>Thực tế: %{y:,.0f}<extra></extra>",
        )
    )
    forecast_x = pd.concat(
        [recent_history["date"].tail(1), forecast["date"]],
        ignore_index=True,
    )
    forecast_y = pd.concat(
        [recent_history["sales"].tail(1), forecast["forecast"]],
        ignore_index=True,
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="lines",
            name="Dự báo",
            line={"color": "#f97316", "width": 3, "dash": "dash", "shape": "spline", "smoothing": 0.5},
            fill="tozeroy",
            fillcolor="rgba(249,115,22,.08)",
            hovertemplate="%{x|%d/%m/%Y}<br>Dự báo: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_vline(
        x=forecast["date"].min(),
        line={"color": "#cbd5e1", "width": 1, "dash": "dot"},
    )
    figure.update_layout(
        height=350,
        margin={"l": 55, "r": 18, "t": 22, "b": 42},
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
        key="sales_forecast_chart",
    )

sales = get_active_sales()
training_filter = st.session_state.get("training_filter") or {}
if sales is not None and not sales.empty and "family" in sales.columns:
    scoped_sales = sales.copy()
    selected_store = training_filter.get("store_nbr", "Tất cả cửa hàng")
    selected_family = training_filter.get("family", "Tất cả nhóm hàng")
    if selected_store != "Tất cả cửa hàng" and "store_nbr" in scoped_sales:
        scoped_sales = scoped_sales[
            scoped_sales["store_nbr"].astype(str) == str(selected_store)
        ]
    if selected_family != "Tất cả nhóm hàng":
        scoped_sales = scoped_sales[
            scoped_sales["family"].astype(str) == str(selected_family)
        ]
    scoped_sales["date"] = pd.to_datetime(scoped_sales["date"], errors="coerce")
    scoped_sales["sales"] = pd.to_numeric(scoped_sales["sales"], errors="coerce")
    first_forecast_date = forecast["date"].min()
    period_start = first_forecast_date - pd.Timedelta(days=forecast_days)
    sales_by_family = (
        scoped_sales.loc[
            scoped_sales["date"].ge(period_start)
            & scoped_sales["date"].lt(first_forecast_date)
        ]
        .dropna(subset=["family", "sales"])
        .groupby("family", as_index=False)["sales"]
        .sum()
    )
    recent_family_total = float(sales_by_family["sales"].sum())
    if recent_family_total > 0:
        recent_by_family = sales_by_family.nlargest(10, "sales").copy()
        recent_by_family["forecast"] = (
            recent_by_family["sales"] / recent_family_total * forecast_total
        )
        recent_by_family = recent_by_family.sort_values(
            "sales", ascending=True
        )

        with st.container(border=True):
            st.subheader(
                f"Top {len(recent_by_family)} nhóm hàng: {forecast_days} ngày qua "
                "so với ước tính"
            )
            st.caption(
                "Ước tính nhu cầu từng nhóm được phân bổ theo tỷ trọng bán trong "
                f"{forecast_days} ngày gần nhất; tổng các nhóm hiển thị có thể "
                "không bằng toàn bộ dự báo."
            )
            family_figure = go.Figure()
            family_figure.add_trace(
                go.Bar(
                    x=recent_by_family["sales"],
                    y=recent_by_family["family"],
                    orientation="h",
                    name=f"{forecast_days} ngày qua",
                    marker={"color": "#10b981", "cornerradius": 5},
                    hovertemplate="%{y}<br>Đã bán: %{x:,.0f}<extra></extra>",
                )
            )
            family_figure.add_trace(
                go.Bar(
                    x=recent_by_family["forecast"],
                    y=recent_by_family["family"],
                    orientation="h",
                    name=f"Dự báo {forecast_days} ngày",
                    marker={"color": "#f97316", "cornerradius": 5},
                    hovertemplate="%{y}<br>Ước tính: %{x:,.0f}<extra></extra>",
                )
            )
            family_figure.update_layout(
                height=max(320, 42 * len(recent_by_family) + 95),
                barmode="group",
                margin={"l": 135, "r": 18, "t": 32, "b": 42},
                legend={
                    "orientation": "h",
                    "y": 1.12,
                    "x": 0.5,
                    "xanchor": "center",
                },
                xaxis={
                    "gridcolor": "#e2e8f0",
                    "tickformat": ",.0f",
                    "tickfont": {"size": 11},
                },
                yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
            )
            st.plotly_chart(
                family_figure,
                width="stretch",
                config={"displayModeBar": False},
                key="family_forecast_chart",
            )
    else:
        st.info("Không có doanh số theo nhóm hàng trong kỳ gần nhất để so sánh.")

cols = st.columns(3)
cols[0].metric("Trung bình dự báo/ngày", f"{forecast['forecast'].mean():,.1f}")
peak = forecast.loc[forecast["forecast"].idxmax()]
cols[1].metric("Ngày dự báo cao nhất", f"{peak['date']:%d/%m/%Y}")
cols[2].download_button(
    "Tải dự báo CSV",
    forecast.to_csv(index=False).encode("utf-8-sig"),
    "du_bao_doanh_so.csv",
    "text/csv",
    width="stretch",
)

with st.expander("Dự báo cho các dòng trong bảng test"):
    test_data = st.session_state.get("test_df")
    selected_store = training_filter.get("store_nbr", "Tất cả cửa hàng")
    selected_family = training_filter.get("family", "Tất cả nhóm hàng")
    can_map_test_rows = (
        test_data is not None
        and selected_store != "Tất cả cửa hàng"
        and selected_family != "Tất cả nhóm hàng"
    )

    if test_data is None or test_data.empty:
        st.info("Chưa nạp bảng test.csv.")
    elif not can_map_test_rows:
        st.info(
            "Để xuất dự báo có khóa `id`, hãy huấn luyện riêng một cửa hàng và "
            "một nhóm hàng. Mô hình tổng hợp nhiều chuỗi không thể gán dự báo "
            "chính xác cho từng dòng test."
        )
    else:
        test_rows = test_data[
            (test_data["store_nbr"].astype(str) == str(selected_store))
            & (test_data["family"].astype(str) == str(selected_family))
        ].copy()
        test_rows = test_rows.dropna(subset=["date"]).sort_values("date")
        if test_rows.empty:
            st.info("Không có dòng test khớp cửa hàng/nhóm hàng đã dùng để huấn luyện.")
        else:
            st.caption(
                f"{len(test_rows):,} dòng test khớp cửa hàng {selected_store}, "
                f"nhóm hàng {selected_family}."
            )
            if st.button("Dự báo các dòng test"):
                first_date = pd.Timestamp(series["date"].max()) + pd.Timedelta(days=1)
                last_test_date = pd.Timestamp(test_rows["date"].max())
                days_to_predict = (
                    last_test_date - pd.Timestamp(series["date"].max())
                ).days
                if days_to_predict < 1:
                    st.error("Ngày trong test.csv không nằm sau ngày cuối tập huấn luyện.")
                else:
                    try:
                        values = forecast_future(
                            model,
                            scaled[-st.session_state.look_back :],
                            scaler,
                            days_to_predict,
                        )
                        date_predictions = pd.DataFrame(
                            {
                                "date": pd.date_range(
                                    first_date,
                                    periods=days_to_predict,
                                    freq="D",
                                ),
                                "sales": values,
                            }
                        )
                        test_result = test_rows.merge(
                            date_predictions,
                            on="date",
                            how="left",
                            validate="many_to_one",
                        )
                        output = test_result.dropna(subset=["sales"])[["id", "sales"]]
                        template = st.session_state.source_tables.get(
                            "sample_submission"
                        )
                        if template is not None and "id" in template:
                            output = template[["id"]].merge(
                                output,
                                on="id",
                                how="inner",
                                validate="one_to_one",
                            )
                        st.session_state.submission_result = output
                    except ValueError as exc:
                        st.error(f"Không thể dự báo test: {exc}")

            submission = st.session_state.get("submission_result")
            if submission is not None:
                st.dataframe(submission.head(30), width="stretch", hide_index=True)
                st.download_button(
                    "Tải dự báo test theo id",
                    submission.to_csv(index=False).encode("utf-8-sig"),
                    "test_predictions.csv",
                    "text/csv",
                )
