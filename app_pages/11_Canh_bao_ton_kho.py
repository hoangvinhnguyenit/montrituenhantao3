# -*- coding: utf-8 -*-
"""User-entered stock alerts based on recent product-family demand."""

import numpy as np
import pandas as pd
import streamlit as st

from app_common import (
    configure_page,
    get_active_sales,
    initialize_session_state,
    show_data_required,
)
from inventory import build_inventory_plan
from theme import page_intro


configure_page("Cảnh báo tồn kho")
initialize_session_state()

page_intro(
    "Cảnh báo tồn kho",
    "Đối chiếu tồn kho thực tế với nhu cầu gần đây để nhận diện nhóm hàng cần bổ sung.",
)

sales = get_active_sales()
if show_data_required(sales):
    st.stop()
if not {"family", "date", "sales"}.issubset(sales.columns):
    st.error("Cần có các cột nhóm hàng, ngày và doanh số để lập cảnh báo.")
    st.stop()

history = sales[["family", "date", "sales"]].copy()
history["date"] = pd.to_datetime(history["date"], errors="coerce")
history["sales"] = pd.to_numeric(history["sales"], errors="coerce")
history = history.dropna(subset=["family", "date", "sales"])
if history.empty:
    st.warning("Không có dữ liệu lịch sử hợp lệ để ước tính nhu cầu.")
    st.stop()

as_of_date = history["date"].max().normalize()
window_start = as_of_date - pd.Timedelta(days=27)
history = history[history["date"].between(window_start, as_of_date)]
families = sorted(history["family"].astype(str).unique().tolist())
if not families:
    st.info("Không tìm thấy nhóm hàng có doanh số trong 28 ngày gần nhất.")
    st.stop()

control_col, service_col = st.columns(2)
lead_time = control_col.number_input(
    "Thời gian bổ sung hàng (ngày)",
    min_value=1,
    max_value=365,
    value=7,
    step=1,
)
service_level = service_col.slider(
    "Mức độ phục vụ",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01,
)

st.caption(
    "Nhập tồn kho thực tế cho từng nhóm hàng. ROP và tồn kho an toàn được ước "
    "tính từ doanh số 28 ngày gần nhất; nguồn dữ liệu hiện không lưu tồn kho "
    "thực tế nên các ô để trống sẽ không bị xem là bằng 0."
)
stock_inputs = pd.DataFrame(
    {
        "Nhóm hàng": families,
        "Tồn kho hiện tại": [None] * len(families),
    }
)
edited_stock = st.data_editor(
    stock_inputs,
    hide_index=True,
    width="stretch",
    key="inventory_alert_stock_editor",
    column_config={
        "Nhóm hàng": st.column_config.TextColumn(disabled=True),
        "Tồn kho hiện tại": st.column_config.NumberColumn(
            min_value=0,
            step=1,
            format="%.0f",
        ),
    },
)

calendar = pd.date_range(window_start, as_of_date, freq="D")
alerts = []
for _, row in edited_stock.iterrows():
    stock_value = pd.to_numeric(row["Tồn kho hiện tại"], errors="coerce")
    if pd.isna(stock_value):
        continue

    family_history = history[history["family"].astype(str) == row["Nhóm hàng"]]
    daily_demand = (
        family_history.groupby("date")["sales"]
        .sum()
        .reindex(calendar, fill_value=0)
        .to_numpy(dtype=float)
    )
    expected_demand = float(np.mean(daily_demand))
    forecast = np.full(int(lead_time), expected_demand, dtype=float)
    try:
        plan = build_inventory_plan(
            forecast,
            current_stock=float(stock_value),
            lead_time=int(lead_time),
            service_level=float(service_level),
            historical_demand=daily_demand,
        )
    except ValueError as exc:
        st.error(f"Không thể tính cảnh báo cho {row['Nhóm hàng']}: {exc}")
        continue
    alerts.append(
        {
            "Nhóm hàng": row["Nhóm hàng"],
            "Tồn kho hiện tại": float(stock_value),
            "Nhu cầu TB/ngày": plan["avg_daily_demand"],
            "Tồn kho an toàn": plan["safety_stock"],
            "Điểm đặt hàng lại (ROP)": plan["reorder_point"],
            "Đề xuất bổ sung": plan["suggested_order_qty"],
            "Trạng thái": "Cần bổ sung" if plan["need_reorder"] else "Đang đáp ứng",
        }
    )

if alerts:
    alert_table = pd.DataFrame(alerts).sort_values(
        ["Trạng thái", "Điểm đặt hàng lại (ROP)"],
        ascending=[True, False],
    )
    needs_order = alert_table[alert_table["Trạng thái"] == "Cần bổ sung"]
    headline = st.columns(3)
    headline[0].metric("Nhóm hàng đã kiểm tra", f"{len(alert_table):,}")
    headline[1].metric("Cần bổ sung", f"{len(needs_order):,}")
    headline[2].metric("Ngày dữ liệu gần nhất", f"{as_of_date:%d/%m/%Y}")

    if not needs_order.empty:
        st.warning(
            "Có nhóm hàng có tồn kho thấp hơn hoặc bằng điểm đặt hàng lại. "
            "Hãy kiểm tra tồn kho thực tế trước khi tạo đơn."
        )
    else:
        st.success("Các nhóm hàng đã nhập tồn kho đang trên điểm đặt hàng lại.")

    st.dataframe(
        alert_table.style.format(
            {
                "Tồn kho hiện tại": "{:,.0f}",
                "Nhu cầu TB/ngày": "{:,.1f}",
                "Tồn kho an toàn": "{:,.1f}",
                "Điểm đặt hàng lại (ROP)": "{:,.1f}",
                "Đề xuất bổ sung": "{:,.1f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.info("Nhập tồn kho hiện tại ở ít nhất một dòng để xem cảnh báo.")
