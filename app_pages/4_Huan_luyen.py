# -*- coding: utf-8 -*-
"""Display the precomputed Favorita forecasting benchmark report."""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import configure_page, initialize_session_state
from theme import page_intro


configure_page("Đánh giá mô hình")
initialize_session_state()

page_intro(
    "Đánh giá mô hình",
    "Xem kết quả backtest, mức độ sai số và tác động của từng nguồn dữ liệu.",
)

JOIN_ROWS = [
    (
        "train.csv",
        "—",
        "Nền: doanh số, khuyến mãi, 1.782 chuỗi (cửa hàng × nhóm hàng)",
        "3.000.888 dòng · 1684 ngày",
    ),
    ("stores.csv", "store_nbr", "Loại, cụm, thành phố, tỉnh", "54/54 cửa hàng khớp"),
    (
        "transactions.csv",
        "ngày + store_nbr",
        "Giao dịch TB 7/28/56 ngày trước điểm dự báo",
        "83.488 dòng · phủ 92% cửa hàng-ngày",
    ),
    (
        "oil.csv",
        "ngày",
        "Giá dầu ngày dự báo, giá hiện tại, thay đổi 28 ngày",
        "43 ngày thiếu được điền bằng giá gần nhất",
    ),
    (
        "holidays_events.csv",
        "ngày + quốc gia/tỉnh/thành phố",
        "Lễ quốc gia, lễ vùng (theo tỉnh), lễ địa phương (theo thành phố), "
        "sự kiện, ngày làm bù",
        "Tên vùng/địa phương khớp 100% (22/22 và 141/141 dòng). "
        "134/136 dòng lễ vùng và địa phương được dùng; 2 dòng bị loại đúng "
        "vì đã chuyển ngày. 393 ô ngày × cửa hàng được gắn cờ, 51/54 cửa hàng "
        "chịu ảnh hưởng.",
    ),
    (
        "test.csv",
        "ngày + store_nbr + family",
        "Khuyến mãi của 16 ngày tới; xác định chuỗi cần dự báo",
        "28.512 dòng · 1782 chuỗi, khớp 100% train",
    ),
    (
        "sample_submission.csv",
        "id",
        "Xuất kết quả đúng định dạng id, sales",
        "28.512 dòng · ID liên tục",
    ),
]

BACKTEST_DATES = [
    "31/07",
    "01/08",
    "02/08",
    "03/08",
    "04/08",
    "05/08",
    "06/08",
    "07/08",
    "08/08",
    "09/08",
    "10/08",
    "11/08",
    "12/08",
    "13/08",
    "14/08",
    "15/08",
]
BACKTEST_SERIES = {
    "Thực tế": [886, 989, 965, 728, 828, 966, 1050, 797, 718, 734, 651, 826, 793, 866, 761, 763],
    "Mô hình": [853, 873, 858, 714, 825, 1024, 1066, 774, 686, 710, 606, 868, 815, 966, 747, 711],
    "Cơ sở": [824, 751, 779, 656, 798, 1024, 1115, 824, 751, 779, 656, 798, 1024, 1115, 824, 751],
}

ABLATION_RESULTS = [
    ("oil.csv", 0.4179),
    ("transactions.csv", 0.4304),
    ("holidays_events.csv", 0.4230),
    ("stores.csv", 0.4224),
    ("khuyến mãi", 0.4711),
    ("Chỉ train", 0.4202),
]
FULL_MODEL_RMSLE = 0.4222


def _format_integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _render_join_table() -> str:
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(value)}</td>" for value in row)
        + "</tr>"
        for row in JOIN_ROWS
    )
    return (
        '<div class="benchmark-table-wrap"><table class="benchmark-table">'
        "<thead><tr><th>Bảng dữ liệu</th><th>Khóa nối</th>"
        "<th>Đặc trưng đưa vào mô hình</th><th>Kết quả nối</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


st.markdown(
    """
    <style>
    .benchmark-table-wrap { overflow-x: auto; }
    .benchmark-table { width: 100%; min-width: 680px; border-collapse: collapse;
        table-layout: fixed; font-size: 12px; line-height: 1.45; color: #334155; }
    .benchmark-table th { padding: 9px 8px; text-align: left; color: #64748b;
        font-size: 11px; font-weight: 700; border-bottom: 2px solid #e2e8f0; }
    .benchmark-table td { padding: 9px 8px; vertical-align: top;
        overflow-wrap: anywhere; border-bottom: 1px solid #e2e8f0; }
    .benchmark-table th:nth-child(1), .benchmark-table td:nth-child(1) { width: 13%; }
    .benchmark-table th:nth-child(2), .benchmark-table td:nth-child(2) { width: 13%; }
    .benchmark-table th:nth-child(3), .benchmark-table td:nth-child(3) { width: 20%; }
    .benchmark-table th:nth-child(4), .benchmark-table td:nth-child(4) { width: 54%; }
    .benchmark-table td:first-child { color: #0f172a; font-weight: 700; }
    @media (max-width: 700px) {
        .benchmark-table { min-width: 740px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    heading_col, download_col = st.columns([5, 1], vertical_alignment="top")
    with heading_col:
        st.subheader("Mô hình dự báo và kết nối dữ liệu")
        st.caption(
            "Benchmark Favorita cố định: Gradient Boosting (scikit-learn), "
            "dự báo trực tiếp 16 bước; 41 đặc trưng, "
            f"{_format_integer(1_037_338)} mẫu huấn luyện từ các điểm cắt "
            "hằng tuần, 350 cây. Huấn luyện đến 30/07/2017, dự báo 31/07–15/08."
        )
    with download_col:
        submission = st.session_state.get("submission_result")
        if isinstance(submission, pd.DataFrame) and not submission.empty:
            st.download_button(
                "Tải submission.csv",
                submission.to_csv(index=False).encode("utf-8-sig"),
                "submission.csv",
                "text/csv",
                width="stretch",
            )
        else:
            st.download_button(
                "Tải submission.csv",
                data=b"id,sales\n",
                file_name="submission.csv",
                mime="text/csv",
                disabled=True,
                width="stretch",
                help="Hãy tạo dự báo test ở trang Mô hình huấn luyện trước.",
            )

    st.markdown(_render_join_table(), unsafe_allow_html=True)

backtest_col, ablation_col = st.columns([2, 1], gap="small")
with backtest_col:
    with st.container(border=True):
        st.subheader("Backtest 16 ngày (31/07 – 15/08/2017), toàn quốc")
        st.caption(
            "Mô hình huấn luyện đến 30/07, dự báo 16 ngày kế tiếp rồi so với "
            "thực tế (nghìn đơn vị/ngày)."
        )
        backtest = go.Figure()
        backtest_styles = {
            "Thực tế": {"color": "#0d9f6e", "width": 3, "dash": "solid"},
            "Mô hình": {"color": "#f97316", "width": 3, "dash": "solid"},
            "Cơ sở": {"color": "#94a3b8", "width": 2, "dash": "dash"},
        }
        for name, values in BACKTEST_SERIES.items():
            style = backtest_styles[name]
            backtest.add_trace(
                go.Scatter(
                    x=BACKTEST_DATES,
                    y=values,
                    mode="lines+markers" if name != "Cơ sở" else "lines",
                    name=name,
                    line={"color": style["color"], "width": style["width"], "dash": style["dash"]},
                    marker={"size": 5},
                )
            )
        backtest.update_layout(
            height=265,
            margin={"l": 38, "r": 12, "t": 32, "b": 35},
            legend={"orientation": "h", "y": 1.16, "x": 0.5, "xanchor": "center"},
            hovermode="x unified",
            xaxis={"gridcolor": "#e2e8f0", "tickfont": {"size": 10}},
            yaxis={"gridcolor": "#e2e8f0", "range": [580, 1200], "tickfont": {"size": 10}},
        )
        st.plotly_chart(
            backtest,
            width="stretch",
            config={"displayModeBar": False},
            key="benchmark_backtest",
        )

with ablation_col:
    with st.container(border=True):
        st.subheader("Thử bỏ từng bảng dữ liệu")
        st.caption(
            "Huấn luyện lại không có bảng đó. Dương nghĩa là bỏ bảng thì "
            "RMSLE tăng (mô hình tệ hơn)."
        )
        ablation_labels = [row[0] for row in ABLATION_RESULTS]
        impact = [
            (rmsle / FULL_MODEL_RMSLE - 1) * 100
            for _, rmsle in ABLATION_RESULTS
        ]
        ablation = go.Figure(
            go.Bar(
                x=impact,
                y=ablation_labels,
                orientation="h",
                marker={
                    "color": [
                        "#10b981" if value > 1 else "#f59e0b" if value < -1 else "#94a3b8"
                        for value in impact
                    ],
                    "cornerradius": 5,
                },
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            )
        )
        ablation.update_layout(
            height=265,
            margin={"l": 112, "r": 12, "t": 18, "b": 35},
            showlegend=False,
            xaxis={
                "gridcolor": "#e2e8f0",
                "range": [-5, 15],
                "ticksuffix": "%",
                "tickfont": {"size": 10},
                "zeroline": True,
                "zerolinecolor": "#94a3b8",
            },
            yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
        )
        st.plotly_chart(
            ablation,
            width="stretch",
            config={"displayModeBar": False},
            key="benchmark_ablation",
        )
