# -*- coding: utf-8 -*-
"""Exploratory sales analysis, reference library and supply-chain assistant."""

import json
import re
import unicodedata
from pathlib import Path

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
from utils import aggregate_daily


CITY_COORDINATES = {
    "Ambato": (-1.24, -78.62),
    "Babahoyo": (-1.80, -79.53),
    "Cayambe": (0.04, -78.15),
    "Cuenca": (-2.90, -79.00),
    "Daule": (-1.86, -79.98),
    "El Carmen": (-0.27, -79.43),
    "Esmeraldas": (0.96, -79.65),
    "Guaranda": (-1.59, -79.00),
    "Guayaquil": (-2.17, -79.92),
    "Ibarra": (0.35, -78.12),
    "Latacunga": (-0.93, -78.62),
    "Libertad": (-2.23, -80.91),
    "Loja": (-3.99, -79.20),
    "Machala": (-3.26, -79.96),
    "Manta": (-0.97, -80.71),
    "Playas": (-2.63, -80.39),
    "Puyo": (-1.49, -78.00),
    "Quevedo": (-1.02, -79.46),
    "Quito": (-0.18, -78.47),
    "Riobamba": (-1.67, -78.65),
    "Salinas": (-2.21, -80.96),
    "Santo Domingo": (-0.25, -79.17),
}
KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "knowledge_base.json"
WEEKDAY_LABELS = {
    0: "Thứ 2",
    1: "Thứ 3",
    2: "Thứ 4",
    3: "Thứ 5",
    4: "Thứ 6",
    5: "Thứ 7",
    6: "Chủ nhật",
}


def _load_knowledge_base() -> list[dict]:
    try:
        content = KNOWLEDGE_PATH.read_text(encoding="utf-8")
        knowledge = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Không thể đọc thư viện kiến thức tại {KNOWLEDGE_PATH}: {exc}")
        return []
    if not isinstance(knowledge, list):
        st.error("Thư viện kiến thức cần có định dạng danh sách câu hỏi.")
        return []
    return [entry for entry in knowledge if isinstance(entry, dict)]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized.replace("đ", "d")


def _rank_questions(question: str, knowledge: list[dict]) -> list[tuple[float, dict]]:
    normalized_question = _normalize_text(question)
    question_words = set(re.findall(r"[a-z0-9]+", normalized_question))
    ranked = []
    for entry in knowledge:
        keywords = entry.get("k", [])
        if not isinstance(keywords, list):
            keywords = []
        phrases = [str(entry.get("q", "")), *(str(word) for word in keywords)]
        normalized_phrases = [_normalize_text(phrase) for phrase in phrases]
        entry_words = set(
            re.findall(r"[a-z0-9]+", " ".join(normalized_phrases))
        )
        overlap = len(question_words & entry_words) / max(len(question_words), 1)
        phrase_match = any(
            phrase and phrase in normalized_question for phrase in normalized_phrases
        )
        ranked.append((overlap + (1.0 if phrase_match else 0.0), entry))
    return sorted(ranked, key=lambda result: result[0], reverse=True)


def _build_store_summary(filtered: pd.DataFrame) -> pd.DataFrame:
    daily_columns = {"store_nbr", "date", "sales"}
    if not daily_columns.issubset(filtered.columns):
        return pd.DataFrame()

    daily_aggregations = {"daily_sales": ("sales", "sum")}
    if "transactions" in filtered:
        daily_aggregations["transactions"] = ("transactions", "max")
    daily_store = filtered.groupby(
        ["store_nbr", "date"], as_index=False, dropna=False
    ).agg(**daily_aggregations)
    store_aggregations = {
        "total_sales": ("daily_sales", "sum"),
        "avg_daily_sales": ("daily_sales", "mean"),
        "active_days": ("date", "nunique"),
    }
    if "transactions" in daily_store:
        store_aggregations["transactions"] = ("transactions", "sum")
    summary = daily_store.groupby(
        "store_nbr", as_index=False, dropna=False
    ).agg(**store_aggregations)
    if "onpromotion" in filtered.columns:
        promotion_rate = (
            filtered.assign(_promoted=filtered["onpromotion"].fillna(0).gt(0))
            .groupby("store_nbr", as_index=False)["_promoted"]
            .mean()
            .rename(columns={"_promoted": "promotion_rate"})
        )
        summary = summary.merge(promotion_rate, on="store_nbr", how="left")

    dimensions = st.session_state.get("source_tables", {}).get("stores")
    metadata_columns = ["store_nbr", "city", "state", "type", "cluster"]
    fact_metadata = [
        column
        for column in metadata_columns
        if column in filtered.columns and column != "store_nbr"
    ]
    fact_metadata_frame = filtered[["store_nbr", *fact_metadata]].drop_duplicates(
        "store_nbr"
    )
    fact_metadata_frame["store_nbr"] = fact_metadata_frame["store_nbr"].astype(
        "string"
    )
    if isinstance(dimensions, pd.DataFrame) and "store_nbr" in dimensions:
        metadata = dimensions[
            [column for column in metadata_columns if column in dimensions.columns]
        ].copy()
        metadata["store_nbr"] = metadata["store_nbr"].astype("string")
        missing_columns = [
            column
            for column in fact_metadata_frame.columns
            if column not in metadata.columns
        ]
        if missing_columns:
            metadata = metadata.merge(
                fact_metadata_frame[["store_nbr", *missing_columns]],
                on="store_nbr",
                how="left",
                validate="one_to_one",
            )
    else:
        metadata = fact_metadata_frame

    if "store_nbr" in metadata:
        summary["store_nbr"] = summary["store_nbr"].astype("string")
        metadata["store_nbr"] = metadata["store_nbr"].astype("string")
        metadata = metadata.drop_duplicates("store_nbr")
        missing_columns = [
            column
            for column in metadata.columns
            if column == "store_nbr" or column not in summary.columns
        ]
        summary = summary.merge(
            metadata[missing_columns], on="store_nbr", how="left", validate="one_to_one"
        )
    return summary.sort_values("total_sales", ascending=False).reset_index(drop=True)


def _format_store_table(stores: pd.DataFrame) -> pd.DataFrame:
    display = stores.copy()
    labels = {
        "store_nbr": "Cửa hàng",
        "city": "Thành phố",
        "state": "Tỉnh / bang",
        "type": "Loại",
        "cluster": "Cụm",
        "total_sales": "Tổng doanh số",
        "avg_daily_sales": "TB doanh số/ngày",
        "active_days": "Số ngày",
        "transactions": "Giao dịch",
        "promotion_rate": "Tỷ lệ khuyến mãi",
    }
    display = display.rename(columns=labels)
    if "Tỷ lệ khuyến mãi" in display:
        display["Tỷ lệ khuyến mãi"] = display["Tỷ lệ khuyến mãi"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "—"
        )
    for column in ("Tổng doanh số", "TB doanh số/ngày", "Giao dịch"):
        if column in display:
            display[column] = display[column].map(
                lambda value: f"{value:,.1f}" if pd.notna(value) else "—"
            )
    return display


def _answer_question(question: str, knowledge: list[dict]) -> dict:
    ranked = _rank_questions(question, knowledge)
    if not ranked or ranked[0][0] < 0.15:
        suggestions = [entry.get("q", "") for _, entry in ranked[:3]]
        suggestion_text = " • ".join(str(item) for item in suggestions if item)
        answer = (
            "Mình chưa tìm thấy câu trả lời đủ sát trong thư viện hiện có. "
            "Bạn có thể thử hỏi về: "
            f"{suggestion_text or 'dự báo, kiểm định mô hình, sai số hoặc tồn kho'}."
        )
        return {"role": "assistant", "content": answer, "source": None}

    _, entry = ranked[0]
    answer = str(entry.get("a", ""))
    if not answer:
        answer = "Mục kiến thức phù hợp chưa có nội dung trả lời."
    return {
        "role": "assistant",
        "content": answer,
        "source": str(entry.get("q", "")),
    }


def _speech_button(answer: str) -> None:
    encoded_answer = (
        json.dumps(answer, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    st.html(
        f"""
        <button id="read-answer" type="button"
          style="border:1px solid #cbd5e1;border-radius:8px;background:#fff;
          color:#334155;padding:8px 12px;font:500 13px sans-serif;cursor:pointer">
          🔊 Đọc phản hồi
        </button>
        <script>
          const answer = {encoded_answer};
          const button = document.getElementById("read-answer");
          button.addEventListener("click", () => {{
            if (!("speechSynthesis" in window)) {{
              button.textContent = "Trình duyệt chưa hỗ trợ đọc văn bản";
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(answer);
            utterance.lang = "vi-VN";
            window.speechSynthesis.speak(utterance);
          }});
        </script>
        """,
        width="stretch",
        unsafe_allow_javascript=True,
    )


configure_page("Phân tích dữ liệu")
initialize_session_state()
if "eda_chat_history" not in st.session_state:
    st.session_state.eda_chat_history = []

st.title("Phân tích dữ liệu")
st.caption(
    "Khám phá cơ cấu doanh số, hiệu quả cửa hàng và các yếu tố ảnh hưởng "
    "đến nhu cầu bán hàng."
)

sales = get_active_sales()
if show_data_required(sales):
    st.stop()

with st.expander("Bộ lọc phân tích", expanded=False):
    filtered = filter_controls(sales, key_prefix="eda")
    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    filtered = filtered.dropna(subset=["date", "sales"])
    if not filtered.empty:
        minimum_date = filtered["date"].min().date()
        maximum_date = filtered["date"].max().date()
        date_range = st.date_input(
            "Khoảng thời gian",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            key="eda_date_range",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            filtered = filtered[
                (filtered["date"] >= start_date) & (filtered["date"] < end_date)
            ]

if filtered.empty:
    st.warning("Không có dòng bán hàng hợp lệ trong phạm vi đã chọn.")

with st.container(border=True):
    st.subheader("Cơ cấu doanh số theo nhóm hàng")
    if "family" in filtered.columns and not filtered.empty:
        family_sales = (
            filtered.groupby("family", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )
        family_chart = px.pie(
            family_sales,
            names="family",
            values="sales",
            hole=0.58,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        family_chart.update_traces(
            textposition="inside",
            texttemplate="%{percent:.0%}",
            hovertemplate="%{label}<br>Doanh số: %{value:,.1f}<br>Tỷ trọng: %{percent}<extra></extra>",
        )
        family_chart.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", y=-0.08),
        )
        st.plotly_chart(family_chart, width="stretch", theme=None)
    else:
        st.info("Dữ liệu hiện tại chưa có cột nhóm hàng để lập biểu đồ cơ cấu.")

store_summary = _build_store_summary(filtered)
with st.container(border=True):
    st.subheader("Bản đồ cửa hàng tại Ecuador")
    st.caption(
        "Vị trí được suy ra từ tên thành phố trong bảng stores; màu và kích thước "
        "điểm phản ánh doanh số của phạm vi đang lọc."
    )
    if {"city", "store_nbr", "total_sales"}.issubset(store_summary.columns):
        map_data = store_summary.dropna(subset=["city"]).copy()
        map_data["_city_key"] = map_data["city"].astype(str).str.strip().str.casefold()
        coordinate_lookup = {
            city.casefold(): coordinates for city, coordinates in CITY_COORDINATES.items()
        }
        map_data["coordinates"] = map_data["_city_key"].map(coordinate_lookup)
        missing_cities = sorted(map_data.loc[map_data["coordinates"].isna(), "city"].unique())
        map_data = map_data.dropna(subset=["coordinates"])
        if not map_data.empty:
            map_data["latitude"] = map_data["coordinates"].map(lambda point: point[0])
            map_data["longitude"] = map_data["coordinates"].map(lambda point: point[1])
            city_sales = map_data.groupby(
                ["city", "latitude", "longitude"], as_index=False
            ).agg(
                total_sales=("total_sales", "sum"),
                store_count=("store_nbr", "nunique"),
            )
            map_chart = px.scatter_geo(
                city_sales,
                lat="latitude",
                lon="longitude",
                size="total_sales",
                color="total_sales",
                hover_name="city",
                hover_data={
                    "store_count": True,
                    "total_sales": ":,.1f",
                    "latitude": False,
                    "longitude": False,
                },
                color_continuous_scale="Tealgrn",
                size_max=32,
            )
            map_chart.update_traces(
                marker={"opacity": 0.85, "line": {"width": 1, "color": "white"}}
            )
            map_chart.update_geos(
                fitbounds="locations",
                projection_type="mercator",
                showland=True,
                landcolor="#eef5ee",
                showocean=True,
                oceancolor="#dceef4",
                showcountries=True,
                countrycolor="#bdc9c5",
                showcoastlines=True,
                coastlinecolor="#bdc9c5",
            )
            map_chart.update_layout(
                height=420,
                margin=dict(l=5, r=5, t=10, b=10),
                coloraxis_colorbar_title="Doanh số",
            )
            st.plotly_chart(map_chart, width="stretch", theme=None)
            if missing_cities:
                st.caption(
                    "Chưa có vị trí tham chiếu cho: " + ", ".join(missing_cities) + "."
                )
        else:
            st.info(
                "Chưa thể dựng bản đồ: tên thành phố trong dữ liệu không khớp "
                "với danh mục vị trí Ecuador hiện có."
            )
    elif "store_nbr" in store_summary:
        st.info(
            "Chưa có thông tin thành phố trong bảng stores. Bảng hiệu quả bên dưới "
            "vẫn hiển thị theo dữ liệu cửa hàng."
        )
    else:
        st.info("Dữ liệu hiện tại chưa có mã cửa hàng để hiển thị trên bản đồ.")

with st.container(border=True):
    st.subheader("Hiệu suất cửa hàng")
    if not store_summary.empty:
        st.caption(
            f"{len(store_summary):,} cửa hàng · sắp xếp theo tổng doanh số "
            "trong phạm vi bộ lọc."
        )
        st.dataframe(
            _format_store_table(store_summary.head(15)),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Chưa có dữ liệu đủ để tổng hợp hiệu suất theo cửa hàng.")

knowledge = _load_knowledge_base()
with st.container(border=True):
    st.subheader("Thư viện kiến thức thống kê và dự báo")
    st.caption(
        "Tra cứu nhanh kiến thức chuỗi thời gian, kiểm định mô hình, sai số "
        "và vận hành chuỗi cung ứng."
    )
    if knowledge:
        category_options = ["Tất cả chủ đề"] + sorted(
            {str(entry.get("c", "Khác")) for entry in knowledge}
        )
        filter_col, search_col = st.columns([1, 2])
        selected_category = filter_col.selectbox(
            "Chủ đề",
            category_options,
            key="eda_knowledge_category",
        )
        search_term = search_col.text_input(
            "Tìm câu hỏi hoặc từ khóa",
            placeholder="Ví dụ: sai số, tồn kho, khuyến mãi...",
            key="eda_knowledge_search",
        )
        visible_knowledge = knowledge
        if selected_category != "Tất cả chủ đề":
            visible_knowledge = [
                entry
                for entry in visible_knowledge
                if str(entry.get("c", "Khác")) == selected_category
            ]
        normalized_search = _normalize_text(search_term.strip())
        if normalized_search:
            visible_knowledge = [
                entry
                for entry in visible_knowledge
                if normalized_search
                in _normalize_text(
                    " ".join(
                        [
                            str(entry.get("q", "")),
                            str(entry.get("a", "")),
                            *[str(word) for word in entry.get("k", [])],
                        ]
                    )
                )
            ]

        page_size = 12
        page_count = max(1, (len(visible_knowledge) + page_size - 1) // page_size)
        page_col, count_col = st.columns([1, 3])
        page_number = page_col.selectbox(
            "Trang",
            range(1, page_count + 1),
            key="eda_knowledge_page",
        )
        count_col.caption(f"Tìm thấy {len(visible_knowledge):,} câu hỏi.")
        start_index = (page_number - 1) * page_size
        for entry in visible_knowledge[start_index : start_index + page_size]:
            title = str(entry.get("q", "Câu hỏi"))
            category = str(entry.get("c", "Khác"))
            with st.expander(f"{title}  ·  {category}"):
                st.write(str(entry.get("a", "Chưa có nội dung trả lời.")))
                keywords = entry.get("k", [])
                if isinstance(keywords, list) and keywords:
                    st.caption("Từ khóa: " + " · ".join(str(word) for word in keywords))
    else:
        st.info("Thư viện kiến thức chưa có nội dung để tra cứu.")

with st.container(border=True):
    assistant_title, speech_column, clear_column = st.columns([5, 1, 1])
    assistant_title.subheader("Trợ lý chuỗi cung ứng")
    latest_answer = next(
        (
            message["content"]
            for message in reversed(st.session_state.eda_chat_history)
            if message["role"] == "assistant"
        ),
        None,
    )
    if latest_answer:
        with speech_column:
            _speech_button(latest_answer)
    if clear_column.button("Xóa hội thoại", key="eda_clear_chat"):
        st.session_state.eda_chat_history = []
        st.rerun()
    st.caption(
        "Đặt câu hỏi bằng văn bản hoặc ghi âm; câu trả lời được tra cứu từ "
        "thư viện kiến thức của dự án."
    )

    history = st.session_state.eda_chat_history
    if history:
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("source"):
                    st.caption(f"Tham khảo: {message['source']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(
                "Xin chào! Mình có thể giúp tra cứu kiến thức về dự báo, "
                "đánh giá mô hình và quản trị tồn kho."
            )

    suggested_questions = [
        "Nên đánh giá mô hình dự báo bằng chỉ số nào?",
        "Làm thế nào để dự báo khi dữ liệu có nhiều số 0?",
        "Điểm đặt hàng lại (ROP) được tính như thế nào?",
    ]
    suggestion_columns = st.columns(len(suggested_questions))
    for index, (column, question) in enumerate(
        zip(suggestion_columns, suggested_questions)
    ):
        if column.button(
            question,
            key=f"eda_suggested_question_{index}",
            width="stretch",
        ):
            st.session_state.eda_chat_history.append(
                {"role": "user", "content": question}
            )
            st.session_state.eda_chat_history.append(
                _answer_question(question, knowledge)
            )
            st.rerun()

    chat_value = st.chat_input(
        "Hỏi về dự báo, chuỗi thời gian, sai số hoặc tồn kho...",
        key="eda_chat_input",
        accept_audio=True,
    )
    if chat_value is not None:
        if isinstance(chat_value, str):
            question = chat_value.strip()
            audio_file = None
        else:
            question = chat_value.text.strip()
            audio_file = chat_value.audio

        if question:
            st.session_state.eda_chat_history.append(
                {"role": "user", "content": question}
            )
            st.session_state.eda_chat_history.append(
                _answer_question(question, knowledge)
            )
            st.rerun()
        elif audio_file is not None:
            st.info(
                "Đã nhận bản ghi âm. Ứng dụng hiện chưa cấu hình dịch vụ "
                "chuyển giọng nói thành văn bản để phân tích câu hỏi."
            )
            st.audio(audio_file)

tab_trend, tab_structure, tab_features = st.tabs(
    ["Xu hướng & mùa vụ", "Nhóm hàng & cửa hàng", "Khuyến mãi & yếu tố ngoài"]
)
with tab_trend:
    if not filtered.empty:
        frequency = st.radio(
            "Tần suất biểu đồ",
            ["Ngày", "Tuần", "Tháng"],
            horizontal=True,
            key="eda_frequency",
        )
        period = {"Ngày": "D", "Tuần": "W", "Tháng": "MS"}[frequency]
        chart_data = aggregate_daily(filtered, freq=period)
        chart_data["Trung bình trượt"] = chart_data["sales"].rolling(
            7, min_periods=1
        ).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=chart_data["date"], y=chart_data["sales"], name="Doanh số")
        )
        fig.add_trace(
            go.Scatter(
                x=chart_data["date"],
                y=chart_data["Trung bình trượt"],
                name="Trung bình trượt 7 kỳ",
                line={"dash": "dash"},
            )
        )
        fig.update_layout(
            xaxis_title="Ngày", yaxis_title="Số lượng bán", hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")

        if "date" in filtered:
            weekday = filtered.assign(
                weekday=filtered["date"].dt.dayofweek.map(WEEKDAY_LABELS)
            )
            weekday_sales = weekday.groupby("weekday", as_index=False)["sales"].mean()
            weekday_sales["weekday"] = pd.Categorical(
                weekday_sales["weekday"],
                categories=list(WEEKDAY_LABELS.values()),
                ordered=True,
            )
            weekday_sales = weekday_sales.sort_values("weekday")
            st.plotly_chart(
                px.bar(
                    weekday_sales,
                    x="weekday",
                    y="sales",
                    title="Doanh số trung bình theo thứ",
                ),
                width="stretch",
            )
    else:
        st.info("Chọn phạm vi ngày có dữ liệu để xem xu hướng.")

with tab_structure:
    if not filtered.empty and "family" in filtered:
        family_sales = (
            filtered.groupby("family", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                family_sales.head(20),
                x="sales",
                y="family",
                orientation="h",
                title="Top nhóm hàng theo doanh số",
            ),
            width="stretch",
        )
    if not filtered.empty and "store_nbr" in filtered:
        store_sales = (
            filtered.groupby("store_nbr", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                store_sales,
                x="store_nbr",
                y="sales",
                title="Doanh số theo cửa hàng",
            ),
            width="stretch",
        )
    if not filtered.empty and {"date", "family"}.issubset(filtered.columns):
        heatmap = (
            filtered.assign(month=filtered["date"].dt.to_period("M").astype(str))
            .groupby(["month", "family"], as_index=False)["sales"]
            .sum()
        )
        st.plotly_chart(
            px.density_heatmap(
                heatmap,
                x="month",
                y="family",
                z="sales",
                title="Nhiệt đồ doanh số theo tháng và nhóm hàng",
                color_continuous_scale="Blues",
            ),
            width="stretch",
        )

with tab_features:
    if not filtered.empty and "onpromotion" in filtered:
        promo = filtered.assign(
            Trạng_thái=filtered["onpromotion"].fillna(0).gt(0).map(
                {True: "Có khuyến mãi", False: "Không khuyến mãi"}
            )
        )
        promo_summary = promo.groupby("Trạng_thái", as_index=False)["sales"].mean()
        st.plotly_chart(
            px.bar(
                promo_summary,
                x="Trạng_thái",
                y="sales",
                title="Doanh số trung bình theo trạng thái khuyến mãi",
            ),
            width="stretch",
        )
    if not filtered.empty and "has_holiday" in filtered:
        holiday_summary = (
            filtered.assign(
                Loại_ngày=filtered["has_holiday"].gt(0).map(
                    {True: "Ngày lễ/sự kiện", False: "Ngày thường"}
                )
            )
            .groupby("Loại_ngày", as_index=False)["sales"]
            .mean()
        )
        st.plotly_chart(
            px.bar(
                holiday_summary,
                x="Loại_ngày",
                y="sales",
                title="Ngày lễ và ngày thường",
            ),
            width="stretch",
        )
    if not filtered.empty and "dcoilwtico" in filtered:
        external = filtered.dropna(subset=["dcoilwtico"])
        if not external.empty:
            st.plotly_chart(
                px.scatter(
                    external.sample(min(len(external), 5000), random_state=42),
                    x="dcoilwtico",
                    y="sales",
                    title="Tương quan giá dầu WTI và doanh số",
                ),
                width="stretch",
            )
    if not filtered.empty and not {
        "dcoilwtico",
        "has_holiday",
    }.intersection(filtered.columns):
        st.info("Nạp thêm bảng oil.csv/holidays_events.csv để phân tích biến ngoại sinh.")

st.subheader("Thống kê mô tả và outlier")
if not filtered.empty:
    q1, q3 = filtered["sales"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_rows = filtered[
        (filtered["sales"] < q1 - 1.5 * iqr)
        | (filtered["sales"] > q3 + 1.5 * iqr)
    ]
    left, right = st.columns([1, 1])
    left.dataframe(filtered["sales"].describe().to_frame().T, width="stretch")
    right.metric("Số điểm ngoài ngưỡng 1.5×IQR", f"{len(outlier_rows):,}")
else:
    st.info("Chưa có dữ liệu để tính thống kê mô tả và outlier.")
