# -*- coding: utf-8 -*-
"""Load local Kaggle-style files or connect to Supabase."""

import pandas as pd
import streamlit as st

from app_common import configure_page, initialize_session_state
from data_repository import (
    ERD_TABLES,
    get_supabase_configuration,
    list_dataset_snapshots,
    load_dataset_snapshot,
    load_supabase_sources,
    load_uploaded_sources,
    make_sample_dataset,
    save_dataset_snapshot,
)


configure_page("Dữ liệu")
initialize_session_state()

st.title("Dữ liệu bán hàng")
st.caption(
    "Nạp dữ liệu theo ERD: train2, test, stores, transactions, oil, "
    "holidays_events và sample_submission."
)


def save_dataset(
    sales: pd.DataFrame,
    test: pd.DataFrame | None,
    tables: dict[str, pd.DataFrame],
    source: str,
    *,
    persist: bool = True,
) -> str | None:
    st.session_state.raw_df = sales
    st.session_state.processed_df = None
    st.session_state.pop("inventory_alert_stock_editor", None)
    st.session_state.test_df = test
    st.session_state.source_tables = tables
    st.session_state.source_label = source
    for key, default in (
        ("model", None),
        ("scaler", None),
        ("test_metrics", None),
        ("test_pred_data", None),
        ("training_history", None),
        ("forecast_result", None),
        ("submission_result", None),
        ("inventory_plan", None),
        ("training_filter", None),
        ("training_series", None),
        ("full_scaled", None),
        ("model_config", None),
    ):
        st.session_state[key] = default

    if not persist:
        return None
    url, key = get_supabase_configuration()
    if not url or not key:
        return None
    return save_dataset_snapshot(tables, source)


local_tab, supabase_tab = st.tabs(["Tệp cục bộ", "Supabase"])
with local_tab:
    files = st.file_uploader(
        "Chọn CSV/Excel hoặc ZIP chứa các bảng ERD",
        type=["csv", "xlsx", "xls", "zip"],
        accept_multiple_files=True,
        help=(
            "Tên file được nhận diện theo train/train2, test, stores, "
            "transactions, oil, holidays_events, sample_submission."
        ),
    )
    load_button = st.button("Nạp dữ liệu", type="primary", disabled=not files)
    sample_button = st.button("Dùng dữ liệu mẫu theo ERD")
    if load_button:
        try:
            sales, test, tables = load_uploaded_sources(files)
            snapshot_id = save_dataset(sales, test, tables, "Tệp cục bộ")
            st.success(f"Đã nạp {len(sales):,} dòng bán hàng.")
            if snapshot_id:
                st.success("Đã lưu snapshot trên Supabase; dữ liệu có thể khôi phục sau F5.")
            else:
                st.warning("Chưa cấu hình Supabase; dữ liệu hiện chỉ nằm trong phiên này.")
        except (ValueError, RuntimeError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Không thể nạp hoặc lưu dữ liệu: {exc}")
    if sample_button:
        sales, tables = make_sample_dataset()
        try:
            snapshot_id = save_dataset(sales, None, tables, "Dữ liệu mẫu")
            st.success(f"Đã tạo dữ liệu mẫu theo ERD ({len(sales):,} dòng).")
            if snapshot_id:
                st.success("Đã lưu snapshot trên Supabase; dữ liệu có thể khôi phục sau F5.")
            else:
                st.warning("Chưa cấu hình Supabase; dữ liệu hiện chỉ nằm trong phiên này.")
        except (ValueError, RuntimeError) as exc:
            st.error(f"Đã tạo dữ liệu mẫu nhưng không thể lưu lên Supabase: {exc}")

    st.markdown("**Khóa nối:** train2/test → stores theo `store_nbr`; transactions theo "
                "`date + store_nbr`; oil và holidays_events theo `date`.")
    st.caption(
        "Có thể tải trực tiếp file train.csv và stores/transactions/oil/holidays_events "
        "từ bộ dữ liệu Favorita. File có tên khác sẽ được đọc như bảng doanh số tổng quát."
    )

with supabase_tab:
    supabase_url, supabase_key = get_supabase_configuration()
    if supabase_url and supabase_key:
        st.success("Đã tìm thấy cấu hình Supabase phía máy chủ; khóa không được hiển thị.")
        row_limit = st.number_input(
            "Giới hạn dòng gần nhất cho train2/test",
            min_value=1000,
            max_value=200000,
            value=50000,
            step=1000,
        )
        if st.button("Nạp từ Supabase", type="primary"):
            try:
                sales, test, tables = load_supabase_sources(row_limit=int(row_limit))
                snapshot_id = save_dataset(sales, test, tables, "Supabase")
                st.success(f"Đã đọc {len(sales):,} dòng doanh số từ Supabase.")
                if snapshot_id:
                    st.success("Đã lưu snapshot; dữ liệu có thể khôi phục sau F5.")
                else:
                    st.warning("Chưa cấu hình Supabase; dữ liệu hiện chỉ nằm trong phiên này.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"Không thể đọc hoặc lưu Supabase: {exc}")

        if st.session_state.dataset_restore_error:
            st.caption(
                "Kiểm tra kết nối, cấu hình và quyền truy cập Supabase; xem README "
                "để tạo bảng snapshot."
            )
            if st.button("Thử khôi phục snapshot lại"):
                st.session_state.dataset_restore_attempted = False
                st.session_state.dataset_restore_error = None
                st.rerun()
            snapshots = []
        else:
            try:
                snapshots = list_dataset_snapshots()
            except RuntimeError as exc:
                snapshots = []
                st.error(f"Không thể đọc lịch sử snapshot: {exc}")
        if snapshots:
            with st.expander("Lịch sử các lần lưu", expanded=False):
                snapshot_labels = {
                    (
                        f"{snapshot['created_at']} — {snapshot['source_label']} — "
                        f"{sum(table['row_count'] for table in snapshot['table_metadata'].values()):,} dòng "
                        f"({snapshot['id'][:8]})"
                    ): snapshot["id"]
                    for snapshot in snapshots
                }
                selected_snapshot = st.selectbox(
                    "Chọn snapshot cần khôi phục",
                    options=list(snapshot_labels),
                    key="dataset_snapshot_to_restore",
                )
                if st.button("Khôi phục snapshot đã chọn"):
                    try:
                        sales, test, tables, source = load_dataset_snapshot(
                            snapshot_labels[selected_snapshot]
                        )
                        save_dataset(sales, test, tables, source, persist=False)
                        st.success("Đã khôi phục snapshot vào phiên làm việc.")
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"Không thể khôi phục snapshot: {exc}")
    else:
        st.info(
            "Khi cần kết nối, đặt `SUPABASE_URL` và `SUPABASE_KEY` trong biến môi "
            "trường của máy chủ Streamlit. Không đưa service-role key vào mã nguồn."
        )
    st.code(
        "\n".join(f"{table}" for table in ERD_TABLES),
        language="text",
    )

sales = st.session_state.raw_df
if sales is not None and not sales.empty:
    st.divider()
    st.subheader(f"Đang dùng: {st.session_state.source_label or 'dữ liệu đã nạp'}")
    first, second, third, fourth = st.columns(4)
    first.metric("Số dòng", f"{len(sales):,}")
    second.metric("Nhóm hàng", f"{sales['family'].nunique():,}")
    third.metric(
        "Cửa hàng",
        f"{sales['store_nbr'].nunique():,}" if "store_nbr" in sales else "—",
    )
    fourth.metric(
        "Khoảng ngày",
        f"{sales['date'].min():%d/%m/%Y} – {sales['date'].max():%d/%m/%Y}",
    )
    st.dataframe(sales.head(30), width="stretch", hide_index=True)
    if st.session_state.source_tables:
        with st.expander("Các bảng nguồn đã nạp"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Bảng": name, "Số dòng": len(frame), "Số cột": len(frame.columns)}
                        for name, frame in st.session_state.source_tables.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
