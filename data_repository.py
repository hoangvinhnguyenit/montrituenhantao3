# -*- coding: utf-8 -*-
"""Local-file and optional Supabase adapters for the ERD data model."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from utils import load_sales_data


ERD_TABLES = (
    "train2",
    "test",
    "stores",
    "transactions",
    "oil",
    "holidays_events",
    "sample_submission",
)

SNAPSHOT_TABLE = "app_dataset_snapshots"
SNAPSHOT_ROWS_TABLE = "app_dataset_snapshot_rows"
SNAPSHOT_BATCH_SIZE = 500

TABLE_ALIASES = {
    "train": "train2",
    "train2": "train2",
    "test": "test",
    "stores": "stores",
    "transactions": "transactions",
    "oil": "oil",
    "holidays": "holidays_events",
    "holidays_events": "holidays_events",
    "sample_submission": "sample_submission",
}


def _table_name(filename: str) -> str | None:
    stem = Path(filename).stem.strip().lower()
    return TABLE_ALIASES.get(stem)


def _read_csv_bytes(name: str, payload: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(payload))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Không đọc được file CSV {name}: {exc}") from exc


def _read_upload(upload) -> dict[str, pd.DataFrame]:
    filename = upload.name.lower()
    if filename.endswith(".zip"):
        try:
            archive = zipfile.ZipFile(io.BytesIO(upload.getvalue()))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{upload.name} không phải file ZIP hợp lệ.") from exc
        tables = {}
        for member in archive.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue
            table = _table_name(member)
            if table:
                tables[table] = _read_csv_bytes(
                    member, archive.read(member)
                )
        return tables

    table = _table_name(upload.name)
    if filename.endswith(".csv"):
        data = _read_csv_bytes(upload.name, upload.getvalue())
    elif filename.endswith((".xlsx", ".xls")):
        data = pd.read_excel(upload)
    else:
        raise ValueError(f"Định dạng chưa hỗ trợ: {upload.name}")
    if table is None:
        table = "generic_sales"
    return {table: data}


def load_uploaded_sources(uploads) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Read individual CSVs, Excel, or a Kaggle-style ZIP and join by ERD keys."""
    tables = {}
    for upload in uploads:
        for name, frame in _read_upload(upload).items():
            if name in tables:
                raise ValueError(f"Bạn đã tải lên nhiều hơn một file cho bảng {name}.")
            tables[name] = frame

    if not tables:
        raise ValueError("Chưa có file dữ liệu để nạp.")
    if "train2" not in tables:
        generic = tables.get("generic_sales")
        if generic is None and len(tables) == 1:
            generic = next(iter(tables.values()))
        if generic is None:
            raise ValueError("Cần file train2.csv (hoặc train.csv) làm bảng doanh số.")
        sales = load_sales_data(io.BytesIO(generic.to_csv(index=False).encode("utf-8")))
        return sales, None, tables
    sales, test = normalize_erd_tables(tables)
    return sales, test, tables


def _normalized_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result


def _assert_unique(frame: pd.DataFrame, keys: list[str], table_name: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(
            f"Bảng {table_name} vi phạm khóa duy nhất theo "
            f"{', '.join(keys)}. Hãy xử lý bản ghi trùng trước khi nối."
        )


def normalize_erd_tables(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Join the ERD tables without multiplying training/test fact rows."""
    if "train2" not in tables:
        raise ValueError("Thiếu bảng train2 (bảng doanh số huấn luyện).")

    stores = _normalized_table(tables["stores"]) if "stores" in tables else None
    transactions = (
        _normalized_table(tables["transactions"])
        if "transactions" in tables
        else None
    )
    oil = _normalized_table(tables["oil"]) if "oil" in tables else None
    holidays = (
        _normalized_table(tables["holidays_events"])
        if "holidays_events" in tables
        else None
    )

    if stores is not None:
        if "store_nbr" not in stores:
            raise ValueError("Bảng stores cần có khóa store_nbr.")
        _assert_unique(stores, ["store_nbr"], "stores")

    if transactions is not None:
        required = {"date", "store_nbr", "transactions"}
        if not required.issubset(transactions.columns):
            raise ValueError(
                "Bảng transactions cần có date, store_nbr và transactions."
            )
        # This is the ERD's composite key. Sum repeated imports before joining.
        transactions = (
            transactions.groupby(["date", "store_nbr"], as_index=False, dropna=False)[
                "transactions"
            ]
            .sum(min_count=1)
        )

    if oil is not None and "date" in oil:
        if "dcoilwtico" not in oil:
            candidates = [column for column in oil if column != "date"]
            if candidates:
                oil = oil.rename(columns={candidates[0]: "dcoilwtico"})
        if "dcoilwtico" in oil:
            oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
            oil = oil.groupby("date", as_index=False)["dcoilwtico"].mean()

    holiday_by_date = None
    if holidays is not None and "date" in holidays:
        holiday_columns = [
            column
            for column in ("type", "locale", "locale_name", "description")
            if column in holidays
        ]
        holiday_by_date = (
            holidays.groupby("date", dropna=False)
            .agg(
                has_holiday=("date", "size"),
                holiday_types=(
                    holiday_columns[0],
                    lambda values: ", ".join(
                        sorted(values.dropna().astype(str).unique())
                    ),
                )
                if holiday_columns
                else ("date", lambda values: "Ngày lễ"),
            )
            .reset_index()
        )

    def enrich(facts: pd.DataFrame, *, has_target: bool) -> pd.DataFrame:
        facts = _normalized_table(facts)
        if not {"date", "store_nbr", "family"}.issubset(facts.columns):
            raise ValueError(
                "Bảng train2/test cần có date, store_nbr và family theo ERD."
            )
        if "id" not in facts:
            raise ValueError("Bảng train2/test cần có khóa chính id theo ERD.")
        _assert_unique(facts, ["id"], "train2" if has_target else "test")
        if has_target and "sales" not in facts:
            raise ValueError("Bảng train2 cần có cột sales.")
        if "onpromotion" not in facts:
            facts["onpromotion"] = 0
        facts["onpromotion"] = pd.to_numeric(
            facts["onpromotion"], errors="coerce"
        ).fillna(0)
        facts["store_nbr"] = facts["store_nbr"].astype("string")

        if stores is not None:
            dimension = stores.copy()
            dimension["store_nbr"] = dimension["store_nbr"].astype("string")
            facts = facts.merge(
                dimension, on="store_nbr", how="left", validate="many_to_one"
            )
        if transactions is not None:
            transaction_dim = transactions.copy()
            transaction_dim["store_nbr"] = transaction_dim["store_nbr"].astype(
                "string"
            )
            facts = facts.merge(
                transaction_dim,
                on=["date", "store_nbr"],
                how="left",
                validate="many_to_one",
            )
        if oil is not None and "dcoilwtico" in oil:
            facts = facts.merge(oil, on="date", how="left", validate="many_to_one")
        if holiday_by_date is not None:
            facts = facts.merge(
                holiday_by_date, on="date", how="left", validate="many_to_one"
            )
            facts["has_holiday"] = facts["has_holiday"].fillna(0).astype("int16")
        facts["family"] = facts["family"].astype("string").fillna("Không xác định")
        if has_target:
            facts["sales"] = pd.to_numeric(facts["sales"], errors="coerce")
        return facts.sort_values("date").reset_index(drop=True)

    sales = enrich(tables["train2"], has_target=True)
    test = enrich(tables["test"], has_target=False) if "test" in tables else None
    return sales, test


def get_supabase_configuration() -> tuple[str | None, str | None]:
    """Read server-side credentials from environment variables or Streamlit secrets."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if url and key:
        return url, key

    try:
        secrets = st.secrets
    except FileNotFoundError:
        secrets = {}

    url = url or secrets.get("SUPABASE_URL")
    key = key or secrets.get("SUPABASE_KEY")
    return url, key


@st.cache_resource
def _create_supabase_client(url: str, key: str) -> Any:
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu gói supabase-py. Cài bằng `pip install supabase` để bật nguồn này."
        ) from exc
    return create_client(url, key)


def _get_supabase_client() -> Any:
    url, key = get_supabase_configuration()
    if not url or not key:
        raise ValueError("Hãy cấu hình SUPABASE_URL và SUPABASE_KEY ở môi trường máy chủ.")
    return _create_supabase_client(url, key)


def _execute_supabase_query(query: Any) -> Any:
    try:
        from httpx import HTTPError
        from postgrest.exceptions import APIError
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu gói supabase-py. Cài dependencies bằng `pip install -r requirements.txt`."
        ) from exc
    try:
        return query.execute()
    except (APIError, HTTPError) as exc:
        raise RuntimeError(f"Yêu cầu Supabase thất bại: {exc}") from exc


def save_dataset_snapshot(tables: dict[str, pd.DataFrame], source_label: str) -> str:
    """Persist one immutable dataset snapshot and its rows in Supabase."""
    if not tables:
        raise ValueError("Không có bảng dữ liệu để lưu.")

    client = _get_supabase_client()
    snapshot_id = str(uuid4())
    table_metadata = {
        name: {"columns": [str(column) for column in frame.columns], "row_count": len(frame)}
        for name, frame in tables.items()
    }
    _execute_supabase_query(
        client.table(SNAPSHOT_TABLE).insert(
            {
                "id": snapshot_id,
                "source_label": source_label,
                "table_metadata": table_metadata,
                "is_complete": False,
            }
        )
    )

    for table_name, frame in tables.items():
        records = json.loads(frame.to_json(orient="records", date_format="iso"))
        for start in range(0, len(records), SNAPSHOT_BATCH_SIZE):
            batch = records[start : start + SNAPSHOT_BATCH_SIZE]
            _execute_supabase_query(
                client.table(SNAPSHOT_ROWS_TABLE).insert(
                    [
                        {
                            "snapshot_id": snapshot_id,
                            "table_name": table_name,
                            "row_index": start + index,
                            "row_data": record,
                        }
                        for index, record in enumerate(batch)
                    ]
                )
            )

    _execute_supabase_query(
        client.table(SNAPSHOT_TABLE)
        .update({"is_complete": True})
        .eq("id", snapshot_id)
    )
    return snapshot_id


def list_dataset_snapshots(limit: int = 20) -> list[dict]:
    """Return recent complete snapshots, newest first."""
    if limit < 1:
        raise ValueError("Giới hạn snapshot phải lớn hơn 0.")
    response = _execute_supabase_query(
        _get_supabase_client()
        .table(SNAPSHOT_TABLE)
        .select("id, created_at, source_label, table_metadata")
        .eq("is_complete", True)
        .order("created_at", desc=True)
        .limit(limit)
    )
    return response.data or []


def load_dataset_snapshot(
    snapshot_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, pd.DataFrame], str]:
    """Restore source tables from one complete Supabase snapshot."""
    client = _get_supabase_client()
    header_response = _execute_supabase_query(
        client.table(SNAPSHOT_TABLE)
        .select("source_label, table_metadata")
        .eq("id", snapshot_id)
        .eq("is_complete", True)
        .limit(1)
    )
    if not header_response.data:
        raise ValueError("Không tìm thấy snapshot hoàn chỉnh đã chọn.")
    header = header_response.data[0]
    table_metadata = header["table_metadata"]
    tables: dict[str, pd.DataFrame] = {}

    for table_name, metadata in table_metadata.items():
        rows = []
        offset = 0
        while True:
            response = _execute_supabase_query(
                client.table(SNAPSHOT_ROWS_TABLE)
                .select("row_index, row_data")
                .eq("snapshot_id", snapshot_id)
                .eq("table_name", table_name)
                .order("row_index")
                .range(offset, offset + 999)
            )
            batch = response.data or []
            rows.extend(batch)
            if len(batch) < 1000:
                break
            offset += len(batch)
        tables[table_name] = pd.DataFrame(
            [row["row_data"] for row in rows],
            columns=metadata["columns"],
        )

    if "train2" in tables:
        sales, test = normalize_erd_tables(tables)
    elif "generic_sales" in tables:
        sales = load_sales_data(
            io.BytesIO(tables["generic_sales"].to_csv(index=False).encode("utf-8"))
        )
        test = None
    else:
        raise ValueError("Snapshot không có bảng doanh số để khôi phục.")

    return sales, test, tables, header["source_label"]


def load_latest_dataset_snapshot(
) -> tuple[str, pd.DataFrame, pd.DataFrame | None, dict[str, pd.DataFrame], str] | None:
    """Restore the latest complete snapshot, if one exists."""
    snapshots = list_dataset_snapshots(limit=1)
    if not snapshots:
        return None
    snapshot = snapshots[0]
    sales, test, tables, source_label = load_dataset_snapshot(snapshot["id"])
    return snapshot["id"], sales, test, tables, source_label


def load_supabase_sources(
    *,
    row_limit: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Load a bounded, recent slice of ERD tables from a configured Supabase project."""
    url, key = get_supabase_configuration()
    if not url or not key:
        raise ValueError("Hãy cấu hình SUPABASE_URL và SUPABASE_KEY ở môi trường máy chủ.")
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu gói supabase-py. Cài bằng `pip install supabase` để bật nguồn này."
        ) from exc

    client = create_client(url, key)
    tables = {}
    for table in ERD_TABLES:
        limit = row_limit if table in {"train2", "test", "transactions"} else min(row_limit, 5000)
        rows = []
        for offset in range(0, limit, 1000):
            query = client.table(table).select("*")
            if table in {"train2", "test", "transactions"}:
                query = query.order("date", desc=True)
            page_size = min(1000, limit - offset)
            result = query.range(offset, offset + page_size - 1).execute()
            rows.extend(result.data or [])
            if len(result.data or []) < page_size:
                break
        if rows:
            tables[table] = pd.DataFrame(rows)

    if "train2" not in tables:
        raise ValueError("Không tìm thấy bảng train2 hoặc bảng này không có dữ liệu.")
    sales, test = normalize_erd_tables(tables)
    return sales, test, tables


def make_sample_dataset(days: int = 730) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a compact synthetic dataset that follows the ERD relationships."""
    random = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    stores = pd.DataFrame(
        [
            {"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1},
            {"store_nbr": 2, "city": "Guayaquil", "state": "Guayas", "type": "B", "cluster": 2},
            {"store_nbr": 3, "city": "Cuenca", "state": "Azuay", "type": "A", "cluster": 1},
        ]
    )
    families = ["GROCERY I", "BEVERAGES", "PRODUCE", "DAIRY"]
    facts = []
    transactions = []
    for store in stores["store_nbr"]:
        store_multiplier = 0.8 + 0.25 * int(store)
        for family_index, family in enumerate(families):
            base = 32 + family_index * 13
            weekly = 8 * np.sin(2 * np.pi * dates.dayofweek / 7)
            seasonal = 6 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            promotion = random.binomial(1, 0.12, size=days)
            noise = random.normal(0, 5, size=days)
            sales = np.maximum(
                0, store_multiplier * (base + weekly + seasonal + 18 * promotion + noise)
            )
            for index, date in enumerate(dates):
                facts.append(
                    {
                        "id": len(facts) + 1,
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": round(float(sales[index]), 2),
                        "onpromotion": int(promotion[index]),
                    }
                )
        transactions.extend(
            {
                "date": date,
                "store_nbr": store,
                "transactions": int(random.integers(250, 900)),
            }
            for date in dates
        )

    oil = pd.DataFrame(
        {"date": dates, "dcoilwtico": 65 + 4 * np.sin(np.arange(days) / 45)}
    )
    holiday_dates = dates[dates.dayofweek == 6][::7]
    holidays = pd.DataFrame(
        {
            "date": holiday_dates,
            "type": "Holiday",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Ngày lễ mẫu",
            "transferred": False,
        }
    )
    tables = {
        "train2": pd.DataFrame(facts),
        "stores": stores,
        "transactions": pd.DataFrame(transactions),
        "oil": oil,
        "holidays_events": holidays,
    }
    sales, _ = normalize_erd_tables(tables)
    return sales, tables
