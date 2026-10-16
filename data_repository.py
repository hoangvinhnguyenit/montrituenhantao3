# -*- coding: utf-8 -*-
"""Local-file and optional Supabase adapters for the ERD data model."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

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
    try:
        secrets = st.secrets
    except FileNotFoundError:
        secrets = {}
    url = os.getenv("SUPABASE_URL") or secrets.get("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or secrets.get("SUPABASE_KEY")
    return url, key


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
