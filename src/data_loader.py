"""
Data Loader and Preprocessing Pipeline for E-Commerce Business Analytics Dashboard
Handles dataset ingestion, missing values, duplicates, cancellation segregation,
and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validates if required columns or flexible aliases exist in the dataframe."""
    col_map = {col.lower().replace(" ", "").replace("_", ""): col for col in df.columns}
    
    required_keys = ["invoiceno", "quantity", "unitprice", "invoicedate"]
    missing = [req for req in required_keys if req not in col_map]
    
    if missing:
        return False, f"Missing essential columns: {', '.join(missing)}. Available columns: {list(df.columns)}"
    return True, "Dataset schema valid."


def clean_and_process_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw e-commerce data and computes derived business features.
    
    Returns:
        clean_df: Successful sale transactions
        cancelled_df: Cancelled / returned transactions
        data_health: Dictionary of audit metrics (counts, duplicates removed, etc.)
    """
    df = raw_df.copy()

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        norm = col.lower().replace(" ", "").replace("_", "")
        if "invoiceno" in norm or "invoice" in norm and "date" not in norm:
            col_mapping[col] = "InvoiceNo"
        elif "stockcode" in norm or "productcode" in norm or "sku" in norm:
            col_mapping[col] = "StockCode"
        elif "description" in norm or "productname" in norm or "title" in norm:
            col_mapping[col] = "Description"
        elif "quantity" in norm or "qty" in norm:
            col_mapping[col] = "Quantity"
        elif "invoicedate" in norm or "orderdate" in norm or "date" in norm:
            col_mapping[col] = "InvoiceDate"
        elif "unitprice" in norm or "price" in norm:
            col_mapping[col] = "UnitPrice"
        elif "customerid" in norm or "user" in norm or "client" in norm:
            col_mapping[col] = "CustomerID"
        elif "country" in norm or "region" in norm:
            col_mapping[col] = "Country"
        elif "category" in norm:
            col_mapping[col] = "Category"

    df = df.rename(columns=col_mapping)

    # Defaults for optional missing columns
    if "StockCode" not in df.columns:
        df["StockCode"] = "SKU-" + df.index.astype(str)
    if "Description" not in df.columns:
        df["Description"] = "Item " + df["StockCode"].astype(str)
    if "CustomerID" not in df.columns:
        df["CustomerID"] = np.nan
    if "Country" not in df.columns:
        df["Country"] = "Global"
    if "Category" not in df.columns:
        df["Category"] = "General Retail"

    initial_rows = len(df)

    # 1. Deduplication
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Date parsing
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # 3. Numeric conversions
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)

    # Clean description and string fields
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].replace({"": "UNKNOWN PRODUCT", "NAN": "UNKNOWN PRODUCT"})
    df["Country"] = df["Country"].astype(str).str.strip().str.title()
    df["Category"] = df["Category"].astype(str).str.strip().str.title()

    # 4. Handle CustomerID (Guest vs Registered)
    missing_customer_count = df["CustomerID"].isna().sum() + (df["CustomerID"].astype(str).str.strip() == "").sum()
    df["CustomerID_Clean"] = df["CustomerID"].apply(
        lambda x: "Guest Checkout" if pd.isna(x) or str(x).strip() in ["", "nan", "None"] 
        else str(int(float(x))) if str(x).replace(".", "").isdigit() 
        else str(x).strip()
    )

    # 5. Handle Cancellations
    # Transactions are cancelled if InvoiceNo starts with 'C' or Quantity <= 0
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    is_cancellation = df["InvoiceNo"].str.startswith("C", na=False) | (df["Quantity"] < 0)
    df["IsCancelled"] = is_cancellation

    # Compute Revenue = Quantity * UnitPrice
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Date temporal features
    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["MonthYear"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    # Split into clean sales and cancellations
    cancelled_df = df[df["IsCancelled"]].copy()
    cancelled_df["CancelledRevenue"] = cancelled_df["Revenue"].abs()

    # Valid sales: Quantity > 0 and UnitPrice > 0 and not cancelled
    clean_df = df[(~df["IsCancelled"]) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    data_health = {
        "initial_rows": int(initial_rows),
        "duplicates_removed": int(duplicates_count),
        "missing_customer_ids": int(missing_customer_count),
        "clean_sales_rows": int(len(clean_df)),
        "cancelled_rows": int(len(cancelled_df)),
        "cancellation_rate_pct": round((len(cancelled_df) / max(initial_rows, 1)) * 100, 2),
        "min_date": clean_df["InvoiceDate"].min().strftime("%Y-%m-%d") if len(clean_df) > 0 else "N/A",
        "max_date": clean_df["InvoiceDate"].max().strftime("%Y-%m-%d") if len(clean_df) > 0 else "N/A",
        "total_revenue": round(float(clean_df["Revenue"].sum()), 2),
        "total_orders": int(clean_df["InvoiceNo"].nunique()),
        "total_products": int(clean_df["StockCode"].nunique()),
        "total_customers": int(clean_df["CustomerID_Clean"].nunique()),
    }

    return clean_df, cancelled_df, data_health


def load_dataset(file_or_path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads dataset from uploaded file object or file path.
    Returns (clean_df, cancelled_df, health_stats, error_message)
    """
    try:
        if hasattr(file_or_path, "read"):
            # Streamlit UploadedFile
            df_raw = pd.read_csv(file_or_path, encoding="utf-8")
        elif isinstance(file_or_path, str):
            df_raw = pd.read_csv(file_or_path, encoding="ISO-8859-1")
        else:
            return None, None, None, "Invalid file format provided."

        is_valid, msg = validate_dataset_schema(df_raw)
        if not is_valid:
            return None, None, None, msg

        clean_df, cancelled_df, health = clean_and_process_data(df_raw)
        return clean_df, cancelled_df, health, None
    except Exception as e:
        return None, None, None, f"Error processing file: {str(e)}"

