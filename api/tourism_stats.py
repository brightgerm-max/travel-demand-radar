"""
Public Data Portal - Korean Tourism Statistics API client.
Dataset: Korean Tourism Organization monthly overseas departure stats (#15136390)
"""
import os
import requests
import pandas as pd
from urllib.parse import quote


BASE_URL = "https://apis.data.go.kr/B551011/TarDepart/tarDepart"


def fetch_departure_stats(year, month=None):
    """
    Fetch monthly overseas departure statistics.

    Args:
        year: int (e.g. 2025)
        month: int 1-12 or None for full year

    Returns:
        pd.DataFrame with columns [natKorNm, natEngNm, ed, num]
        (country_kr, country_en, year-month, departure_count)
    """
    service_key = os.environ.get("PUBLIC_DATA_SERVICE_KEY", "")
    if not service_key:
        return pd.DataFrame()

    all_rows = []
    months = [month] if month else list(range(1, 13))

    for m in months:
        ym = f"{year}{m:02d}"
        params = {
            "serviceKey": service_key,
            "numOfRows": 500,
            "pageNo": 1,
            "MobileOS": "ETC",
            "MobileApp": "TravelDemand",
            "type": "json",
            "ed": ym,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            body = data.get("response", {}).get("body", {})
            items = body.get("items", {}).get("item", [])
            if isinstance(items, dict):
                items = [items]
            all_rows.extend(items)
        except Exception as e:
            print(f"[Tourism API Error] {ym}: {e}")
            continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "num" in df.columns:
        df["num"] = pd.to_numeric(df["num"], errors="coerce").fillna(0).astype(int)
    return df


def is_available():
    return bool(os.environ.get("PUBLIC_DATA_SERVICE_KEY"))


def test_connection():
    if not is_available():
        return False, "API key not configured"
    try:
        df = fetch_departure_stats(2025, 1)
        if df.empty:
            return False, "No data returned"
        return True, "Connected"
    except Exception as e:
        return False, str(e)
