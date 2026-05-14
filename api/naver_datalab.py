"""
Naver DataLab Search Trend API client.
Endpoint: POST https://openapi.naver.com/v1/datalab/search
Limit: max 5 keyword groups per call, 1000 calls/day.
"""
import os
import requests
import pandas as pd
from datetime import datetime


def _get_headers():
    return {
        "X-Naver-Client-Id": os.environ.get("NAVER_CLIENT_ID", ""),
        "X-Naver-Client-Secret": os.environ.get("NAVER_CLIENT_SECRET", ""),
        "Content-Type": "application/json",
    }


def _build_keyword_groups(keywords):
    """Convert keyword list to keyword group format (batch of 5)."""
    groups = []
    for kw in keywords:
        if isinstance(kw, dict):
            groups.append(kw)
        else:
            groups.append({"groupName": kw, "keywords": [kw]})
    return groups


def fetch_trend(keywords, start_date, end_date, time_unit="month"):
    """
    Fetch search trend data from Naver DataLab.

    Args:
        keywords: list of keyword strings or dicts with groupName/keywords
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        time_unit: 'month', 'week', or 'date'

    Returns:
        pd.DataFrame with columns [keyword, period, ratio]
    """
    headers = _get_headers()
    if not headers["X-Naver-Client-Id"]:
        return pd.DataFrame()

    all_groups = _build_keyword_groups(keywords)
    all_results = []

    # Process in batches of 5
    for i in range(0, len(all_groups), 5):
        batch = all_groups[i:i+5]
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": time_unit,
            "keywordGroups": batch,
        }
        try:
            resp = requests.post(
                "https://openapi.naver.com/v1/datalab/search",
                headers=headers,
                json=body,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            for result in data.get("results", []):
                group_name = result["title"]
                for item in result.get("data", []):
                    all_results.append({
                        "keyword": group_name,
                        "period": item["period"],
                        "ratio": item["ratio"],
                    })
        except Exception as e:
            print(f"[DataLab API Error] batch {i//5}: {e}")
            continue

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["period"] = pd.to_datetime(df["period"])
    return df


def is_available():
    """Check if DataLab API credentials are configured."""
    return bool(os.environ.get("NAVER_CLIENT_ID")) and bool(os.environ.get("NAVER_CLIENT_SECRET"))


def test_connection():
    """Test API connection with a simple query."""
    if not is_available():
        return False, "API keys not configured"
    try:
        df = fetch_trend(
            ["테스트"],
            "2025-01-01",
            "2025-02-01",
            time_unit="month",
        )
        if df.empty:
            return False, "No data returned"
        return True, "Connected"
    except Exception as e:
        return False, str(e)
