"""
Naver Shopping Search API client.
Endpoint: GET https://openapi.naver.com/v1/search/shop.json
Uses same Naver Developer App credentials as DataLab (with Search API enabled).
"""
import os
import requests
import pandas as pd


def _get_headers():
    # 쇼핑 전용 키가 있으면 우선 사용, 없으면 공용 키
    return {
        "X-Naver-Client-Id": os.environ.get("NAVER_SHOPPING_CLIENT_ID", os.environ.get("NAVER_CLIENT_ID", "")),
        "X-Naver-Client-Secret": os.environ.get("NAVER_SHOPPING_CLIENT_SECRET", os.environ.get("NAVER_CLIENT_SECRET", "")),
    }


def search_shopping(query, display=100, start=1, sort="sim"):
    """
    Search Naver Shopping.

    Args:
        query: search keyword
        display: number of results (max 100)
        start: start position (1-based)
        sort: sim(relevance), date, asc(price low), dsc(price high)

    Returns:
        list of dicts with keys: title, link, lprice, hprice, mallName, maker, brand, category1~4, image
    """
    headers = _get_headers()
    if not headers["X-Naver-Client-Id"]:
        return []

    params = {
        "query": query,
        "display": min(display, 100),
        "start": start,
        "sort": sort,
    }
    try:
        resp = requests.get(
            "https://openapi.naver.com/v1/search/shop.json",
            headers=headers,
            params=params,
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[Shopping API Error] status={resp.status_code}, body={resp.text[:200]}")
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])

        # Clean up items
        for item in items:
            # Remove HTML tags from title
            item["title"] = (
                item.get("title", "")
                .replace("<b>", "")
                .replace("</b>", "")
            )
            # Convert price to int
            for price_key in ["lprice", "hprice"]:
                val = item.get(price_key, "0")
                item[price_key] = int(val) if val else 0

        return items
    except Exception as e:
        print(f"[Shopping API Error] {e}")
        return []


def search_shopping_df(query, display=100, sort="sim"):
    """Search and return as DataFrame."""
    items = search_shopping(query, display=display, sort=sort)
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    cols = ["title", "link", "lprice", "hprice", "mallName", "maker", "brand",
            "category1", "category2", "category3", "category4", "image"]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols]


def is_available():
    has_shopping = bool(os.environ.get("NAVER_SHOPPING_CLIENT_ID"))
    has_common = bool(os.environ.get("NAVER_CLIENT_ID"))
    return has_shopping or has_common


def test_connection():
    if not is_available():
        return False, "API keys not configured"
    try:
        items = search_shopping("여행", display=1)
        if not items:
            return False, "No data returned"
        return True, "Connected"
    except Exception as e:
        return False, str(e)
