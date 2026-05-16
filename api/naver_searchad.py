"""
Naver Search Ad Keyword Tool API client.
Endpoint: GET https://api.searchad.naver.com/keywordstool
Auth: HMAC-SHA256 signature.
"""
import os
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd


def _get_signature(timestamp, method, path):
    secret_key = os.environ.get("NAVER_SEARCHAD_SECRET_KEY", "")
    message = f"{timestamp}.{method}.{path}"
    sign = hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(sign).decode("utf-8")


def _get_headers(method, path):
    timestamp = str(int(time.time() * 1000))
    return {
        "X-Timestamp": timestamp,
        "X-API-KEY": os.environ.get("NAVER_SEARCHAD_API_KEY", ""),
        "X-Customer": os.environ.get("NAVER_SEARCHAD_CUSTOMER_ID", ""),
        "X-Signature": _get_signature(timestamp, method, path),
        "Content-Type": "application/json",
    }


def get_keyword_stats(keywords, show_detail=1):
    """
    Get monthly search volume and related keywords.

    Args:
        keywords: list of keyword strings
        show_detail: 1 for detailed stats

    Returns:
        pd.DataFrame with columns [relKeyword, monthlyPcQcCnt, monthlyMobileQcCnt,
                                    monthlyAvePcClkCnt, monthlyAveMobileClkCnt,
                                    compIdx, plAvgDepth]
    """
    path = "/keywordstool"
    headers = _get_headers("GET", path)

    if not headers["X-API-KEY"]:
        return pd.DataFrame()

    all_results = []
    # Process one keyword at a time; remove spaces (API rejects keywords with spaces)
    for kw in keywords:
        clean_kw = kw.replace(" ", "")
        params = {
            "hintKeywords": clean_kw,
            "showDetail": show_detail,
        }
        try:
            resp = requests.get(
                f"https://api.searchad.naver.com{path}",
                headers=_get_headers("GET", path),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get("keywordList", []))
        except Exception as e:
            print(f"[SearchAd API Error] '{kw}': {e}")
            continue

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    # Convert '< 10' style values to numeric
    for col in ["monthlyPcQcCnt", "monthlyMobileQcCnt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("< ", "").str.replace(",", ""),
                errors="coerce",
            ).fillna(0).astype(int)

    return df


def is_available():
    return all([
        os.environ.get("NAVER_SEARCHAD_API_KEY"),
        os.environ.get("NAVER_SEARCHAD_SECRET_KEY"),
        os.environ.get("NAVER_SEARCHAD_CUSTOMER_ID"),
    ])


def test_connection():
    if not is_available():
        return False, "API keys not configured"
    try:
        df = get_keyword_stats(["여행"])
        if df.empty:
            return False, "No data returned"
        return True, "Connected"
    except Exception as e:
        return False, str(e)
