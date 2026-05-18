"""
Public Data Portal - Korean Tourism Statistics API client.
Dataset: 한국문화관광연구원_출입국관광통계서비스 (#15000297)
- ED_CD=D: 국민해외관광객 (한국인이 어느 나라로 갔는지)
- ED_CD=E: 방한외래관광객 (외국인이 한국에 온 것)
"""
import os
import requests
import pandas as pd


BASE_URL = "http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList"


def fetch_departure_stats(year, month=None):
    """
    Fetch monthly overseas departure statistics BY DESTINATION COUNTRY.

    Args:
        year: int (e.g. 2025)
        month: int 1-12 or None for full year

    Returns:
        pd.DataFrame with columns [natKorNm, num, ed, ym]
        (country_kr, departure_count, direction_label, year-month)
    """
    service_key = os.environ.get("PUBLIC_DATA_SERVICE_KEY", "")
    if not service_key:
        return pd.DataFrame()

    all_rows = []
    months = [month] if month else list(range(1, 13))

    for m in months:
        ym = f"{year}{m:02d}"
        page = 1
        while True:
            params = {
                "serviceKey": service_key,
                "YM": ym,
                "ED_CD": "D",  # D=국민해외관광객 (출국)
                "_type": "json",
                "numOfRows": 100,
                "pageNo": page,
            }
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                header = data.get("response", {}).get("header", {})
                if header.get("resultCode") != "0000":
                    print(f"[Tourism API Error] {ym}: code={header.get('resultCode')}, msg={header.get('resultMsg')}")
                    break

                body = data.get("response", {}).get("body", {})
                items = body.get("items", {})
                if isinstance(items, str) or not items:
                    break
                item_list = items.get("item", [])
                if isinstance(item_list, dict):
                    item_list = [item_list]
                if not item_list:
                    break

                all_rows.extend(item_list)

                # 페이징
                total = int(body.get("totalCount", 0))
                if page * 100 >= total:
                    break
                page += 1
            except Exception as e:
                print(f"[Tourism API Error] {ym} page {page}: {e}")
                break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "num" in df.columns:
        df["num"] = pd.to_numeric(df["num"], errors="coerce").fillna(0).astype(int)
    return df


def fetch_inbound_stats(year, month=None):
    """Fetch inbound foreign tourist stats (방한외래관광객)."""
    service_key = os.environ.get("PUBLIC_DATA_SERVICE_KEY", "")
    if not service_key:
        return pd.DataFrame()

    all_rows = []
    months = [month] if month else list(range(1, 13))

    for m in months:
        ym = f"{year}{m:02d}"
        params = {
            "serviceKey": service_key,
            "YM": ym,
            "ED_CD": "E",  # E=방한외래관광객 (입국)
            "_type": "json",
            "numOfRows": 100,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            header = data.get("response", {}).get("header", {})
            if header.get("resultCode") != "0000":
                continue
            body = data.get("response", {}).get("body", {})
            items = body.get("items", {})
            if isinstance(items, str) or not items:
                continue
            item_list = items.get("item", [])
            if isinstance(item_list, dict):
                item_list = [item_list]
            all_rows.extend(item_list)
        except Exception as e:
            print(f"[Tourism API Error] inbound {ym}: {e}")

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
        service_key = os.environ.get("PUBLIC_DATA_SERVICE_KEY", "")
        params = {
            "serviceKey": service_key,
            "YM": "202401",
            "ED_CD": "D",
            "_type": "json",
            "numOfRows": 1,
        }
        resp = requests.get(BASE_URL, params=params, timeout=15)
        data = resp.json()
        header = data.get("response", {}).get("header", {})
        if header.get("resultCode") == "0000":
            return True, "Connected"
        return False, header.get("resultMsg", "Unknown error")
    except Exception as e:
        return False, str(e)
