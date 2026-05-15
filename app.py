import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
from pathlib import Path

# ── 환경변수 로드: st.secrets → .env → os.environ ───
# 1) Streamlit Cloud Secrets (영구 저장, 최우선)
_SECRET_KEYS = [
    "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET",
    "NAVER_SEARCHAD_API_KEY", "NAVER_SEARCHAD_SECRET_KEY", "NAVER_SEARCHAD_CUSTOMER_ID",
    "PUBLIC_DATA_SERVICE_KEY",
]
try:
    for _k in _SECRET_KEYS:
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

# 2) 로컬 .env 파일 (개발용)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── 페이지 설정 ─────────────────────────────────────
st.set_page_config(page_title="Travel Demand Radar", layout="wide", page_icon="🌏")

# ── 경로 상수 ───────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ── API 임포트 ──────────────────────────────────────
from api import naver_datalab, naver_searchad, naver_shopping, tourism_stats

# ── 데이터 로더 ─────────────────────────────────────
def load_json(filename):
    path = DATA_DIR / filename
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def save_json(filename, data):
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

@st.cache_data
def load_fallback_data():
    csv_path = DATA_DIR / "fallback_data.csv"
    if not csv_path.exists():
        csv_path = BASE_DIR / "data.csv"
    df = pd.read_csv(csv_path)
    date_cols = [c for c in df.columns if c not in ("국가", "키워드")]
    valid_dates = []
    for c in date_cols:
        try:
            pd.to_datetime(c)
            valid_dates.append(c)
        except:
            pass
    df = df.melt(
        id_vars=["국가", "키워드"], value_vars=valid_dates,
        var_name="주차시작일", value_name="쿼리수",
    )
    df["주차시작일"] = pd.to_datetime(df["주차시작일"])
    df["연월"] = df["주차시작일"].dt.to_period("M")
    df["연도"] = df["주차시작일"].dt.year
    df["월"] = df["주차시작일"].dt.month
    df["월표시"] = df["월"].astype(str) + "월"
    df["쿼리수"] = (
        df["쿼리수"].astype(str)
        .str.replace(",", "", regex=False).str.strip()
    )
    df["쿼리수"] = pd.to_numeric(df["쿼리수"], errors="coerce").fillna(0)
    return df

# ── API 기반 데이터 로더 (캐싱) ────────────────────
@st.cache_data(ttl=3600, show_spinner="🔍 검색량 데이터 조회 중...")
def load_searchad_data(_keywords_tuple):
    """검색광고 API로 국가별 대표 키워드 월간 검색수 조회. 결과를 국가별로 집계."""
    keywords_by_country = dict(_keywords_tuple)
    rows = []
    for country, kw_list in keywords_by_country.items():
        if not kw_list:
            continue
        df = naver_searchad.get_keyword_stats(kw_list[:5])
        if df.empty:
            continue
        # 입력 키워드만 필터 (연관키워드 제외)
        kw_set = set(k.lower() for k in kw_list[:5])
        matched = df[df["relKeyword"].str.lower().isin(kw_set)]
        if matched.empty:
            matched = df.head(len(kw_list[:5]))
        for _, r in matched.iterrows():
            pc = int(r.get("monthlyPcQcCnt", 0))
            mo = int(r.get("monthlyMobileQcCnt", 0))
            rows.append({
                "국가": country,
                "키워드": r.get("relKeyword", ""),
                "PC검색수": pc,
                "모바일검색수": mo,
                "총검색수": pc + mo,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="📈 트렌드 데이터 조회 중...")
def load_trend_data(_keywords_tuple, start_date, end_date):
    """데이터랩 API로 국가별 키워드 트렌드 조회. 5개국씩 배치."""
    if not naver_datalab.is_available():
        return pd.DataFrame()
    keywords_by_country = dict(_keywords_tuple)
    countries = list(keywords_by_country.keys())
    all_dfs = []
    # 5개국씩 배치 (API 1회당 5개 keywordGroup 제한)
    for i in range(0, len(countries), 5):
        batch_countries = countries[i:i+5]
        groups = []
        for c in batch_countries:
            kw_list = keywords_by_country[c]
            if kw_list:
                groups.append({"groupName": c, "keywords": kw_list[:5]})
        if not groups:
            continue
        df = naver_datalab.fetch_trend(groups, start_date, end_date, time_unit="month")
        if not df.empty:
            df = df.rename(columns={"keyword": "국가"})
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ── 국가별 고정 색상 팔레트 ──────────────────────────
COUNTRY_COLORS = [
    "#667eea", "#f5576c", "#11998e", "#f093fb", "#36d1dc",
    "#ff9a9e", "#a18cd1", "#fbc2eb", "#84fab0", "#fccb90",
    "#e0c3fc", "#8fd3f4", "#d4fc79", "#96e6a1", "#dfe6e9",
    "#fab1a0", "#81ecec", "#74b9ff", "#fd79a8", "#ffeaa7",
]

def get_country_color_map(countries):
    return {c: COUNTRY_COLORS[i % len(COUNTRY_COLORS)] for i, c in enumerate(countries)}

# ── CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1929 0%, #142338 50%, #1a2d45 100%);
    min-width: 250px; max-width: 250px;
}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] p {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    max-height: 260px; overflow-y: auto;
}
/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #9ca3af !important;
    text-align: left !important;
    font-size: 0.85rem !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 20px 24px; color: white; text-align: center;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3); transition: transform 0.2s ease;
    height: 130px; display: flex; flex-direction: column; justify-content: center;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card.blue { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.kpi-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 15px rgba(17,153,142,0.3); }
.kpi-card.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); box-shadow: 0 4px 15px rgba(245,87,108,0.3); }
.kpi-card.red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); box-shadow: 0 4px 15px rgba(235,51,73,0.3); }
.kpi-label { font-size: 13px; opacity: 0.9; margin-bottom: 6px; font-weight: 500; letter-spacing: 0.5px; }
.kpi-value { font-size: 24px; font-weight: 700; line-height: 1.2; word-break: keep-all; }
.kpi-icon { font-size: 22px; margin-bottom: 4px; }
/* Insight Card */
.insight-card {
    border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;
    border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.insight-card.surge { background: #eef9ee; border-color: #2ecc71; }
.insight-card.drop { background: #fef4f4; border-color: #e74c3c; }
.insight-card.up { background: #eef6ff; border-color: #3498db; }
.insight-card.down { background: #fff8ee; border-color: #f39c12; }
.insight-title { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
.insight-body { font-size: 13px; color: #555; line-height: 1.5; }
/* Filter summary badges */
.badge {
    display: inline-block; background: #667eea; color: white;
    border-radius: 10px; padding: 2px 10px; font-size: 11px;
    font-weight: 600; margin-right: 6px;
}
/* Section Header */
.section-header {
    font-size: 18px; font-weight: 700; color: #1a1a2e;
    margin: 8px 0 12px 0; padding-bottom: 8px;
    border-bottom: 2px solid #667eea; display: inline-block;
}
/* Top Banner */
.top-banner {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white; padding: 16px 28px; border-radius: 12px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.top-banner .banner-icon { font-size: 32px; }
.top-banner .banner-text h2 { margin: 0; font-size: 22px; font-weight: 700; }
.top-banner .banner-text p { margin: 4px 0 0 0; font-size: 13px; opacity: 0.8; }
/* Source Indicator */
.source-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.source-dot.on { background: #2ecc71; }
.source-dot.off { background: #e74c3c; }
/* Download Button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 8px; padding: 8px 20px;
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── 세션 상태 초기화 ────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "forecast"

MENU_ITEMS = [
    {"key": "forecast",  "label": "🗺️ 수요 예측"},
    {"key": "query",     "label": "🔍 쿼리 분석"},
    {"key": "price",     "label": "💰 가격 비교 분석"},
    {"key": "discover",  "label": "🔎 키워드 발굴"},
    {"key": "settings",  "label": "⚙️ 설정"},
]

# ── 사이드바 ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:36px;">🌏</div>
        <div style="font-size:18px; font-weight:700; color:white; margin-top:4px;">Travel Demand Radar</div>
        <div style="font-size:12px; color:#aaa;">여행 수요 예측 플랫폼</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    for item in MENU_ITEMS:
        is_active = st.session_state["current_page"] == item["key"]
        if st.button(item["label"], key=f"nav_{item['key']}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            if not is_active:
                st.session_state["current_page"] = item["key"]
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="padding: 4px 8px; font-size: 12px;">
        <div style="color: #aaa; font-weight: 600; margin-bottom: 6px;">모니터링 소스</div>
    </div>
    """, unsafe_allow_html=True)
    sources = [
        ("네이버 데이터랩", naver_datalab.is_available()),
        ("네이버 검색광고", naver_searchad.is_available()),
        ("네이버 쇼핑", naver_shopping.is_available()),
        ("공공데이터포털", tourism_stats.is_available()),
    ]
    for name, avail in sources:
        dot_class = "on" if avail else "off"
        st.markdown(f'<div style="padding:2px 12px;font-size:12px;color:#ccc;"><span class="source-dot {dot_class}"></span>{name}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("v3.0 — 여행 수요 예측 플랫폼")


# ====================================================================
#  유틸리티 함수
# ====================================================================
def get_trend_slope(values):
    if len(values) < 2:
        return 0
    x = np.arange(len(values))
    return np.polyfit(x, values, 1)[0]

def api_or_fallback():
    """Return True if any API is available, False for fallback mode."""
    return any([
        naver_datalab.is_available(),
        naver_searchad.is_available(),
        tourism_stats.is_available(),
    ])


# ====================================================================
#  메뉴 1: 수요 예측
# ====================================================================
def page_forecast():
    country_map = load_json("country_mapping.json")
    keywords_data = load_json("trend_keywords.json")
    country_kw = keywords_data.get("국가별", {})

    # API 모드 판별
    use_api = naver_searchad.is_available()
    api_mode = "실시간 API" if use_api else "데모 모드 (Fallback CSV)"

    # 국가 목록: trend_keywords.json 기준
    국가_목록 = sorted(country_kw.keys())

    # 배너
    n_countries = len(국가_목록)
    n_keywords = sum(len(v) for v in country_kw.values())
    st.markdown(f"""
    <div class="top-banner">
        <div class="banner-icon">🗺️</div>
        <div class="banner-text">
            <h2>수요 예측</h2>
            <p>🏳️ {n_countries}개 국가 | 🔑 {n_keywords}개 키워드 | 📡 {api_mode}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    월_목록 = list(range(1, 13))

    # expander 열림 상태 추적
    if "fc_exp_month" not in st.session_state:
        st.session_state["fc_exp_month"] = False
    if "fc_exp_country" not in st.session_state:
        st.session_state["fc_exp_country"] = False

    with st.container(border=True):
        # 1행: 연도
        선택_연도 = st.selectbox("연도", [2024, 2025, 2026], index=2, key="fc_year")

        # 2행: 월 선택
        월_선택수 = sum(1 for m in 월_목록 if st.session_state.get(f"fc_m_{m}", True))
        with st.expander(f"📆 월 선택 ({월_선택수}개월 선택됨)", expanded=st.session_state["fc_exp_month"]):
            ma1, ma2, _ = st.columns([1, 1, 4])
            with ma1:
                if st.button("전체 선택", key="fc_m_all", use_container_width=True):
                    for m in 월_목록:
                        st.session_state[f"fc_m_{m}"] = True
                    st.session_state["fc_exp_month"] = True
                    st.rerun()
            with ma2:
                if st.button("전체 해제", key="fc_m_none", use_container_width=True):
                    for m in 월_목록:
                        st.session_state[f"fc_m_{m}"] = False
                    st.session_state["fc_exp_month"] = True
                    st.rerun()
            cols = st.columns(6)
            for i, m in enumerate(월_목록):
                with cols[i % 6]:
                    st.checkbox(f"{m}월", value=True, key=f"fc_m_{m}")
        선택_월 = [m for m in 월_목록 if st.session_state.get(f"fc_m_{m}", True)]

        # 3행: 국가 선택
        국가_선택수 = sum(1 for c in 국가_목록 if st.session_state.get(f"fc_c_{c}", True))
        with st.expander(f"🌍 국가 선택 ({국가_선택수}개국 선택됨)", expanded=st.session_state["fc_exp_country"]):
            ca1, ca2, _ = st.columns([1, 1, 4])
            with ca1:
                if st.button("전체 선택", key="fc_c_all", use_container_width=True):
                    for c in 국가_목록:
                        st.session_state[f"fc_c_{c}"] = True
                    st.session_state["fc_exp_country"] = True
                    st.rerun()
            with ca2:
                if st.button("전체 해제", key="fc_c_none", use_container_width=True):
                    for c in 국가_목록:
                        st.session_state[f"fc_c_{c}"] = False
                    st.session_state["fc_exp_country"] = True
                    st.rerun()
            cols = st.columns(5)
            for i, c in enumerate(국가_목록):
                with cols[i % 5]:
                    st.checkbox(c, value=True, key=f"fc_c_{c}")
        선택_국가 = [c for c in 국가_목록 if st.session_state.get(f"fc_c_{c}", True)]

        # 요약
        st.markdown(f"""
        <div style="padding: 6px 0 8px 0;">
            <span class="badge">{len(선택_국가)}개국</span>
            <span class="badge">{선택_연도}년</span>
            <span class="badge">{len(선택_월)}개월</span>
        </div>
        """, unsafe_allow_html=True)

    if not 선택_국가:
        st.warning("국가를 1개 이상 선택해주세요.")
        return
    if not 선택_월:
        st.warning("월을 1개 이상 선택해주세요.")
        return

    # ── 데이터 로드: API 우선, CSV fallback ───────
    # 선택 국가의 키워드 가져오기
    selected_kw = {c: country_kw.get(c, [f"{c}여행"]) for c in 선택_국가}
    kw_tuple = tuple(sorted(selected_kw.items()))  # cache key용

    if use_api:
        # 검색광고 API로 월간 검색수
        search_df = load_searchad_data(kw_tuple)
        # 데이터랩 API로 트렌드
        from datetime import date
        today = date.today().isoformat()
        end_dt = today if 선택_연도 >= date.today().year else f"{선택_연도}-12-31"
        trend_api_df = load_trend_data(kw_tuple, f"{선택_연도}-01-01", end_dt)
    else:
        search_df = pd.DataFrame()
        trend_api_df = pd.DataFrame()

    # API 데이터 또는 CSV fallback
    if not search_df.empty:
        # API 데이터로 국가별 검색량 집계
        country_search = search_df.groupby("국가")["총검색수"].sum().reset_index()
        country_search.columns = ["국가", "검색량"]
        data_source = "API"
    else:
        # CSV fallback
        fallback_df = load_fallback_data()
        fb_filtered = fallback_df[
            (fallback_df["국가"].isin(선택_국가)) &
            (fallback_df["연도"] == 선택_연도) &
            (fallback_df["월"].isin(선택_월))
        ]
        country_search = fb_filtered.groupby("국가")["쿼리수"].sum().reset_index()
        country_search.columns = ["국가", "검색량"]
        data_source = "CSV"

    # 선택했지만 데이터 없는 국가 추가
    for c in 선택_국가:
        if c not in country_search["국가"].values:
            country_search = pd.concat([country_search, pd.DataFrame([{"국가": c, "검색량": 0}])], ignore_index=True)

    # 국가 순위
    국가순위 = country_search.sort_values("검색량", ascending=False)["국가"].tolist()
    color_map = get_country_color_map(국가순위)
    데이터_국가 = [c for c in 국가순위 if country_search[country_search["국가"] == c]["검색량"].iloc[0] > 0]

    # ── KPI 카드 ─────────────────────────────────
    총검색 = int(country_search["검색량"].sum())
    최고국가 = country_search.loc[country_search["검색량"].idxmax(), "국가"] if 총검색 > 0 else "-"
    # 최다 검색 키워드
    if not search_df.empty:
        최고키워드 = search_df.loc[search_df["총검색수"].idxmax(), "키워드"] if not search_df.empty else "-"
    else:
        최고키워드 = "-"
    하락국가_이름 = "-"  # 트렌드 데이터에서 계산

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card blue"><div class="kpi-icon">📊</div><div class="kpi-label">전체 검색량 합계</div><div class="kpi-value">{총검색:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card green"><div class="kpi-icon">🏆</div><div class="kpi-label">최고 수요 국가</div><div class="kpi-value">{최고국가}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card orange"><div class="kpi-icon">🔑</div><div class="kpi-label">최다 검색 키워드</div><div class="kpi-value">{최고키워드}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card red"><div class="kpi-icon">⚠️</div><div class="kpi-label">데이터 소스</div><div class="kpi-value">{data_source} ({len(데이터_국가)}개국)</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 세계지도 히트맵 ──────────────────────────
    st.markdown('<div class="section-header">🗺️ 세계 여행 수요 예측 지도</div>', unsafe_allow_html=True)

    # demand_score 계산
    country_query = country_search.copy()
    max_s = country_query["검색량"].max()
    country_query["demand_score"] = (country_query["검색량"] / max_s * 100) if max_s > 0 else 0

    # ISO 코드 매핑
    country_query["iso_alpha3"] = country_query["국가"].map(
        lambda x: country_map.get(x, {}).get("iso_alpha3", "")
    )
    map_data = country_query[country_query["iso_alpha3"] != ""]

    if not map_data.empty:
        fig_map = go.Figure(data=go.Choropleth(
            locations=map_data["iso_alpha3"],
            z=map_data["demand_score"],
            text=map_data["국가"],
            customdata=np.stack([map_data["검색량"], map_data["demand_score"]], axis=-1),
            hovertemplate="<b>%{text}</b><br>수요 점수: %{customdata[1]:.1f}<br>검색량: %{customdata[0]:,.0f}<extra></extra>",
            colorscale=[
                [0, "#e8eaf6"], [0.2, "#7986cb"], [0.4, "#5c6bc0"],
                [0.6, "#3f51b5"], [0.8, "#e65100"], [1, "#ff6d00"],
            ],
            autocolorscale=False,
            marker_line_color="rgba(255,255,255,0.6)",
            marker_line_width=0.5,
            colorbar=dict(
                title=dict(text="수요 점수", font=dict(size=12)),
                thickness=12, len=0.6, tickformat=".0f",
                bgcolor="rgba(255,255,255,0.8)", borderwidth=0,
            ),
        ))
        fig_map.update_layout(
            height=480, margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(
                showframe=False,
                showcoastlines=True, coastlinecolor="rgba(180,180,200,0.4)", coastlinewidth=0.5,
                showland=True, landcolor="#f0f0f5",
                showocean=True, oceancolor="#fafbff",
                showlakes=True, lakecolor="#fafbff",
                showcountries=True, countrycolor="rgba(200,200,220,0.3)", countrywidth=0.3,
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # TOP 10 & 12개월 추이
    col_top, col_trend_map = st.columns([1, 2])
    with col_top:
        st.markdown("**🏆 TOP 10 핫 국가**")
        top10 = country_query.sort_values("demand_score", ascending=False).head(10)
        top10_display = top10[["국가", "검색량", "demand_score"]].copy()
        top10_display.columns = ["국가", "검색량", "수요점수"]
        top10_display["검색량"] = top10_display["검색량"].apply(lambda x: f"{int(x):,}")
        top10_display["수요점수"] = top10_display["수요점수"].apply(lambda x: f"{x:.1f}")
        top10_display.index = range(1, len(top10_display) + 1)
        st.dataframe(top10_display, use_container_width=True, height=380)

    with col_trend_map:
        st.markdown("**📊 상위 국가 트렌드**")
        if not trend_api_df.empty:
            # API 트렌드 데이터 사용
            top5 = 국가순위[:5]
            t5_df = trend_api_df[trend_api_df["국가"].isin(top5)].copy()
            if not t5_df.empty:
                t5_df["period_str"] = t5_df["period"].dt.strftime("%Y-%m")
                fig_trend12 = px.line(
                    t5_df, x="period_str", y="ratio", color="국가",
                    markers=True, color_discrete_map=color_map,
                    category_orders={"국가": top5},
                    labels={"period_str": "연월", "ratio": "검색 트렌드"},
                )
                fig_trend12.update_layout(
                    height=400, margin=dict(l=0, r=0, t=30, b=40),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    xaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickangle=-45, title=""),
                    yaxis=dict(gridcolor="rgba(0,0,0,0.08)", tickformat=",", title=""),
                )
                st.plotly_chart(fig_trend12, use_container_width=True)
            else:
                st.info("트렌드 데이터가 없습니다.")
        else:
            st.info("트렌드 API 미연결 상태입니다.")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 섹션 3: 수요 트렌드 (API 기반) ─────────────
    if not 데이터_국가:
        st.info("선택한 국가에 검색 데이터가 없습니다.")
        return

    st.markdown('<div class="section-header">📈 수요 트렌드</div>', unsafe_allow_html=True)
    col_chart, col_rank = st.columns([3, 1])

    with col_chart:
        if not trend_api_df.empty:
            trend_filtered = trend_api_df[trend_api_df["국가"].isin(데이터_국가)].copy()
            if not trend_filtered.empty:
                trend_filtered["period_str"] = trend_filtered["period"].dt.strftime("%Y-%m")
                # 선택 월 필터
                trend_filtered["월"] = trend_filtered["period"].dt.month
                trend_filtered = trend_filtered[trend_filtered["월"].isin(선택_월)]
                fig = px.line(
                    trend_filtered, x="period_str", y="ratio", color="국가",
                    markers=True, color_discrete_map=color_map,
                    category_orders={"국가": 데이터_국가},
                    labels={"period_str": "연월", "ratio": "검색 트렌드"},
                )
                fig.update_layout(
                    height=420, margin=dict(l=0, r=0, t=20, b=40),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    xaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickangle=-45, title=""),
                    yaxis=dict(gridcolor="rgba(0,0,0,0.08)", tickformat=",", title=""),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("선택 조건의 트렌드 데이터가 없습니다.")
        else:
            st.info("트렌드 API 데이터를 불러올 수 없습니다.")

    with col_rank:
        st.markdown("**🏅 국가 순위**")
        rank = country_search[country_search["검색량"] > 0].sort_values("검색량", ascending=False).copy()
        rank.columns = ["국가", "검색량"]
        rank["검색량"] = rank["검색량"].apply(lambda x: f"{int(x):,}")
        rank.index = range(1, len(rank) + 1)
        rank.index.name = "순위"
        st.dataframe(rank, use_container_width=True, height=420)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 섹션 4: 키워드별 검색량 히트맵 ─────────────
    if not search_df.empty:
        st.markdown('<div class="section-header">🔥 국가별 키워드 검색량</div>', unsafe_allow_html=True)

        # 국가별 키워드 검색량 히트맵
        heat_data_api = search_df[search_df["국가"].isin(데이터_국가)].copy()
        if not heat_data_api.empty:
            heat_pivot = heat_data_api.pivot_table(
                index="국가", columns="키워드", values="총검색수", aggfunc="sum"
            ).fillna(0)
            # 국가순위대로 정렬
            heat_pivot = heat_pivot.reindex(index=[c for c in 데이터_국가 if c in heat_pivot.index])

            text_display = []
            for row in heat_pivot.values:
                text_display.append([f"{int(v):,}" if v > 0 else "" for v in row])

            n_c = len(heat_pivot)
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_pivot.values, x=heat_pivot.columns.tolist(), y=heat_pivot.index.tolist(),
                text=text_display, texttemplate="%{text}", textfont={"size": 10},
                colorscale=[[0,"#f0f4ff"],[0.25,"#b8ccff"],[0.5,"#667eea"],[0.75,"#4a5acf"],[1,"#2d3a8c"]],
                hoverongaps=False,
                hovertemplate="국가: %{y}<br>키워드: %{x}<br>검색량: %{z:,}<extra></extra>",
                colorbar=dict(title="검색량", tickformat=","),
            ))
            fig_heat.update_layout(
                height=max(400, n_c * 35 + 100), margin=dict(l=80, r=10, t=10, b=100),
                xaxis=dict(tickangle=-45, side="bottom"),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # 국가별 총 검색량 상위/하위
        st.markdown("**국가별 검색량 순위**")
        rank_full = country_search[country_search["검색량"] > 0].sort_values("검색량", ascending=False).copy()
        rank_full["검색량"] = rank_full["검색량"].apply(lambda x: f"{int(x):,}")
        st.dataframe(rank_full, hide_index=True, use_container_width=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 섹션 5: 자동 인사이트 (트렌드 API 기반) ─────
    st.markdown('<div class="section-header">💡 자동 인사이트</div>', unsafe_allow_html=True)

    if not trend_api_df.empty:
        insights = []
        for country in 데이터_국가:
            c_data = trend_api_df[trend_api_df["국가"] == country].sort_values("period")
            c_vals = c_data["ratio"].values
            if len(c_vals) < 2:
                continue
            slope = get_trend_slope(c_vals)
            mean_val = c_vals.mean()
            slope_rate = (slope / mean_val) if mean_val > 0 else 0
            mom_rate = ((c_vals[-1] - c_vals[-2]) / c_vals[-2]) if c_vals[-2] > 0 else 0

            if mom_rate >= 0.2:
                insights.append(("surge", "⚡ 급등", country, f"전월 대비 {mom_rate*100:.0f}% 급등"))
            elif mom_rate <= -0.2:
                insights.append(("drop", "📉 급락", country, f"전월 대비 {abs(mom_rate)*100:.0f}% 급락"))
            elif slope_rate > 0.02:
                insights.append(("up", "📈 상승세", country, f"최근 {len(c_vals)}개월 우상향 추세"))
            elif slope_rate < -0.02:
                insights.append(("down", "⚠️ 하락세", country, f"최근 {len(c_vals)}개월 우하향 추세"))

        if insights:
            for group_type in ["surge", "drop", "up", "down"]:
                for css_class, badge, country, msg in [i for i in insights if i[0] == group_type]:
                    st.markdown(f'<div class="insight-card {css_class}"><div class="insight-title">{badge} {country}</div><div class="insight-body">{msg}</div></div>', unsafe_allow_html=True)
        else:
            st.write("현재 기간에서 특이 패턴이 감지되지 않았어요.")
    else:
        st.write("트렌드 데이터가 없어 인사이트를 생성할 수 없습니다.")


# ====================================================================
#  메뉴 2: 쿼리 분석
# ====================================================================
def page_query():
    st.markdown("""
    <div class="top-banner">
        <div class="banner-icon">🔍</div>
        <div class="banner-text">
            <h2>쿼리 분석</h2>
            <p>자사/국가별/경쟁사 키워드 검색 트렌드 분석</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    keywords_data = load_json("trend_keywords.json")
    company_info = load_json("company_info.json")
    df = load_fallback_data()

    from datetime import date, timedelta
    f1, f2, f3 = st.columns([2, 1.5, 1.5])
    with f1:
        분석유형 = st.selectbox("분석 유형", ["자사 분석", "국가별 분석", "경쟁사 분석"], key="q_type")
    with f2:
        시작일 = st.date_input("시작일", value=date.today() - timedelta(days=365), key="q_start")
    with f3:
        종료일 = st.date_input("종료일", value=date.today(), key="q_end")
    기간 = (시작일, 종료일)

    st.markdown("---")

    # ── 자사 분석 ──────────────────────────────────
    if 분석유형 == "자사 분석":
        own_keywords = company_info.get("자사", {}).get("brand_keywords", keywords_data.get("자사", []))
        if not own_keywords:
            st.info("설정에서 자사 브랜드 키워드를 등록해주세요.")
            return

        st.markdown('<div class="section-header">📊 자사 키워드 검색 트렌드</div>', unsafe_allow_html=True)

        if naver_datalab.is_available():
            start_str = str(기간[0])
            end_str = str(기간[1])
            time_unit = "month"
            trend_df = naver_datalab.fetch_trend(own_keywords, start_str, end_str, time_unit)
            if not trend_df.empty:
                fig = px.line(trend_df, x="period", y="ratio", color="keyword", markers=True,
                              labels={"period": "기간", "ratio": "검색 비율", "keyword": "키워드"})
                fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔑 네이버 데이터랩 API 키를 설정하면 실시간 트렌드를 확인할 수 있습니다. (현재: 데모 모드)")
            # 데모: fallback CSV에서 키워드 매칭되는 데이터로 트렌드 표시
            matched = df[df["키워드"].str.contains("|".join(own_keywords), na=False, regex=True)]
            if not matched.empty:
                demo_trend = matched.groupby(["키워드", "연월"])["쿼리수"].sum().reset_index()
                demo_trend["연월_str"] = demo_trend["연월"].astype(str)
                demo_trend = demo_trend.sort_values(["키워드", "연월_str"])
                fig = px.line(demo_trend, x="연월_str", y="쿼리수", color="키워드", markers=True,
                              labels={"연월_str": "연월", "쿼리수": "검색량"})
                fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(tickangle=-45))
                st.plotly_chart(fig, use_container_width=True)

        if naver_searchad.is_available():
            st.markdown('<div class="section-header">📈 월간 검색량</div>', unsafe_allow_html=True)
            with st.spinner("검색량 조회 중..."):
                stats_df = naver_searchad.get_keyword_stats(own_keywords)
            if not stats_df.empty and "relKeyword" in stats_df.columns:
                # 입력 키워드만 필터
                kw_lower = set(k.lower() for k in own_keywords)
                filtered_stats = stats_df[stats_df["relKeyword"].str.lower().isin(kw_lower)].copy()
                if filtered_stats.empty:
                    filtered_stats = stats_df.head(len(own_keywords))
                display_cols = ["relKeyword", "monthlyPcQcCnt", "monthlyMobileQcCnt"]
                existing = [c for c in display_cols if c in filtered_stats.columns]
                if existing:
                    show_df = filtered_stats[existing].copy()
                    show_df.columns = ["키워드", "PC 검색량", "모바일 검색량"][:len(existing)]
                    if "PC 검색량" in show_df.columns and "모바일 검색량" in show_df.columns:
                        show_df["총 검색량"] = show_df["PC 검색량"] + show_df["모바일 검색량"]
                        chart_df = show_df.melt(id_vars="키워드", value_vars=["PC 검색량", "모바일 검색량"], var_name="구분", value_name="검색량")
                        fig_bar = px.bar(chart_df, x="키워드", y="검색량", color="구분", barmode="group")
                        fig_bar.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    st.dataframe(show_df, hide_index=True, use_container_width=True)
            else:
                st.info("검색량 데이터가 없습니다.")
        else:
            st.info("🔑 네이버 검색광고 API 키를 설정하면 월간 검색량을 확인할 수 있습니다.")

    # ── 국가별 분석 ────────────────────────────────
    elif 분석유형 == "국가별 분석":
        국가_목록 = sorted(df["국가"].dropna().unique())
        선택국가 = st.selectbox("국가 선택", 국가_목록, key="q_country")

        국가순위 = df[df["국가"] == 선택국가].groupby("국가")["쿼리수"].sum().sort_values(ascending=False).index.tolist()
        color_map = get_country_color_map(국가순위)

        st.markdown(f'<div class="section-header">🔍 {선택국가} 키워드 분석 (상위 25개)</div>', unsafe_allow_html=True)

        kw_data = (
            df[df["국가"] == 선택국가]
            .groupby("키워드")["쿼리수"].sum().reset_index()
            .sort_values("쿼리수", ascending=False)
        )
        top25 = kw_data.head(25)
        전체_키워드 = kw_data["키워드"].tolist()
        기본_키워드 = 전체_키워드[:5]

        선택_키워드 = st.multiselect(
            "트렌드 확인할 키워드", options=전체_키워드, default=기본_키워드, key="q_kw_select"
        )

        # 수평 바차트
        top25_sorted = top25.sort_values("쿼리수", ascending=True)
        max_val = top25_sorted["쿼리수"].max() if not top25_sorted.empty else 1
        bar_colors = []
        for val in top25_sorted["쿼리수"]:
            ratio = val / max_val
            r = int(102 + (245 - 102) * ratio)
            g = int(126 + (87 - 126) * ratio)
            b = int(234 + (108 - 234) * ratio)
            bar_colors.append(f"rgb({r},{g},{b})")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=top25_sorted["쿼리수"], y=top25_sorted["키워드"], orientation="h",
            text=top25_sorted["쿼리수"].apply(lambda x: f"{int(x):,}"),
            textposition="outside",
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="키워드: %{y}<br>쿼리수: %{x:,}<extra></extra>",
        ))
        fig_bar.update_layout(
            height=650, margin=dict(l=120, r=80, t=20, b=40),
            xaxis=dict(title="쿼리수", tickformat=",", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="", automargin=True),
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # 키워드 트렌드
        if 선택_키워드:
            st.markdown("**키워드별 수요 트렌드**")
            kw_filtered = df[(df["국가"] == 선택국가) & (df["키워드"].isin(선택_키워드))]
            kw_trend = kw_filtered.groupby(["키워드", "연월"])["쿼리수"].sum().reset_index()
            kw_trend["연월_str"] = kw_trend["연월"].astype(str)
            kw_trend = kw_trend.sort_values(["키워드", "연월_str"])
            fig_kw = px.line(kw_trend, x="연월_str", y="쿼리수", color="키워드", markers=True,
                             labels={"연월_str": "연월", "쿼리수": "쿼리 수"})
            fig_kw.update_xaxes(tickangle=-45)

            fig_kw.update_layout(
                height=380, margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.08)", tickformat=","),
            )
            st.plotly_chart(fig_kw, use_container_width=True)

        # 연관키워드 (검색광고 API)
        if naver_searchad.is_available() and 선택_키워드:
            st.markdown('<div class="section-header">🏷️ 연관 키워드</div>', unsafe_allow_html=True)
            rel_df = naver_searchad.get_keyword_stats(선택_키워드[:3])
            if not rel_df.empty and "relKeyword" in rel_df.columns:
                rel_tags = rel_df["relKeyword"].head(30).tolist()
                tag_html = " ".join([
                    f'<span style="display:inline-block;background:#eef;border-radius:12px;padding:4px 12px;margin:3px;font-size:13px;">{t}</span>'
                    for t in rel_tags
                ])
                st.markdown(tag_html, unsafe_allow_html=True)

        # CSV 다운로드
        dl_data = df[df["국가"] == 선택국가]
        st.download_button(
            "📥 데이터 다운로드",
            dl_data.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"travel_demand_{선택국가}.csv",
            mime="text/csv",
        )

    # ── 경쟁사 분석 ────────────────────────────────
    elif 분석유형 == "경쟁사 분석":
        own_kw = company_info.get("자사", {}).get("brand_keywords", [])
        competitors = company_info.get("경쟁사", [])

        # 경쟁사 브랜드별 키워드 선택
        brand_options = ["전체 경쟁사"] + [c["name"] for c in competitors]
        선택_브랜드 = st.multiselect("경쟁사 브랜드 선택", brand_options, default=["전체 경쟁사"], key="q_comp_brands")

        comp_kw = []
        if "전체 경쟁사" in 선택_브랜드:
            for comp in competitors:
                comp_kw.extend(comp.get("brand_keywords", []))
        else:
            for comp in competitors:
                if comp["name"] in 선택_브랜드:
                    comp_kw.extend(comp.get("brand_keywords", []))

        all_kw = own_kw + comp_kw
        if not all_kw:
            st.info("설정에서 자사/경쟁사 키워드를 등록해주세요.")
            return

        st.markdown('<div class="section-header">📊 자사 vs 경쟁사 검색 트렌드 비교</div>', unsafe_allow_html=True)

        if naver_datalab.is_available():
            start_str = str(기간[0])
            end_str = str(기간[1])
            time_unit = "month"
            trend_df = naver_datalab.fetch_trend(all_kw, start_str, end_str, time_unit)
            if not trend_df.empty:
                # 자사/경쟁사 구분 추가
                trend_df["구분"] = trend_df["keyword"].apply(
                    lambda x: "자사" if x in own_kw else "경쟁사"
                )
                fig = px.line(trend_df, x="period", y="ratio", color="keyword",
                              line_dash="구분", markers=True,
                              labels={"period": "기간", "ratio": "검색 비율", "keyword": "키워드"})
                fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                # 점유율 파이차트
                st.markdown("**검색량 점유율**")
                share = trend_df.groupby("keyword")["ratio"].sum().reset_index()
                share.columns = ["키워드", "누적비율"]
                fig_pie = px.pie(share, names="키워드", values="누적비율", hole=0.4)
                fig_pie.update_layout(height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("🔑 네이버 데이터랩 API 키를 설정하면 실시간 경쟁사 비교가 가능합니다. (현재: 데모 모드)")
            # 데모: fallback CSV에서 키워드 매칭
            matched = df[df["키워드"].str.contains("|".join(all_kw), na=False, regex=True)]
            if not matched.empty:
                demo_trend = matched.groupby(["키워드", "연월"])["쿼리수"].sum().reset_index()
                demo_trend["연월_str"] = demo_trend["연월"].astype(str)
                demo_trend["구분"] = demo_trend["키워드"].apply(lambda x: "자사" if x in own_kw else "경쟁사")
                demo_trend = demo_trend.sort_values(["키워드", "연월_str"])
                fig = px.line(demo_trend, x="연월_str", y="쿼리수", color="키워드",
                              line_dash="구분", markers=True,
                              labels={"연월_str": "연월", "쿼리수": "검색량"})
                fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(tickangle=-45))
                st.plotly_chart(fig, use_container_width=True)

                # 점유율 파이차트
                st.markdown("**검색량 점유율**")
                share = matched.groupby("키워드")["쿼리수"].sum().reset_index()
                share.columns = ["키워드", "검색량"]
                fig_pie = px.pie(share, names="키워드", values="검색량", hole=0.4)
                fig_pie.update_layout(height=350)
                st.plotly_chart(fig_pie, use_container_width=True)

        if naver_searchad.is_available():
            st.markdown('<div class="section-header">📈 키워드별 검색량 비교</div>', unsafe_allow_html=True)
            stats_df = naver_searchad.get_keyword_stats(all_kw)
            if not stats_df.empty and "relKeyword" in stats_df.columns:
                # 자사/경쟁사 매칭
                filtered_stats = stats_df[stats_df["relKeyword"].isin(all_kw)].copy()
                filtered_stats["구분"] = filtered_stats["relKeyword"].apply(
                    lambda x: "자사" if x in own_kw else "경쟁사"
                )
                display_cols = ["relKeyword", "monthlyPcQcCnt", "monthlyMobileQcCnt", "구분"]
                existing = [c for c in display_cols if c in filtered_stats.columns]
                show_df = filtered_stats[existing].copy()
                rename_map = {"relKeyword": "키워드", "monthlyPcQcCnt": "PC 검색량", "monthlyMobileQcCnt": "모바일 검색량"}
                show_df = show_df.rename(columns=rename_map)
                st.dataframe(show_df, hide_index=True, use_container_width=True)
        else:
            st.info("🔑 네이버 검색광고 API 키를 설정하면 검색량 비교가 가능합니다.")


# ====================================================================
#  메뉴 3: 가격 비교 분석
# ====================================================================
def page_price():
    st.markdown("""
    <div class="top-banner">
        <div class="banner-icon">💰</div>
        <div class="banner-text">
            <h2>가격 비교 분석</h2>
            <p>네이버 쇼핑 검색 기반 자사/경쟁사 가격 비교</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    company_info = load_json("company_info.json")
    own_malls = company_info.get("자사", {}).get("mall_names", [])
    competitors = company_info.get("경쟁사", [])

    # 검색 입력
    f1, f2, f3 = st.columns([4, 1, 1])
    with f1:
        query = st.text_input("검색 키워드", placeholder="예: 도쿄 여행 패스", key="price_query")
    with f2:
        sort_opt = st.selectbox("정렬", ["관련도순", "가격낮은순", "가격높은순", "날짜순"], key="price_sort")
    with f3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        search_btn = st.button("🔍 검색", key="price_search", use_container_width=True)

    sort_map = {"관련도순": "sim", "가격낮은순": "asc", "가격높은순": "dsc", "날짜순": "date"}

    if not query:
        st.info("검색 키워드를 입력하고 검색 버튼을 눌러주세요.")
        return

    if not naver_shopping.is_available():
        st.warning("🔑 네이버 쇼핑 API 키가 설정되지 않았습니다. 설정 > API 키 관리에서 등록해주세요.")
        # 데모 데이터
        st.markdown("---")
        st.markdown("**💡 데모 모드: 샘플 데이터로 UI를 보여드립니다.**")
        demo_items = [
            {"title": f"{query} 상품 A", "mallName": own_malls[0] if own_malls else "자사몰", "lprice": 89000, "category1": "여행", "link": "#", "company_type": "자사", "company_name": own_malls[0] if own_malls else "자사몰"},
            {"title": f"{query} 상품 B", "mallName": "야놀자", "lprice": 92000, "category1": "여행", "link": "#", "company_type": "경쟁사", "company_name": "야놀자"},
            {"title": f"{query} 상품 C", "mallName": "마이리얼트립", "lprice": 91500, "category1": "여행", "link": "#", "company_type": "경쟁사", "company_name": "마이리얼트립"},
            {"title": f"{query} 상품 D", "mallName": "트립닷컴", "lprice": 88500, "category1": "여행", "link": "#", "company_type": "경쟁사", "company_name": "트립닷컴"},
            {"title": f"{query} 상품 E", "mallName": own_malls[0] if own_malls else "자사몰", "lprice": 85000, "category1": "여행", "link": "#", "company_type": "자사", "company_name": own_malls[0] if own_malls else "자사몰"},
        ]
        results_df = pd.DataFrame(demo_items)
        _render_price_analysis(results_df, own_malls, competitors)
        return

    # 실제 API 호출
    with st.spinner("네이버 쇼핑 검색 중..."):
        items = naver_shopping.search_shopping(query, display=100, sort=sort_map.get(sort_opt, "sim"))
    if not items:
        st.warning("검색 결과가 없습니다. 네이버 개발자센터에서 '검색' API 권한이 활성화되어 있는지 확인해주세요.")
        return

    # 자사/경쟁사 분류
    for item in items:
        item["company_type"] = "기타"
        item["company_name"] = item.get("mallName", "")
        if any(m.lower() in item.get("mallName", "").lower() for m in own_malls):
            item["company_type"] = "자사"
            item["company_name"] = company_info.get("자사", {}).get("name", "자사")
        else:
            for comp in competitors:
                if any(m.lower() in item.get("mallName", "").lower() for m in comp.get("mall_names", [])):
                    item["company_type"] = "경쟁사"
                    item["company_name"] = comp["name"]
                    break

    results_df = pd.DataFrame(items)
    _render_price_analysis(results_df, own_malls, competitors)


def _render_price_analysis(results_df, own_malls, competitors):
    """Render price comparison charts and tables."""

    # 업체별 가격 비교 바차트
    st.markdown('<div class="section-header">📊 업체별 가격 비교</div>', unsafe_allow_html=True)

    # 자사/경쟁사만 필터
    known = results_df[results_df["company_type"].isin(["자사", "경쟁사"])]
    if not known.empty:
        avg_prices = known.groupby("company_name")["lprice"].mean().reset_index()
        avg_prices.columns = ["업체명", "평균가격"]
        avg_prices = avg_prices.sort_values("평균가격")

        # 색상 지정
        color_list = []
        for name in avg_prices["업체명"]:
            if any(m in name for m in own_malls) or name == load_json("company_info.json").get("자사", {}).get("name", ""):
                color_list.append("#667eea")
            else:
                color_list.append("#f5576c")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=avg_prices["업체명"], y=avg_prices["평균가격"],
            text=avg_prices["평균가격"].apply(lambda x: f"₩{int(x):,}"),
            textposition="outside",
            marker_color=color_list,
        ))
        fig_bar.update_layout(
            height=350, margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(title="평균 가격 (원)", tickformat=","),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 상품 비교 테이블
    st.markdown('<div class="section-header">📋 상품 비교 테이블</div>', unsafe_allow_html=True)
    table_cols = ["company_type", "company_name", "title", "lprice", "mallName"]
    existing = [c for c in table_cols if c in results_df.columns]
    display_df = results_df[existing].copy()
    rename = {"company_type": "구분", "company_name": "업체명", "title": "상품명", "lprice": "가격", "mallName": "판매처"}
    display_df = display_df.rename(columns=rename)
    if "가격" in display_df.columns:
        display_df["가격"] = display_df["가격"].apply(lambda x: f"₩{int(x):,}" if pd.notna(x) and x > 0 else "-")
    st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)

    # 가격 분포 & 인사이트
    col_dist, col_insight = st.columns(2)

    with col_dist:
        st.markdown('<div class="section-header">📊 가격 분포</div>', unsafe_allow_html=True)
        known_for_box = results_df[results_df["company_type"].isin(["자사", "경쟁사"])]
        if not known_for_box.empty and "lprice" in known_for_box.columns:
            fig_box = px.box(
                known_for_box, x="company_type", y="lprice", color="company_type",
                color_discrete_map={"자사": "#667eea", "경쟁사": "#f5576c"},
                labels={"company_type": "구분", "lprice": "가격"},
            )
            fig_box.update_layout(
                height=350, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(tickformat=","),
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with col_insight:
        st.markdown('<div class="section-header">💡 가격 인사이트</div>', unsafe_allow_html=True)
        own_items = results_df[results_df["company_type"] == "자사"]
        comp_items = results_df[results_df["company_type"] == "경쟁사"]

        if not own_items.empty and not comp_items.empty:
            own_avg = own_items["lprice"].mean()
            comp_avg = comp_items["lprice"].mean()
            diff_pct = ((own_avg - comp_avg) / comp_avg) * 100

            if diff_pct < 0:
                st.success(f"✅ 자사가 경쟁사 대비 평균 **{abs(diff_pct):.1f}%** 저렴합니다.")
            elif diff_pct > 0:
                st.warning(f"⚠️ 자사가 경쟁사 대비 평균 **{diff_pct:.1f}%** 비쌉니다.")
            else:
                st.info("자사와 경쟁사 평균 가격이 동일합니다.")

            st.metric("자사 평균 가격", f"₩{int(own_avg):,}")
            st.metric("경쟁사 평균 가격", f"₩{int(comp_avg):,}")
            st.metric("가격 차이", f"₩{int(own_avg - comp_avg):,}", delta=f"{diff_pct:+.1f}%", delta_color="inverse")
        elif own_items.empty:
            st.info("자사 상품이 검색 결과에 없습니다.")
        else:
            st.info("경쟁사 상품이 검색 결과에 없습니다.")

    # CSV 내보내기
    st.markdown("---")
    st.download_button(
        "📥 비교 결과 다운로드",
        results_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="price_comparison.csv",
        mime="text/csv",
    )


# ====================================================================
#  메뉴 4: 키워드 발굴
# ====================================================================
def page_discover():
    st.markdown("""
    <div class="top-banner">
        <div class="banner-icon">🔎</div>
        <div class="banner-text">
            <h2>신규 키워드 발굴</h2>
            <p>검색광고 API를 통해 연관 키워드를 자동 조회하고 키워드 DB에 추가합니다.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    seed_kw = st.text_input("시드 키워드", placeholder="예: 일본여행", key="newkw_seed")

    if st.button("🔍 연관 키워드 조회", key="newkw_search", use_container_width=False) and seed_kw:
        if naver_searchad.is_available():
            with st.spinner("연관 키워드 조회 중..."):
                stats_df = naver_searchad.get_keyword_stats([seed_kw])
            if not stats_df.empty:
                col_mapping = {
                    "relKeyword": "키워드",
                    "monthlyPcQcCnt": "PC 검색량",
                    "monthlyMobileQcCnt": "모바일 검색량",
                    "compIdx": "경쟁강도",
                }
                display_cols = [c for c in col_mapping if c in stats_df.columns]
                show_df = stats_df[display_cols].copy().rename(columns=col_mapping)
                if "PC 검색량" in show_df.columns and "모바일 검색량" in show_df.columns:
                    show_df["총 검색량"] = show_df["PC 검색량"] + show_df["모바일 검색량"]
                    show_df = show_df.sort_values("총 검색량", ascending=False)

                st.dataframe(show_df, hide_index=True, use_container_width=True, height=500)

                # 원클릭 추가
                st.markdown("---")
                kw_data = load_json("trend_keywords.json")
                categories = ["자사", "경쟁사", "시즌"]
                country_kw = kw_data.get("국가별", {})
                categories.extend([f"국가별/{c}" for c in sorted(country_kw.keys())])

                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    add_kws = st.multiselect("추가할 키워드", show_df["키워드"].tolist() if "키워드" in show_df.columns else [], key="newkw_add_list")
                with c2:
                    target_cat = st.selectbox("추가 대상 카테고리", categories, key="newkw_target")
                with c3:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    if st.button("➕ 추가", key="newkw_add_btn") and add_kws:
                        if "/" in target_cat:
                            _, country = target_cat.split("/", 1)
                            kw_data["국가별"][country].extend(add_kws)
                        else:
                            if target_cat not in kw_data:
                                kw_data[target_cat] = []
                            kw_data[target_cat].extend(add_kws)
                        save_json("trend_keywords.json", kw_data)
                        st.success(f"{len(add_kws)}개 키워드를 {target_cat}에 추가했습니다!")
                        st.rerun()
            else:
                st.warning("결과가 없습니다.")
        else:
            st.warning("🔑 네이버 검색광고 API 키를 설정해주세요.")


# ====================================================================
#  메뉴 5: 설정
# ====================================================================
def page_settings():
    st.markdown("""
    <div class="top-banner">
        <div class="banner-icon">⚙️</div>
        <div class="banner-text">
            <h2>설정</h2>
            <p>API 키, 키워드, 국가/지역 관리</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_company, tab_kw, tab_api, tab_country = st.tabs([
        "🏢 자사/경쟁사 관리", "📝 키워드 관리",
        "🔑 API 키 관리", "🌍 국가/지역 관리",
    ])

    # ── 탭1: 자사/경쟁사 관리 ──────────────────────
    with tab_company:
        company_info = load_json("company_info.json")

        st.markdown("### 자사 정보")
        own = company_info.get("자사", {"name": "", "mall_names": [], "brand_keywords": []})

        own_name = st.text_input("회사명", value=own.get("name", ""), key="set_own_name")

        st.markdown("**쇼핑몰명**")
        own_malls = own.get("mall_names", [])
        # 태그 표시
        mall_display = " · ".join([f"`{m}`" for m in own_malls]) if own_malls else "(없음)"
        st.markdown(mall_display)
        c1, c2 = st.columns([3, 1])
        with c1:
            new_mall = st.text_input("추가할 쇼핑몰명", key="set_new_mall")
        with c2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("추가", key="set_add_mall") and new_mall:
                own_malls.append(new_mall)
                company_info["자사"] = {"name": own_name, "mall_names": own_malls, "brand_keywords": own.get("brand_keywords", [])}
                save_json("company_info.json", company_info)
                st.rerun()
        if own_malls:
            del_mall = st.selectbox("삭제할 쇼핑몰명", own_malls, key="set_del_mall")
            if st.button("선택 삭제", key="set_rm_mall"):
                own_malls.remove(del_mall)
                company_info["자사"]["mall_names"] = own_malls
                save_json("company_info.json", company_info)
                st.rerun()

        st.markdown("**브랜드 키워드**")
        own_bkw = own.get("brand_keywords", [])
        bkw_display = " · ".join([f"`{k}`" for k in own_bkw]) if own_bkw else "(없음)"
        st.markdown(bkw_display)
        c1, c2 = st.columns([3, 1])
        with c1:
            new_bkw = st.text_input("추가할 브랜드 키워드", key="set_new_bkw")
        with c2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("추가", key="set_add_bkw") and new_bkw:
                own_bkw.append(new_bkw)
                company_info["자사"]["brand_keywords"] = own_bkw
                save_json("company_info.json", company_info)
                st.rerun()

        # 자사 정보 저장
        if st.button("💾 자사 정보 저장", key="set_save_own"):
            company_info["자사"] = {"name": own_name, "mall_names": own_malls, "brand_keywords": own_bkw}
            save_json("company_info.json", company_info)
            st.success("저장 완료!")

        st.markdown("---")
        st.markdown("### 경쟁사 목록")
        competitors = company_info.get("경쟁사", [])

        for idx, comp in enumerate(competitors):
            with st.expander(comp['name'], expanded=False):
                st.markdown(f"**쇼핑몰명**: {' · '.join([f'`{m}`' for m in comp.get('mall_names', [])])}")
                st.markdown(f"**브랜드 키워드**: {' · '.join([f'`{k}`' for k in comp.get('brand_keywords', [])])}")

                c1, c2 = st.columns(2)
                with c1:
                    edit_name = st.text_input("회사명", value=comp["name"], key=f"comp_name_{idx}")
                    edit_malls = st.text_input("쇼핑몰명 (쉼표구분)", value=",".join(comp.get("mall_names", [])), key=f"comp_mall_{idx}")
                    edit_bkw = st.text_input("브랜드키워드 (쉼표구분)", value=",".join(comp.get("brand_keywords", [])), key=f"comp_bkw_{idx}")
                    if st.button("편집 저장", key=f"comp_save_{idx}"):
                        competitors[idx] = {
                            "name": edit_name,
                            "mall_names": [m.strip() for m in edit_malls.split(",") if m.strip()],
                            "brand_keywords": [k.strip() for k in edit_bkw.split(",") if k.strip()],
                        }
                        company_info["경쟁사"] = competitors
                        save_json("company_info.json", company_info)
                        st.success(f"{edit_name} 저장 완료!")
                        st.rerun()
                with c2:
                    if st.button(f"🗑️ {comp['name']} 삭제", key=f"comp_del_{idx}"):
                        competitors.pop(idx)
                        company_info["경쟁사"] = competitors
                        save_json("company_info.json", company_info)
                        st.rerun()

        st.markdown("**+ 경쟁사 추가**")
        c1, c2, c3 = st.columns(3)
        with c1:
            add_comp_name = st.text_input("경쟁사명", key="add_comp_name")
        with c2:
            add_comp_malls = st.text_input("쇼핑몰명 (쉼표구분)", key="add_comp_malls")
        with c3:
            add_comp_bkw = st.text_input("브랜드키워드 (쉼표구분)", key="add_comp_bkw")
        if st.button("➕ 경쟁사 추가", key="add_comp_btn") and add_comp_name:
            competitors.append({
                "name": add_comp_name,
                "mall_names": [m.strip() for m in add_comp_malls.split(",") if m.strip()],
                "brand_keywords": [k.strip() for k in add_comp_bkw.split(",") if k.strip()],
            })
            company_info["경쟁사"] = competitors
            save_json("company_info.json", company_info)
            st.success(f"{add_comp_name} 추가 완료!")
            st.rerun()

    # ── 탭2: 키워드 관리 (CKD 패턴: form + checkbox 삭제) ──
    with tab_kw:
        kw_data = load_json("trend_keywords.json")

        def _save_kw(kw_data):
            save_json("trend_keywords.json", kw_data)

        def _render_kw_section(cat_key, keywords, kw_data, prefix):
            """CKD 패턴: 태그 표시 → form으로 추가 → form으로 체크박스 삭제"""
            import re
            # 현재 키워드 태그 표시
            if keywords:
                st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
            else:
                st.caption("키워드 없음")

            # 추가 form
            with st.form(f"kwadd_{prefix}"):
                new_text = st.text_area("추가할 키워드 (쉼표/줄바꿈 구분)", height=60, key=f"kwinput_{prefix}")
                if st.form_submit_button("추가", type="primary"):
                    parsed = [k.strip() for k in re.split(r"[,;\n\r]+", new_text) if k.strip()]
                    added = [k for k in parsed if k not in keywords]
                    if added:
                        keywords.extend(added)
                        if "/" in cat_key:
                            _, country = cat_key.split("/", 1)
                            kw_data["국가별"][country] = keywords
                        else:
                            kw_data[cat_key] = keywords
                        _save_kw(kw_data)
                        st.rerun()

            # 삭제 form (체크박스 선택)
            if keywords:
                with st.form(f"kwdel_{prefix}"):
                    st.caption("삭제할 키워드를 선택하세요:")
                    checks = {}
                    cols = st.columns(min(len(keywords), 4))
                    for i, kw in enumerate(keywords):
                        with cols[i % len(cols)]:
                            checks[kw] = st.checkbox(kw, key=f"kwchk_{prefix}_{kw}")
                    if st.form_submit_button("선택 삭제"):
                        to_del = [kw for kw, v in checks.items() if v]
                        if to_del:
                            remaining = [k for k in keywords if k not in to_del]
                            if "/" in cat_key:
                                _, country = cat_key.split("/", 1)
                                kw_data["국가별"][country] = remaining
                            else:
                                kw_data[cat_key] = remaining
                            _save_kw(kw_data)
                            st.rerun()

        # 일반 카테고리
        for category in ["자사", "경쟁사", "시즌"]:
            keywords = kw_data.get(category, [])
            with st.expander(f"📁 {category} ({len(keywords)}개)", expanded=False):
                _render_kw_section(category, keywords, kw_data, category)

        # 국가별 키워드
        country_kw = kw_data.get("국가별", {})
        for country in sorted(country_kw.keys()):
            keywords = country_kw[country]
            with st.expander(f"🌍 {country} ({len(keywords)}개)", expanded=False):
                _render_kw_section(f"국가별/{country}", keywords, kw_data, f"c_{country}")

        # 국가 추가
        st.markdown("---")
        with st.form("kw_add_country"):
            new_country = st.text_input("새 국가 카테고리 추가")
            if st.form_submit_button("추가"):
                if new_country:
                    if "국가별" not in kw_data:
                        kw_data["국가별"] = {}
                    kw_data["국가별"][new_country] = []
                    _save_kw(kw_data)
                    st.rerun()

    # ── 탭3: 신규 키워드 발굴 ──────────────────────
    # ── 탭3: API 키 관리 ───────────────────────────
    with tab_api:
        st.markdown("### 🔑 API 키 관리")
        st.info("`.env` 파일에 설정된 키가 우선 적용됩니다. 아래에서 세션 중 오버라이드할 수 있습니다.")

        st.markdown("#### 네이버 데이터랩 / 쇼핑")
        c1, c2 = st.columns(2)
        with c1:
            nv_id = st.text_input("Client ID", value=os.environ.get("NAVER_CLIENT_ID", ""), key="api_nv_id", type="password")
        with c2:
            nv_secret = st.text_input("Client Secret", value=os.environ.get("NAVER_CLIENT_SECRET", ""), key="api_nv_secret", type="password")
        if st.button("💾 저장 & 테스트", key="api_nv_save"):
            if nv_id:
                os.environ["NAVER_CLIENT_ID"] = nv_id
            if nv_secret:
                os.environ["NAVER_CLIENT_SECRET"] = nv_secret
            ok, msg = naver_datalab.test_connection()
            if ok:
                st.success(f"✅ 네이버 데이터랩: {msg}")
            else:
                st.error(f"❌ 네이버 데이터랩: {msg}")

        st.markdown("---")
        st.markdown("#### 네이버 검색광고")
        c1, c2, c3 = st.columns(3)
        with c1:
            sa_key = st.text_input("API Key", value=os.environ.get("NAVER_SEARCHAD_API_KEY", ""), key="api_sa_key", type="password")
        with c2:
            sa_secret = st.text_input("Secret Key", value=os.environ.get("NAVER_SEARCHAD_SECRET_KEY", ""), key="api_sa_secret", type="password")
        with c3:
            sa_cid = st.text_input("Customer ID", value=os.environ.get("NAVER_SEARCHAD_CUSTOMER_ID", ""), key="api_sa_cid", type="password")
        if st.button("💾 저장 & 테스트", key="api_sa_save"):
            if sa_key:
                os.environ["NAVER_SEARCHAD_API_KEY"] = sa_key
            if sa_secret:
                os.environ["NAVER_SEARCHAD_SECRET_KEY"] = sa_secret
            if sa_cid:
                os.environ["NAVER_SEARCHAD_CUSTOMER_ID"] = sa_cid
            ok, msg = naver_searchad.test_connection()
            if ok:
                st.success(f"✅ 네이버 검색광고: {msg}")
            else:
                st.error(f"❌ 네이버 검색광고: {msg}")

        st.markdown("---")
        st.markdown("#### 공공데이터포털")
        pub_key = st.text_input("서비스 키", value=os.environ.get("PUBLIC_DATA_SERVICE_KEY", ""), key="api_pub_key", type="password")
        if st.button("💾 저장 & 테스트", key="api_pub_save"):
            if pub_key:
                os.environ["PUBLIC_DATA_SERVICE_KEY"] = pub_key
            ok, msg = tourism_stats.test_connection()
            if ok:
                st.success(f"✅ 공공데이터포털: {msg}")
            else:
                st.error(f"❌ 공공데이터포털: {msg}")

        st.markdown("---")
        st.markdown("#### 연결 상태 요약")
        status_data = []
        for name, check_fn in [
            ("네이버 데이터랩", naver_datalab.is_available),
            ("네이버 검색광고", naver_searchad.is_available),
            ("네이버 쇼핑", naver_shopping.is_available),
            ("공공데이터포털", tourism_stats.is_available),
        ]:
            avail = check_fn()
            status_data.append({"API": name, "상태": "✅ 연결됨" if avail else "❌ 미설정"})
        st.dataframe(pd.DataFrame(status_data), hide_index=True, use_container_width=True)

    # ── 탭5: 국가/지역 관리 ────────────────────────
    with tab_country:
        st.markdown("### 🌍 국가/지역 관리")
        country_map = load_json("country_mapping.json")

        st.markdown(f"현재 등록된 국가: **{len(country_map)}개**")

        # 테이블 표시
        rows = []
        for kr_name, info in country_map.items():
            rows.append({
                "국가명(한)": kr_name,
                "ISO Alpha-3": info.get("iso_alpha3", ""),
                "ISO Alpha-2": info.get("iso_alpha2", ""),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=400)

        # 추가
        st.markdown("---")
        st.markdown("**국가 추가**")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            add_country_name = st.text_input("국가명 (한글)", key="country_add_name")
        with c2:
            add_iso3 = st.text_input("ISO Alpha-3", key="country_add_iso3")
        with c3:
            add_iso2 = st.text_input("ISO Alpha-2", key="country_add_iso2")
        with c4:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("추가", key="country_add_btn") and add_country_name and add_iso3:
                country_map[add_country_name] = {"iso_alpha3": add_iso3.upper(), "iso_alpha2": add_iso2.upper()}
                save_json("country_mapping.json", country_map)
                st.success(f"{add_country_name} 추가 완료!")
                st.rerun()

        # 삭제
        if country_map:
            st.markdown("**국가 삭제**")
            c1, c2 = st.columns([3, 1])
            with c1:
                del_country = st.selectbox("삭제할 국가", list(country_map.keys()), key="country_del_select")
            with c2:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                if st.button("삭제", key="country_del_btn"):
                    del country_map[del_country]
                    save_json("country_mapping.json", country_map)
                    st.success(f"{del_country} 삭제 완료!")
                    st.rerun()


# ====================================================================
#  라우팅
# ====================================================================
pg = st.session_state["current_page"]
if pg == "forecast":
    page_forecast()
elif pg == "query":
    page_query()
elif pg == "price":
    page_price()
elif pg == "discover":
    page_discover()
elif pg == "settings":
    page_settings()
