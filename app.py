import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
from pathlib import Path

# ── .env 로드 (os.environ 직접) ─────────────────────
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
    {"key": "forecast", "label": "🗺️ 수요 예측"},
    {"key": "query",    "label": "🔍 쿼리 분석"},
    {"key": "price",    "label": "💰 가격 비교 분석"},
    {"key": "settings", "label": "⚙️ 설정"},
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
    df = load_fallback_data()
    country_map = load_json("country_mapping.json")

    # 배너
    min_d = df["주차시작일"].min().strftime("%Y.%m.%d")
    max_d = df["주차시작일"].max().strftime("%Y.%m.%d")
    n_countries = df["국가"].nunique()
    n_keywords = df["키워드"].nunique()
    api_mode = "실시간 API" if api_or_fallback() else "데모 모드 (Fallback CSV)"
    st.markdown(f"""
    <div class="top-banner">
        <div class="banner-icon">🗺️</div>
        <div class="banner-text">
            <h2>수요 예측</h2>
            <p>📅 {min_d} ~ {max_d} | 🏳️ {n_countries}개 국가 | 🔑 {n_keywords}개 키워드 | 📡 {api_mode}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 상단 인라인 필터 ──────────────────────────
    f1, f2, f3, f4 = st.columns([1, 2, 3, 1])
    with f1:
        연도_목록 = sorted(df["연도"].unique())
        선택_연도 = st.selectbox("연도", 연도_목록, index=len(연도_목록)-1, key="fc_year")
    with f2:
        국가_목록 = sorted(df["국가"].dropna().unique())
        선택_국가 = st.multiselect("국가", 국가_목록, default=국가_목록, key="fc_countries")
    with f3:
        월_범위 = st.slider("월 범위", 1, 12, (1, 12), key="fc_month_range")
    with f4:
        집계 = st.radio("집계", ["월간", "주간"], horizontal=True, key="fc_agg")

    if not 선택_국가:
        st.warning("국가를 1개 이상 선택해주세요.")
        return

    # 필터링
    filtered = df[
        (df["국가"].isin(선택_국가)) &
        (df["연도"] == 선택_연도) &
        (df["월"] >= 월_범위[0]) & (df["월"] <= 월_범위[1])
    ]
    if filtered.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    국가순위 = (
        filtered.groupby("국가")["쿼리수"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    color_map = get_country_color_map(국가순위)

    # ── 섹션 1: 세계지도 히트맵 ───────────────────
    st.markdown('<div class="section-header">🗺️ 세계 여행 수요 예측 지도</div>', unsafe_allow_html=True)

    # demand_score 계산: (출국자 0.6 + 검색트렌드 0.4) or 검색트렌드만
    country_query = filtered.groupby("국가")["쿼리수"].sum().reset_index()
    max_search = country_query["쿼리수"].max()
    country_query["normalized_search"] = (country_query["쿼리수"] / max_search) if max_search > 0 else 0

    # 관광통계 API 연결 시 출국자 수 반영
    if tourism_stats.is_available():
        try:
            dep_df = tourism_stats.fetch_departure_stats(선택_연도)
            if not dep_df.empty and "natKorNm" in dep_df.columns and "num" in dep_df.columns:
                dep_agg = dep_df.groupby("natKorNm")["num"].sum().reset_index()
                dep_agg.columns = ["국가", "출국자수"]
                country_query = country_query.merge(dep_agg, on="국가", how="left")
                country_query["출국자수"] = country_query["출국자수"].fillna(0)
                max_dep = country_query["출국자수"].max()
                country_query["normalized_departure"] = (country_query["출국자수"] / max_dep) if max_dep > 0 else 0
                country_query["demand_score"] = (
                    country_query["normalized_departure"] * 0.6 +
                    country_query["normalized_search"] * 0.4
                ) * 100
            else:
                country_query["demand_score"] = country_query["normalized_search"] * 100
        except Exception:
            country_query["demand_score"] = country_query["normalized_search"] * 100
    else:
        country_query["demand_score"] = country_query["normalized_search"] * 100

    # ISO 코드 매핑
    country_query["iso_alpha3"] = country_query["국가"].map(
        lambda x: country_map.get(x, {}).get("iso_alpha3", "")
    )
    map_data = country_query[country_query["iso_alpha3"] != ""]

    if not map_data.empty:
        fig_map = px.choropleth(
            map_data,
            locations="iso_alpha3",
            color="demand_score",
            hover_name="국가",
            hover_data={"쿼리수": ":,", "demand_score": ":.1f", "iso_alpha3": False},
            color_continuous_scale="YlOrRd",
            projection="natural earth",
            labels={"demand_score": "수요 점수", "쿼리수": "검색량"},
        )
        fig_map.update_layout(
            height=450, margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(showframe=False, showcoastlines=True, coastlinecolor="rgba(0,0,0,0.1)"),
            coloraxis_colorbar=dict(title="수요 점수", tickformat=".0f"),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # TOP 10 & 12개월 추이
    col_top, col_trend_map = st.columns([1, 2])
    with col_top:
        st.markdown("**🏆 TOP 10 핫 국가**")
        top10 = country_query.sort_values("demand_score", ascending=False).head(10)
        top10_display = top10[["국가", "쿼리수", "demand_score"]].copy()
        top10_display.columns = ["국가", "검색량", "수요점수"]
        top10_display["검색량"] = top10_display["검색량"].apply(lambda x: f"{int(x):,}")
        top10_display["수요점수"] = top10_display["수요점수"].apply(lambda x: f"{x:.1f}")
        top10_display.index = range(1, len(top10_display) + 1)
        st.dataframe(top10_display, use_container_width=True, height=380)

    with col_trend_map:
        st.markdown("**📊 선택국가 12개월 추이**")
        # 최근 12개월 데이터 (전체 연도에서)
        all_data = df[df["국가"].isin(선택_국가[:5])]  # 상위 5개국
        monthly_trend = all_data.groupby(["국가", "연월"])["쿼리수"].sum().reset_index()
        monthly_trend["연월_str"] = monthly_trend["연월"].astype(str)
        monthly_trend = monthly_trend.sort_values(["국가", "연월_str"]).tail(60)
        if not monthly_trend.empty:
            fig_trend12 = px.line(
                monthly_trend, x="연월_str", y="쿼리수", color="국가",
                markers=True, color_discrete_map=color_map,
                labels={"연월_str": "연월", "쿼리수": "검색량"},
            )
            fig_trend12.update_layout(
                height=380, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickangle=-45),
                yaxis=dict(gridcolor="rgba(0,0,0,0.08)", tickformat=","),
            )
            st.plotly_chart(fig_trend12, use_container_width=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 섹션 2: KPI 카드 ─────────────────────────
    st.markdown('<div class="section-header">📊 핵심 지표</div>', unsafe_allow_html=True)
    총쿼리 = int(filtered["쿼리수"].sum())
    최고국가 = filtered.groupby("국가")["쿼리수"].sum().idxmax()
    최고키워드 = filtered.groupby("키워드")["쿼리수"].sum().idxmax()
    monthly_kpi = filtered.groupby(["국가", "연월"])["쿼리수"].sum().reset_index().sort_values(["국가", "연월"])
    recent = monthly_kpi.groupby("국가").tail(2)
    country_delta = recent.groupby("국가")["쿼리수"].agg(
        lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 2 else None
    ).dropna()
    하락국가_이름 = country_delta.idxmin() if not country_delta.empty else "-"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card blue"><div class="kpi-icon">📊</div><div class="kpi-label">전체 쿼리 합계</div><div class="kpi-value">{총쿼리:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card green"><div class="kpi-icon">🏆</div><div class="kpi-label">최고 수요 국가</div><div class="kpi-value">{최고국가}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card orange"><div class="kpi-icon">🔑</div><div class="kpi-label">최다 검색 키워드</div><div class="kpi-value">{최고키워드}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card red"><div class="kpi-icon">⚠️</div><div class="kpi-label">하락 주의 국가</div><div class="kpi-value">{하락국가_이름}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 섹션 3: 수요 트렌드 ──────────────────────
    st.markdown('<div class="section-header">📈 수요 트렌드</div>', unsafe_allow_html=True)
    col_chart, col_rank = st.columns([3, 1])

    with col_chart:
        if 집계 == "월간":
            보기방식 = st.radio("보기 방식", ["누적", "추이"], horizontal=True, key="fc_trend_mode")
            if 보기방식 == "누적":
                trend = filtered.groupby(["국가", "월"])["쿼리수"].sum().reset_index()
                월_정렬 = [f"{m}월" for m in range(1, 13)]
                trend["월표시"] = pd.Categorical(
                    trend["월"].astype(str) + "월", categories=월_정렬, ordered=True
                )
                trend = trend.sort_values(["국가", "월표시"])
                fig = px.line(
                    trend, x="월표시", y="쿼리수", color="국가", markers=True,
                    category_orders={"국가": 국가순위, "월표시": 월_정렬},
                    color_discrete_map=color_map,
                    labels={"월표시": "월", "쿼리수": "쿼리 수"},
                )
            else:
                trend = filtered.groupby(["국가", "연월"])["쿼리수"].sum().reset_index()
                trend["연월_str"] = trend["연월"].astype(str)
                trend = trend.sort_values(["국가", "연월_str"])
                fig = px.line(
                    trend, x="연월_str", y="쿼리수", color="국가", markers=True,
                    category_orders={"국가": 국가순위},
                    color_discrete_map=color_map,
                    labels={"연월_str": "연월", "쿼리수": "쿼리 수"},
                )
                fig.update_xaxes(tickangle=-45)
        else:
            trend = filtered.groupby(["국가", "주차시작일"])["쿼리수"].sum().reset_index()
            trend = trend.sort_values(["국가", "주차시작일"])
            fig = px.line(
                trend, x="주차시작일", y="쿼리수", color="국가", markers=True,
                category_orders={"국가": 국가순위},
                color_discrete_map=color_map,
                labels={"주차시작일": "주차", "쿼리수": "쿼리 수"},
            )
            fig.update_xaxes(tickangle=-45)

        fig.update_layout(
            height=420, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.08)", tickformat=","),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_rank:
        st.markdown("**🏅 국가 순위**")
        rank = filtered.groupby("국가")["쿼리수"].sum().sort_values(ascending=False).reset_index()
        rank.columns = ["국가", "합계"]
        rank.index = range(1, len(rank) + 1)
        rank.index.name = "순위"
        rank["합계"] = rank["합계"].apply(lambda x: f"{int(x):,}")
        st.dataframe(rank, use_container_width=True, height=420)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 섹션 4: 월별 수요 강세 ────────────────────
    st.markdown('<div class="section-header">🔥 월별 수요 강세 분석</div>', unsafe_allow_html=True)
    heat_data = filtered.groupby(["국가", "월"])["쿼리수"].sum().reset_index()
    heat_data["월표시"] = heat_data["월"].astype(str) + "월"
    월_정렬 = [f"{m}월" for m in range(1, 13)]
    heat_data["월표시"] = pd.Categorical(heat_data["월표시"], categories=월_정렬, ordered=True)
    heat_data = heat_data.sort_values(["국가", "월표시"])

    col_heat, col_summary = st.columns([1, 1.35], gap="large")
    CHART_HEIGHT = 460

    with col_heat:
        st.markdown("**월별 x 국가 히트맵**")
        heat_pivot = (
            heat_data.pivot(index="국가", columns="월표시", values="쿼리수")
            .fillna(0).reindex(index=국가순위, columns=월_정렬)
        )
        text_display = []
        for row in heat_pivot.values:
            text_display.append([f"{int(v):,}" if v > 0 else "" for v in row])

        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_pivot.values, x=heat_pivot.columns.tolist(), y=heat_pivot.index.tolist(),
            text=text_display, texttemplate="%{text}", textfont={"size": 10},
            colorscale=[[0,"#f0f4ff"],[0.25,"#b8ccff"],[0.5,"#667eea"],[0.75,"#4a5acf"],[1,"#2d3a8c"]],
            hoverongaps=False,
            hovertemplate="국가: %{y}<br>월: %{x}<br>쿼리수: %{z:,}<extra></extra>",
            colorbar=dict(title="쿼리수", tickformat=","),
        ))
        fig_heat.update_layout(
            height=CHART_HEIGHT, margin=dict(l=40, r=10, t=10, b=45),
            xaxis=dict(tickmode="array", tickvals=월_정렬, ticktext=월_정렬, tickangle=0, side="bottom"),
            yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=국가순위),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_summary:
        st.markdown("**국가별 상위/하위 월**")
        summary_rows = []
        for country in 국가순위:
            c_month = heat_data[heat_data["국가"] == country].copy()
            top3 = c_month.sort_values("쿼리수", ascending=False).head(3)
            bottom3 = c_month.sort_values("쿼리수", ascending=True).head(3)
            top_text = " / ".join([f"{row['월표시']} ({int(row['쿼리수']):,})" for _, row in top3.iterrows()])
            bottom_text = " / ".join([f"{row['월표시']} ({int(row['쿼리수']):,})" for _, row in bottom3.iterrows()])
            summary_rows.append({"국가": country, "🔺 상위 3개월": top_text, "🔻 하위 3개월": bottom_text})
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True, height=CHART_HEIGHT)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── 섹션 5: 자동 인사이트 ─────────────────────
    st.markdown('<div class="section-header">💡 자동 인사이트</div>', unsafe_allow_html=True)

    # 멀티연도 필터 (인사이트용)
    all_years_data = df[df["국가"].isin(선택_국가)]
    nation_df = all_years_data.groupby(["국가", "연월"])["쿼리수"].sum().reset_index().sort_values(["국가", "연월"])
    kw_df = all_years_data.groupby(["국가", "키워드", "연월"])["쿼리수"].sum().reset_index().sort_values(["국가", "키워드", "연월"])

    def get_top_keywords_insight(국가, mode="trend", top_n=3):
        c_kw = kw_df[kw_df["국가"] == 국가]
        pivot = c_kw.pivot_table(index="키워드", columns="연월", values="쿼리수", aggfunc="sum").fillna(0)
        if pivot.shape[1] < 2:
            return "키워드 데이터 없음"
        rows = []
        for kw in pivot.index:
            vals = pivot.loc[kw].values
            if mode == "trend":
                slope = get_trend_slope(vals)
                before = vals[-min(4, len(vals))]
                after = vals[-1]
                diff = after - before
                rate = ((diff / before) * 100) if before > 0 else 0
                rows.append((kw, slope, diff, rate))
            else:
                if len(vals) < 2:
                    continue
                before, after = vals[-2], vals[-1]
                diff = after - before
                rate = ((diff / before) * 100) if before > 0 else 0
                rows.append((kw, abs(diff), diff, rate))
        if not rows:
            return "키워드 데이터 없음"
        rows_sorted = sorted(rows, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        tags = []
        for kw, _, diff, rate in rows_sorted:
            sign = "+" if diff >= 0 else ""
            tags.append(f"{kw} {sign}{rate:.0f}%({sign}{int(diff):,})")
        return " / ".join(tags) if tags else "키워드 데이터 없음"

    insights = []
    for country in 국가순위:
        c_vals = nation_df[nation_df["국가"] == country]["쿼리수"].values
        if len(c_vals) < 2:
            continue
        slope = get_trend_slope(c_vals)
        mean_val = c_vals.mean()
        slope_rate = (slope / mean_val) if mean_val > 0 else 0
        mom_rate = ((c_vals[-1] - c_vals[-2]) / c_vals[-2]) if c_vals[-2] > 0 else 0

        if mom_rate >= 0.2:
            kw_tag = get_top_keywords_insight(country, mode="mom")
            insights.append(("surge", "⚡ 급등", country, f"전월 대비 {mom_rate*100:.0f}% 급등", f"급등 키워드: {kw_tag}"))
        elif mom_rate <= -0.2:
            kw_tag = get_top_keywords_insight(country, mode="mom")
            insights.append(("drop", "📉 급락", country, f"전월 대비 {abs(mom_rate)*100:.0f}% 급락", f"급락 키워드: {kw_tag}"))
        elif slope_rate > 0.02:
            kw_tag = get_top_keywords_insight(country, mode="trend")
            insights.append(("up", "📈 상승세", country, f"최근 {len(c_vals)}개월 우상향 추세", f"주 상승 키워드: {kw_tag}"))
        elif slope_rate < -0.02:
            kw_tag = get_top_keywords_insight(country, mode="trend")
            insights.append(("down", "⚠️ 하락세", country, f"최근 {len(c_vals)}개월 우하향 추세", f"주 감소 키워드: {kw_tag}"))

    if insights:
        for group_type in ["surge", "drop", "up", "down"]:
            for css_class, badge, country, msg, kw_msg in [i for i in insights if i[0] == group_type]:
                st.markdown(f'<div class="insight-card {css_class}"><div class="insight-title">{badge} {country}</div><div class="insight-body">{msg}<br>↳ {kw_msg}</div></div>', unsafe_allow_html=True)
    else:
        st.write("현재 기간에서 특이 패턴이 감지되지 않았어요.")


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

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        분석유형 = st.selectbox("분석 유형", ["자사 분석", "국가별 분석", "경쟁사 분석"], key="q_type")
    with f2:
        기간 = st.date_input("기간", value=[df["주차시작일"].min(), df["주차시작일"].max()], key="q_period")
    with f3:
        집계_q = st.radio("집계", ["월간", "주간"], horizontal=True, key="q_agg")

    st.markdown("---")

    # ── 자사 분석 ──────────────────────────────────
    if 분석유형 == "자사 분석":
        own_keywords = company_info.get("자사", {}).get("brand_keywords", keywords_data.get("자사", []))
        if not own_keywords:
            st.info("설정에서 자사 브랜드 키워드를 등록해주세요.")
            return

        st.markdown('<div class="section-header">📊 자사 키워드 검색 트렌드</div>', unsafe_allow_html=True)

        if naver_datalab.is_available():
            start_str = str(기간[0]) if len(기간) == 2 else "2024-01-01"
            end_str = str(기간[1]) if len(기간) == 2 else "2026-05-01"
            time_unit = "month" if 집계_q == "월간" else "week"
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
            stats_df = naver_searchad.get_keyword_stats(own_keywords)
            if not stats_df.empty:
                display_cols = ["relKeyword", "monthlyPcQcCnt", "monthlyMobileQcCnt"]
                existing = [c for c in display_cols if c in stats_df.columns]
                if existing:
                    show_df = stats_df[existing].copy()
                    show_df.columns = ["키워드", "PC 검색량", "모바일 검색량"][:len(existing)]
                    if "PC 검색량" in show_df.columns and "모바일 검색량" in show_df.columns:
                        chart_df = show_df.melt(id_vars="키워드", var_name="구분", value_name="검색량")
                        fig_bar = px.bar(chart_df, x="키워드", y="검색량", color="구분", barmode="group")
                        fig_bar.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    st.dataframe(show_df, hide_index=True, use_container_width=True)
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
            if 집계_q == "월간":
                kw_trend = kw_filtered.groupby(["키워드", "연월"])["쿼리수"].sum().reset_index()
                kw_trend["연월_str"] = kw_trend["연월"].astype(str)
                kw_trend = kw_trend.sort_values(["키워드", "연월_str"])
                fig_kw = px.line(kw_trend, x="연월_str", y="쿼리수", color="키워드", markers=True,
                                 labels={"연월_str": "연월", "쿼리수": "쿼리 수"})
                fig_kw.update_xaxes(tickangle=-45)
            else:
                kw_trend = kw_filtered.groupby(["키워드", "주차시작일"])["쿼리수"].sum().reset_index()
                kw_trend = kw_trend.sort_values(["키워드", "주차시작일"])
                fig_kw = px.line(kw_trend, x="주차시작일", y="쿼리수", color="키워드", markers=True,
                                 labels={"주차시작일": "주차", "쿼리수": "쿼리 수"})
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
        comp_kw = []
        for comp in competitors:
            comp_kw.extend(comp.get("brand_keywords", []))

        all_kw = own_kw + comp_kw
        if not all_kw:
            st.info("설정에서 자사/경쟁사 키워드를 등록해주세요.")
            return

        st.markdown('<div class="section-header">📊 자사 vs 경쟁사 검색 트렌드 비교</div>', unsafe_allow_html=True)

        if naver_datalab.is_available():
            start_str = str(기간[0]) if len(기간) == 2 else "2024-01-01"
            end_str = str(기간[1]) if len(기간) == 2 else "2026-05-01"
            time_unit = "month" if 집계_q == "월간" else "week"
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
    items = naver_shopping.search_shopping(query, display=100, sort=sort_map.get(sort_opt, "sim"))
    if not items:
        st.warning("검색 결과가 없습니다.")
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
#  메뉴 4: 설정
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

    tab_company, tab_kw, tab_newkw, tab_api, tab_country = st.tabs([
        "🏢 자사/경쟁사 관리", "📝 키워드 관리", "🔎 신규 키워드 발굴",
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
            with st.expander(f"▼ {comp['name']}", expanded=False):
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

    # ── 탭2: 키워드 관리 ───────────────────────────
    with tab_kw:
        kw_data = load_json("trend_keywords.json")

        for category in ["자사", "경쟁사", "시즌"]:
            with st.expander(f"📂 {category} 키워드", expanded=False):
                keywords = kw_data.get(category, [])
                if isinstance(keywords, list):
                    st.markdown(" · ".join([f"`{k}`" for k in keywords]) if keywords else "(없음)")
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        new_kw = st.text_input(f"{category} 키워드 추가", key=f"kw_add_{category}")
                    with c2:
                        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                        if st.button("추가", key=f"kw_btn_{category}") and new_kw:
                            keywords.append(new_kw)
                            kw_data[category] = keywords
                            save_json("trend_keywords.json", kw_data)
                            st.rerun()
                    if keywords:
                        del_kw = st.selectbox(f"삭제할 키워드", keywords, key=f"kw_del_{category}")
                        if st.button("삭제", key=f"kw_rm_{category}"):
                            keywords.remove(del_kw)
                            kw_data[category] = keywords
                            save_json("trend_keywords.json", kw_data)
                            st.rerun()

        # 국가별 키워드
        country_kw = kw_data.get("국가별", {})
        for country, keywords in country_kw.items():
            with st.expander(f"🌍 {country} 키워드", expanded=False):
                st.markdown(" · ".join([f"`{k}`" for k in keywords]) if keywords else "(없음)")
                c1, c2 = st.columns([3, 1])
                with c1:
                    new_kw = st.text_input(f"{country} 키워드 추가", key=f"kw_add_country_{country}")
                with c2:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    if st.button("추가", key=f"kw_btn_country_{country}") and new_kw:
                        keywords.append(new_kw)
                        kw_data["국가별"][country] = keywords
                        save_json("trend_keywords.json", kw_data)
                        st.rerun()
                if keywords:
                    del_kw = st.selectbox("삭제할 키워드", keywords, key=f"kw_del_country_{country}")
                    if st.button("삭제", key=f"kw_rm_country_{country}"):
                        keywords.remove(del_kw)
                        kw_data["국가별"][country] = keywords
                        save_json("trend_keywords.json", kw_data)
                        st.rerun()

        # 국가 추가
        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        with c1:
            new_country = st.text_input("새 국가 카테고리 추가", key="kw_new_country")
        with c2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("추가", key="kw_btn_new_country") and new_country:
                if "국가별" not in kw_data:
                    kw_data["국가별"] = {}
                kw_data["국가별"][new_country] = []
                save_json("trend_keywords.json", kw_data)
                st.rerun()

    # ── 탭3: 신규 키워드 발굴 ──────────────────────
    with tab_newkw:
        st.markdown("### 🔎 신규 키워드 발굴")
        st.markdown("검색광고 API를 통해 연관 키워드를 자동 조회합니다.")

        seed_kw = st.text_input("시드 키워드", placeholder="예: 일본여행", key="newkw_seed")

        if st.button("🔍 연관 키워드 조회", key="newkw_search") and seed_kw:
            if naver_searchad.is_available():
                with st.spinner("연관 키워드 조회 중..."):
                    stats_df = naver_searchad.get_keyword_stats([seed_kw])
                if not stats_df.empty:
                    display_cols = []
                    col_mapping = {
                        "relKeyword": "키워드",
                        "monthlyPcQcCnt": "PC 검색량",
                        "monthlyMobileQcCnt": "모바일 검색량",
                        "compIdx": "경쟁강도",
                    }
                    for col, label in col_mapping.items():
                        if col in stats_df.columns:
                            display_cols.append(col)

                    show_df = stats_df[display_cols].copy()
                    show_df = show_df.rename(columns=col_mapping)
                    if "PC 검색량" in show_df.columns and "모바일 검색량" in show_df.columns:
                        show_df["총 검색량"] = show_df["PC 검색량"] + show_df["모바일 검색량"]
                        show_df = show_df.sort_values("총 검색량", ascending=False)

                    st.dataframe(show_df, hide_index=True, use_container_width=True, height=500)

                    # 원클릭 추가
                    st.markdown("---")
                    kw_data = load_json("trend_keywords.json")
                    categories = ["자사", "경쟁사", "시즌"]
                    country_kw = kw_data.get("국가별", {})
                    categories.extend([f"국가별/{c}" for c in country_kw.keys()])

                    c1, c2, c3 = st.columns([2, 2, 1])
                    with c1:
                        if "키워드" in show_df.columns:
                            add_kws = st.multiselect("추가할 키워드", show_df["키워드"].tolist(), key="newkw_add_list")
                        else:
                            add_kws = []
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

    # ── 탭4: API 키 관리 ───────────────────────────
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
elif pg == "settings":
    page_settings()
