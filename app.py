from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis_engine import (
    add_indicators,
    calculate_score,
    detect_candlestick_patterns,
    detect_market_structure,
    safe_number,
)
from data_provider import load_market_bundle


# =========================================================
# Streamlit基本設定
# =========================================================
st.set_page_config(
    page_title="株価分析ダッシュボード",
    page_icon="📈",
    layout="wide",
)


SHEET_LINK = (
    "https://docs.google.com/spreadsheets/d/"
    "1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/"
    "edit?usp=drivesdk"
)

BASE_OPTIONS = [
    "NVDA (エヌビディア)",
    "GOOG (アルファベット)",
    "KO (コカ・コーラ)",
    "V (ビザ)",
    "AAPL (アップル)",
    "ISRG (インテュイティブ・サージカル)",
    "COST (コストコ)",
    "MSFT (マイクロソフト)",
]


# =========================================================
# キャッシュ
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_bundle(provider_name: str, symbol: str):
    return load_market_bundle(
        provider_name=provider_name,
        symbol=symbol,
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_options(sheet_link: str) -> tuple[list[str], str | None]:
    if not sheet_link.startswith("http"):
        return [], None

    try:
        csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
        sheet_df = pd.read_csv(csv_url, header=None)

        options: list[str] = []

        for _, row in sheet_df.iterrows():
            if len(row) < 2:
                continue

            company = str(row.iloc[0]).strip()
            code = str(row.iloc[1]).strip().upper()

            invalid_names = {
                "",
                "企業名",
                "名前",
                "会社名",
                "NAN",
                "NONE",
            }

            if company.upper() in invalid_names:
                continue

            if code.upper() in {"", "NAN", "NONE", "銘柄コード"}:
                continue

            options.append(f"{code} ({company})")

        return options, None

    except Exception as exc:
        return [], str(exc)


# =========================================================
# 表示用関数
# =========================================================
def format_value(
    value,
    digits: int = 2,
    suffix: str = "",
    unavailable: str = "取得不可",
) -> str:
    number = safe_number(value)

    if number is None:
        return unavailable

    return f"{number:,.{digits}f}{suffix}"


def format_large_number(value) -> str:
    number = safe_number(value)

    if number is None:
        return "取得不可"

    abs_number = abs(number)

    if abs_number >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:.2f}兆"
    if abs_number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}十億"
    if abs_number >= 1_000_000:
        return f"{number / 1_000_000:.2f}百万"

    return f"{number:,.0f}"


def get_currency_symbol(
    info: dict,
    metadata: dict,
    ticker: str,
) -> tuple[str, str]:
    currency = (
        info.get("currency")
        or metadata.get("currency")
        or ""
    )

    if not currency:
        currency = "JPY" if ticker.endswith(".T") else "USD"

    currency_symbols = {
        "USD": "$",
        "JPY": "¥",
        "EUR": "€",
        "GBP": "£",
        "CAD": "C$",
        "AUD": "A$",
        "HKD": "HK$",
    }

    return currency_symbols.get(currency, f"{currency} "), currency


def format_dividend_yield(value) -> str:
    dividend_yield = safe_number(value)

    if dividend_yield is None:
        return "取得不可"

    # Yahooでは通常、小数形式で返る
    if abs(dividend_yield) <= 1:
        dividend_yield *= 100

    return f"{dividend_yield:.2f}%"


def prepare_chart_data(
    df: pd.DataFrame,
    display_period: str,
) -> pd.DataFrame:
    period_rows = {
        "3ヶ月": 65,
        "6ヶ月": 130,
        "1年": 260,
        "5年": 1300,
    }

    rows = period_rows.get(display_period, 130)
    return df.tail(rows).copy()


def draw_chart(
    df: pd.DataFrame,
    ticker: str,
    chart_mode: str,
    show_ichimoku: bool,
):
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 16),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1, 1, 1, 1],
        },
    )

    ax_price, ax_volume, ax_macd, ax_rsi, ax_kdj = axes

    if chart_mode == "ローソク足":
        up = df["Close"] >= df["Open"]
        down = df["Close"] < df["Open"]

        ax_price.vlines(
            df.index,
            df["Low"],
            df["High"],
            color="#444444",
            linewidth=0.8,
        )

        ax_price.bar(
            df.index[up],
            df.loc[up, "Close"] - df.loc[up, "Open"],
            bottom=df.loc[up, "Open"],
            color="#ef5350",
            edgecolor="#ef5350",
            width=0.65,
            label="Up",
        )

        ax_price.bar(
            df.index[down],
            df.loc[down, "Open"] - df.loc[down, "Close"],
            bottom=df.loc[down, "Close"],
            color="#26a69a",
            edgecolor="#26a69a",
            width=0.65,
            label="Down",
        )
    else:
        ax_price.plot(
            df.index,
            df["Close"],
            color="black",
            linewidth=1.8,
            label="Close",
        )

    ax_price.plot(
        df.index,
        df["MA5"],
        color="#ff9800",
        linewidth=1,
        label="MA5",
    )
    ax_price.plot(
        df.index,
        df["MA20"],
        color="#1565c0",
        linewidth=1.2,
        label="MA20",
    )
    ax_price.plot(
        df.index,
        df["MA50"],
        color="#7b1fa2",
        linewidth=1,
        label="MA50",
    )
    ax_price.plot(
        df.index,
        df["Upper"],
        color="#2e7d32",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label="BOLL +2σ",
    )
    ax_price.plot(
        df.index,
        df["Lower"],
        color="#2e7d32",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label="BOLL -2σ",
    )

    if show_ichimoku:
        ax_price.plot(
            df.index,
            df["Tenkan"],
            color="darkorange",
            linewidth=1,
            label="Tenkan",
        )
        ax_price.plot(
            df.index,
            df["Kijun"],
            color="mediumblue",
            linewidth=1,
            label="Kijun",
        )

        senkou_a = df["SenkouA"].to_numpy(dtype=float)
        senkou_b = df["SenkouB"].to_numpy(dtype=float)

        valid = np.isfinite(senkou_a) & np.isfinite(senkou_b)

        ax_price.fill_between(
            df.index,
            senkou_a,
            senkou_b,
            where=valid & (senkou_a >= senkou_b),
            color="lightcoral",
            alpha=0.25,
        )
        ax_price.fill_between(
            df.index,
            senkou_a,
            senkou_b,
            where=valid & (senkou_a < senkou_b),
            color="lightgreen",
            alpha=0.25,
        )

    structure = detect_market_structure(df)

    if structure["available"]:
        neckline = structure["neckline"]

        ax_price.axhline(
            neckline,
            color="#d32f2f",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Neckline",
        )

    ax_price.set_title(
        f"{ticker} - Technical Dashboard",
        fontsize=12,
    )
    ax_price.legend(
        loc="upper left",
        fontsize="small",
        ncol=3,
    )
    ax_price.grid(True, alpha=0.25)

    # 出来高
    volume_colors = np.where(
        df["Close"] >= df["Open"],
        "#ef5350",
        "#26a69a",
    )

    ax_volume.bar(
        df.index,
        df["Volume"],
        color=volume_colors,
        alpha=0.65,
    )
    ax_volume.plot(
        df.index,
        df["Vol_MA20"],
        color="navy",
        linewidth=1,
        label="Volume MA20",
    )
    ax_volume.set_ylabel("Volume")
    ax_volume.legend(loc="upper left", fontsize="small")
    ax_volume.grid(True, alpha=0.25)

    # MACD
    histogram_colors = np.where(
        df["MACD_Hist"] >= 0,
        "#ef5350",
        "#26a69a",
    )

    ax_macd.plot(
        df.index,
        df["MACD"],
        label="MACD",
        color="blue",
    )
    ax_macd.plot(
        df.index,
        df["Signal"],
        label="Signal",
        color="orange",
    )
    ax_macd.bar(
        df.index,
        df["MACD_Hist"],
        color=histogram_colors,
        alpha=0.4,
    )
    ax_macd.axhline(0, color="gray", linewidth=0.7)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left", fontsize="small")
    ax_macd.grid(True, alpha=0.25)

    # RSI
    ax_rsi.plot(
        df.index,
        df["RSI_9"],
        label="RSI(9)",
        color="magenta",
    )
    ax_rsi.plot(
        df.index,
        df["RSI_14"],
        label="RSI(14)",
        color="deepskyblue",
    )
    ax_rsi.axhline(70, color="red", linestyle=":", alpha=0.7)
    ax_rsi.axhline(30, color="blue", linestyle=":", alpha=0.7)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend(loc="upper left", fontsize="small")
    ax_rsi.grid(True, alpha=0.25)

    # KDJ
    ax_kdj.plot(df.index, df["K"], label="K", color="blue")
    ax_kdj.plot(df.index, df["D"], label="D", color="orange")
    ax_kdj.plot(df.index, df["J"], label="J", color="green")
    ax_kdj.axhline(80, color="red", linestyle=":", alpha=0.5)
    ax_kdj.axhline(20, color="blue", linestyle=":", alpha=0.5)
    ax_kdj.set_ylabel("KDJ")
    ax_kdj.legend(loc="upper left", fontsize="small")
    ax_kdj.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    plt.tight_layout()

    return fig


def category_metric(
    container,
    category_name: str,
    category_data: dict,
):
    normalized = category_data["normalized"]
    earned = category_data["earned"]
    available = category_data["available"]

    if normalized is None:
        container.metric(category_name, "判定不能")
    else:
        container.metric(
            category_name,
            f"{normalized}%",
            help=f"取得できた項目での得点: {earned}/{available}",
        )


# =========================================================
# セッション状態
# =========================================================
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""

if "last_company" not in st.session_state:
    st.session_state.last_company = ""


# =========================================================
# ヘッダー
# =========================================================
st.title("📈 株価テクニカル＆ファンダメンタル分析")
st.caption(
    "企業品質・市場環境・売られ過ぎ・反転確認を分離して判定します。"
    "売買を保証するものではなく、取得データに基づく参考情報です。"
)


# =========================================================
# 銘柄選択
# =========================================================
sheet_options, sheet_error = load_sheet_options(SHEET_LINK)

if sheet_error:
    st.warning(
        "Googleスプレッドシートの銘柄一覧を取得できなかったため、"
        "標準リストを使用します。"
    )

all_options = list(
    dict.fromkeys(BASE_OPTIONS + sheet_options)
)
all_options.append("その他（手入力）")

col1, col2, col3, col4 = st.columns(4)

with col1:
    ticker_choice = st.selectbox(
        "銘柄選択",
        all_options,
    )

    company_name = ""

    if ticker_choice == "その他（手入力）":
        ticker_symbol = st.text_input(
            "銘柄コード",
            value="7203.T",
            help="例：NVDA、AAPL、7203.T",
        ).strip().upper()
    else:
        ticker_symbol = ticker_choice.split(" ")[0].upper()

        if "(" in ticker_choice and ")" in ticker_choice:
            company_name = (
                ticker_choice.split("(", 1)[1]
                .rsplit(")", 1)[0]
                .strip()
            )

with col2:
    display_period = st.selectbox(
        "グラフ表示期間",
        ["3ヶ月", "6ヶ月", "1年", "5年"],
        index=1,
    )

with col3:
    chart_mode = st.radio(
        "表示形式",
        ["ローソク足", "ラインチャート"],
        horizontal=False,
    )

with col4:
    ichimoku_mode = st.radio(
        "一目均衡表",
        ["表示しない", "表示する"],
        horizontal=False,
    )

button_col1, button_col2 = st.columns([3, 1])

with button_col1:
    run_button = st.button(
        "分析を実行する",
        type="primary",
        use_container_width=True,
    )

with button_col2:
    if st.button(
        "キャッシュ更新",
        use_container_width=True,
        help="データ取得キャッシュを削除します。",
    ):
        st.cache_data.clear()
        st.success("キャッシュを削除しました。")


if run_button:
    if not ticker_symbol:
        st.warning("銘柄コードを入力してください。")
    else:
        st.session_state.is_analyzed = True
        st.session_state.last_ticker = ticker_symbol
        st.session_state.last_company = company_name


# =========================================================
# 分析画面
# =========================================================
if st.session_state.is_analyzed:
    ticker_to_analyze = st.session_state.last_ticker
    company_to_analyze = st.session_state.last_company

    try:
        with st.spinner(
            f"{ticker_to_analyze}のデータを取得・分析しています..."
        ):
            bundle = get_cached_bundle(
                provider_name="yfinance",
                symbol=ticker_to_analyze,
            )

    except Exception as exc:
        st.error(f"データ取得に失敗しました: {exc}")
        st.stop()

    primary = bundle.primary
    raw_df = primary.prices.copy()
    info = primary.info or {}

    if raw_df.empty or len(raw_df) < 80:
        st.error(
            f"銘柄「{ticker_to_analyze}」の有効な日足が不足しています。"
            "銘柄コードまたは上場市場を確認してください。"
        )
        st.stop()

    df = add_indicators(raw_df)
    df = detect_candlestick_patterns(df)

    market_df = None
    if bundle.market is not None and not bundle.market.prices.empty:
        market_df = bundle.market.prices

    sector_df = None
    if bundle.sector is not None and not bundle.sector.prices.empty:
        sector_df = bundle.sector.prices

    score_result = calculate_score(
        df=df,
        info=info,
        market_df=market_df,
        sector_df=sector_df,
        market_label=bundle.market_symbol or "市場",
        sector_label=bundle.sector_symbol or "セクター",
    )

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    currency_symbol, currency_code = get_currency_symbol(
        info,
        primary.metadata,
        ticker_to_analyze,
    )

    company_display = (
        company_to_analyze
        or info.get("longName")
        or info.get("shortName")
        or ticker_to_analyze
    )

    st.markdown("---")
    st.subheader(
        f"🏢 {company_display}【{ticker_to_analyze}】"
    )

    # データ状態
    data_col1, data_col2, data_col3, data_col4 = st.columns(4)

    last_date = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
    timezone_name = (
        primary.metadata.get("exchangeTimezoneName")
        or primary.metadata.get("timezone")
        or "不明"
    )

    data_col1.metric("データソース", primary.source)
    data_col2.metric("最終確定日足", last_date)
    data_col3.metric("市場タイムゾーン", timezone_name)
    data_col4.metric(
        "データ完全性",
        f"{score_result.completeness}%",
    )

    if primary.dropped_incomplete_bar:
        st.info(
            "当日の日足が未確定だったため、正式な分析対象から除外しました。"
            "グラフとスコアは直近の確定日足までを使用しています。"
        )

    if bundle.warnings:
        with st.expander("データ取得に関する注意事項"):
            for warning in dict.fromkeys(bundle.warnings):
                st.write(f"・{warning}")

    if score_result.completeness < 80:
        st.warning(
            "一部データが取得できていません。"
            "総合スコアは取得できた項目だけで正規化されています。"
        )

    # 基本情報
    price_change = latest["Close"] - previous["Close"]
    price_change_percent = (
        price_change / previous["Close"] * 100
        if previous["Close"] != 0
        else 0
    )

    pe = safe_number(info.get("trailingPE"))
    pbr = safe_number(info.get("priceToBook"))
    dividend_yield = (
        info.get("dividendYield")
        if info.get("dividendYield") is not None
        else info.get("trailingAnnualDividendYield")
    )

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric(
        f"株価（{currency_code}）",
        f"{currency_symbol}{latest['Close']:,.2f}",
        delta=f"{price_change_percent:+.2f}%",
    )
    info_col2.metric(
        "PER",
        format_value(pe, 1, "倍"),
    )
    info_col3.metric(
        "配当利回り",
        format_dividend_yield(dividend_yield),
    )
    info_col4.metric(
        "PBR",
        format_value(pbr, 2, "倍"),
    )

    with st.expander("企業情報をさらに表示"):
        detail_col1, detail_col2, detail_col3 = st.columns(3)

        detail_col1.write(f"**業種:** {info.get('industry', '取得不可')}")
        detail_col1.write(f"**セクター:** {info.get('sector', '取得不可')}")
        detail_col2.write(f"**時価総額:** {format_large_number(info.get('marketCap'))}")

        # 💡 ここが先ほどのエラー原因（文法エラーを修正しました）
        rev_growth_raw = safe_number(info.get('revenueGrowth'))
        rev_growth_val = rev_growth_raw * 100 if rev_growth_raw is not None else None
        detail_col2.write(f"**売上成長率:** {format_value(rev_growth_val, 1, '%')}")

        op_margin_raw = safe_number(info.get('operatingMargins'))
        op_margin_val = op_margin_raw * 100 if op_margin_raw is not None else None
        detail_col3.write(f"**営業利益率:** {format_value(op_margin_val, 1, '%')}")

        f_pe = safe_number(info.get('forwardPE'))
        detail_col3.write(f"**予想PER:** {format_value(f_pe, 1, '倍')}")

    st.markdown("---")

    # グラフ
    st.subheader("📊 テクニカルチャート")

    chart_df = prepare_chart_data(
        df,
        display_period,
    )

    fig = draw_chart(
        chart_df,
        ticker=ticker_to_analyze,
        chart_mode=chart_mode,
        show_ichimoku=(ichimoku_mode == "表示する"),
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 現在のテクニカル値
    technical_col1, technical_col2, technical_col3, technical_col4 = (
        st.columns(4)
    )

    technical_col1.metric(
        "RSI(14)",
        format_value(latest["RSI_14"], 1),
    )
    technical_col2.metric(
        "MACD",
        format_value(latest["MACD"], 3),
    )
    technical_col3.metric(
        "出来高 / 20日平均",
        (
            f"{latest['Volume'] / latest['Vol_MA20']:.2f}倍"
            if latest["Vol_MA20"] > 0
            else "計算不可"
        ),
    )
    technical_col4.metric(
        "ATR(14)",
        format_value(latest["ATR14"], 2),
    )

    # スコア
    st.markdown("---")
    st.subheader("🎯 総合判定")

    overall_col1, overall_col2 = st.columns([1, 2])

    with overall_col1:
        score_text = (
            f"{score_result.normalized_score}点"
            if score_result.normalized_score is not None
            else "判定不能"
        )

        st.markdown(
            f"""
            <div style="
                text-align:center;
                border:1px solid #dddddd;
                border-radius:12px;
                padding:22px;
            ">
                <div style="font-size:16px;">参考総合スコア</div>
                <div style="
                    font-size:42px;
                    font-weight:bold;
                    color:{score_result.status_color};
                ">
                    {score_text}
                </div>
                <div style="font-size:13px;">
                    取得項目 {score_result.earned}/{score_result.available}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with overall_col2:
        safe_status = html.escape(score_result.status)

        st.markdown(
            f"""
            <div style="
                text-align:center;
                color:{score_result.status_color};
                font-size:24px;
                font-weight:bold;
                margin-top:25px;
            ">
                {safe_status}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "総合スコアは売買指示ではありません。"
            "企業品質・市場環境・売られ過ぎ・反転確認の"
            "取得可能な項目を100点換算した参考値です。"
        )

    category_col1, category_col2, category_col3, category_col4 = (
        st.columns(4)
    )

    category_metric(
        category_col1,
        "企業品質",
        score_result.category_scores["企業品質"],
    )
    category_metric(
        category_col2,
        "市場環境",
        score_result.category_scores["市場環境"],
    )
    category_metric(
        category_col3,
        "売られ過ぎ",
        score_result.category_scores["売られ過ぎ"],
    )
    category_metric(
        category_col4,
        "反転確認",
        score_result.category_scores["反転確認"],
    )

    # リスク自己点検
    st.markdown("---")
    st.subheader("📝 リスク・自己点検")

    risk_score = st.slider(
        "資金管理・リスクリワードの自己評価",
        min_value=0,
        max_value=5,
        value=2,
        help=(
            "この値は総合分析スコアには加算しません。"
            "損切り位置、投資額、セクター偏重を自己確認するための項目です。"
        ),
    )

    atr = safe_number(latest["ATR14"])

    if atr is not None:
        stop_reference = latest["Close"] - atr * 2
        risk_per_share = latest["Close"] - stop_reference
        target_2r = latest["Close"] + risk_per_share * 2

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        risk_col1.metric(
            "参考2ATRストップ",
            f"{currency_symbol}{stop_reference:,.2f}",
        )
        risk_col2.metric(
            "1株あたり参考リスク",
            f"{currency_symbol}{risk_per_share:,.2f}",
        )
        risk_col3.metric(
            "参考2R到達価格",
            f"{currency_symbol}{target_2r:,.2f}",
        )

        st.caption(
            "ATRによる価格は機械的な参考値です。"
            "窓開け、決算、流動性、為替、手数料は考慮していません。"
        )

    st.write(f"自己点検結果：**{risk_score} / 5点**")

    # 詳細採点
    with st.expander(
        "詳細な自動判定の内訳を見る",
        expanded=False,
    ):
        for category in [
            "企業品質",
            "市場環境",
            "売られ過ぎ",
            "反転確認",
        ]:
            category_data = score_result.category_scores[category]

            st.markdown(
                f"### {category} "
                f"（{category_data['earned']}/"
                f"{category_data['available']}点）"
            )

            category_items = [
                item
                for item in score_result.items
                if item.category == category
            ]

            rows = []

            for item in category_items:
                if not item.available:
                    result_text = "判定不能"
                elif item.passed:
                    result_text = "✅ 成立"
                else:
                    result_text = "－ 不成立"

                rows.append(
                    {
                        "判定項目": item.label,
                        "結果": result_text,
                        "得点": (
                            f"{item.earned}/{item.points}"
                            if item.available
                            else f"-/{item.points}"
                        ),
                        "詳細": item.detail,
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # 反転構造
    with st.expander(
        "安値切り上げ・ネックライン判定を見る",
        expanded=False,
    ):
        structure = detect_market_structure(df)

        if not structure["available"]:
            st.info(
                "直近120営業日以内に、比較可能な確定スイング安値が"
                "2つ見つからなかったため判定できません。"
            )
        else:
            structure_col1, structure_col2, structure_col3 = st.columns(3)

            structure_col1.metric(
                "第1スイング安値",
                f"{currency_symbol}{structure['first_low']:,.2f}",
            )
            structure_col2.metric(
                "第2スイング安値",
                f"{currency_symbol}{structure['second_low']:,.2f}",
            )
            structure_col3.metric(
                "ネックライン",
                f"{currency_symbol}{structure['neckline']:,.2f}",
            )

            st.write(
                "安値切り上げ：",
                "✅ 確認" if structure["recent_higher_low"] else "未確認",
            )
            st.write(
                "ネックライン突破：",
                "✅ 確認" if structure["neckline_breakout"] else "未確認",
            )

            st.caption(
                "スイング安値は左右2日間の価格を確認してから確定します。"
                "そのため、直近2本の日足はスイング安値として扱いません。"
            )

    st.markdown("---")
    st.caption(
        "免責事項：本アプリは情報提供・学習目的です。"
        "投資判断は利用者自身の責任で行ってください。"
        "yfinanceのデータには遅延、欠損、修正が発生する場合があります。"
    )
