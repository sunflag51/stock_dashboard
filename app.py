from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis_engine import AnalysisResult, analyze_stock
from data_provider import (
    MarketBundle,
    load_market_bundle,
    normalize_symbol,
)


st.set_page_config(
    page_title="株式分析ダッシュボード",
    page_icon="📈",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 12px;
    }

    .result-card {
        padding: 18px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .small-note {
        color: #777;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def cached_load_market_bundle(
    provider_name: str,
    symbol: str,
) -> MarketBundle:
    return load_market_bundle(
        provider_name=provider_name,
        symbol=symbol,
    )


def safe_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)

        if np.isfinite(converted):
            return converted
    except (TypeError, ValueError):
        pass

    return None


def format_number(
    value: Any,
    decimals: int = 2,
    suffix: str = "",
    missing: str = "－",
) -> str:
    number = safe_float(value)

    if number is None:
        return missing

    return f"{number:,.{decimals}f}{suffix}"


def format_large_number(value: Any) -> str:
    number = safe_float(value)

    if number is None:
        return "－"

    absolute = abs(number)

    if absolute >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:,.2f}兆"

    if absolute >= 1_000_000_000:
        return f"{number / 1_000_000_000:,.2f}十億"

    if absolute >= 1_000_000:
        return f"{number / 1_000_000:,.2f}百万"

    return f"{number:,.0f}"


def percent_from_ratio(value: Any) -> str:
    number = safe_float(value)

    if number is None:
        return "－"

    return f"{number * 100:,.2f}%"


def create_chart(result: AnalysisResult):
    frame = result.frame.tail(260).copy()

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.2, 1.1, 1.4],
            "hspace": 0.08,
        },
    )

    price_axis, volume_axis, macd_axis = axes

    price_axis.plot(
        frame.index,
        frame["Close"],
        color="#1565C0",
        linewidth=1.6,
        label="Close",
    )

    price_axis.plot(
        frame.index,
        frame["SMA20"],
        color="#FB8C00",
        linewidth=1.0,
        label="SMA20",
    )

    price_axis.plot(
        frame.index,
        frame["SMA50"],
        color="#8E24AA",
        linewidth=1.0,
        label="SMA50",
    )

    if frame["SMA200"].notna().any():
        price_axis.plot(
            frame.index,
            frame["SMA200"],
            color="#455A64",
            linewidth=1.0,
            label="SMA200",
        )

    if (
        frame["BollingerUpper"].notna().any()
        and frame["BollingerLower"].notna().any()
    ):
        upper = frame["BollingerUpper"].astype(float).to_numpy()
        lower = frame["BollingerLower"].astype(float).to_numpy()

        price_axis.fill_between(
            frame.index,
            lower,
            upper,
            color="#90CAF9",
            alpha=0.16,
            label="Bollinger Band",
        )

    if result.support is not None:
        price_axis.axhline(
            result.support,
            color="#2E7D32",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label="Support",
        )

    if result.resistance is not None:
        price_axis.axhline(
            result.resistance,
            color="#C62828",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label="Resistance",
        )

    price_axis.set_ylabel("Price")
    price_axis.grid(alpha=0.18)
    price_axis.legend(
        loc="upper left",
        ncol=3,
        fontsize=8,
    )

    colors = np.where(
        frame["Close"].diff().fillna(0) >= 0,
        "#26A69A",
        "#EF5350",
    )

    volume_axis.bar(
        frame.index,
        frame["Volume"],
        color=colors,
        width=1.0,
        alpha=0.75,
    )

    volume_axis.plot(
        frame.index,
        frame["VolumeSMA20"],
        color="#455A64",
        linewidth=1.0,
    )

    volume_axis.set_ylabel("Volume")
    volume_axis.grid(alpha=0.15)

    macd_axis.plot(
        frame.index,
        frame["MACD"],
        color="#1565C0",
        linewidth=1.1,
        label="MACD",
    )

    macd_axis.plot(
        frame.index,
        frame["MACDSignal"],
        color="#FB8C00",
        linewidth=1.0,
        label="Signal",
    )

    histogram_colors = np.where(
        frame["MACDHistogram"].fillna(0) >= 0,
        "#26A69A",
        "#EF5350",
    )

    macd_axis.bar(
        frame.index,
        frame["MACDHistogram"],
        color=histogram_colors,
        width=1.0,
        alpha=0.55,
    )

    macd_axis.axhline(
        0,
        color="#777777",
        linewidth=0.7,
    )

    macd_axis.set_ylabel("MACD")
    macd_axis.grid(alpha=0.15)
    macd_axis.legend(
        loc="upper left",
        fontsize=8,
    )

    figure.autofmt_xdate()
    return figure


def render_fundamentals(info: dict[str, Any]) -> None:
    rows = [
        ("企業名", info.get("longName") or info.get("shortName")),
        ("セクター", info.get("sector")),
        ("業種", info.get("industry")),
        ("時価総額", format_large_number(info.get("marketCap"))),
        ("PER", format_number(info.get("trailingPE"))),
        ("予想PER", format_number(info.get("forwardPE"))),
        ("PBR", format_number(info.get("priceToBook"))),
        ("ROE", percent_from_ratio(info.get("returnOnEquity"))),
        (
            "営業利益率",
            percent_from_ratio(info.get("operatingMargins")),
        ),
        (
            "売上高成長率",
            percent_from_ratio(info.get("revenueGrowth")),
        ),
        (
            "利益成長率",
            percent_from_ratio(info.get("earningsGrowth")),
        ),
        (
            "負債資本比率",
            format_number(info.get("debtToEquity")),
        ),
        (
            "配当利回り",
            percent_from_ratio(info.get("dividendYield")),
        ),
        (
            "ベータ",
            format_number(info.get("beta")),
        ),
        (
            "52週高値",
            format_number(info.get("fiftyTwoWeekHigh")),
        ),
        (
            "52週安値",
            format_number(info.get("fiftyTwoWeekLow")),
        ),
    ]

    table = pd.DataFrame(
        rows,
        columns=["項目", "値"],
    )

    table["値"] = table["値"].fillna("－")

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


def render_score(result: AnalysisResult) -> None:
    st.subheader("総合スコア")

    score_column, judgment_column = st.columns([1, 2])

    with score_column:
        st.metric(
            "スコア",
            f"{result.score:.1f} / 100",
        )

        st.progress(
            min(max(result.score / 100, 0.0), 1.0)
        )

    with judgment_column:
        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:0.9rem;color:#777;">
                    総合判定
                </div>
                <div style="font-size:1.8rem;font-weight:700;">
                    {result.verdict}
                </div>
                <div class="small-note">
                    テクニカル、出来高、相対強度、
                    ファンダメンタルズを点数化した参考評価です。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### スコア内訳")

    for component, score in result.components.items():
        maximum = result.component_maximums[component]
        ratio = score / maximum if maximum else 0

        label_column, progress_column, value_column = st.columns(
            [1.3, 4, 1]
        )

        label_column.write(component)

        progress_column.progress(
            min(max(ratio, 0.0), 1.0)
        )

        value_column.write(
            f"{score:.1f} / {maximum:.0f}"
        )


def main() -> None:
    st.title("📈 株式分析ダッシュボード")

    st.caption(
        "日足価格、テクニカル指標、出来高、"
        "市場・セクター相対強度、企業情報を統合して分析します。"
    )

    with st.sidebar:
        st.header("分析条件")

        provider_name = st.selectbox(
            "データ取得元",
            options=["yfinance"],
            index=0,
        )

        default_symbol = st.session_state.get(
            "last_symbol",
            "NVDA",
        )

        symbol_input = st.text_input(
            "銘柄コード",
            value=default_symbol,
            help=(
                "米国株は NVDA、日本株は 7203.T "
                "のように入力してください。"
            ),
        )

        analyze_button = st.button(
            "分析を実行",
            type="primary",
            use_container_width=True,
        )

        if st.button(
            "キャッシュを削除",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.success("キャッシュを削除しました。")

        st.divider()

        st.markdown(
            """
            **入力例**

            - 米国株：`NVDA`
            - 米国ETF：`SPY`
            - 日本株：`7203.T`
            - 日本指数：`^N225`
            """
        )

    if not analyze_button and "analysis_symbol" not in st.session_state:
        st.info(
            "左側のサイドバーで銘柄コードを入力し、"
            "「分析を実行」を押してください。"
        )
        return

    if analyze_button:
        try:
            normalized_symbol = normalize_symbol(symbol_input)
            st.session_state["analysis_symbol"] = normalized_symbol
            st.session_state["last_symbol"] = normalized_symbol
        except ValueError as exc:
            st.error(str(exc))
            return

    symbol = st.session_state.get(
        "analysis_symbol",
        normalize_symbol(symbol_input),
    )

    try:
        with st.spinner(
            f"{symbol}の価格・企業情報を取得しています..."
        ):
            bundle = cached_load_market_bundle(
                provider_name=provider_name,
                symbol=symbol,
            )
    except Exception as exc:
        st.error(
            "データ取得に失敗しました。銘柄コードと"
            "インターネット接続を確認してください。"
        )
        st.exception(exc)
        return

    if bundle.primary.prices.empty:
        st.error(
            f"{symbol}の有効な価格データを取得できませんでした。"
        )

        for warning in bundle.warnings:
            st.warning(warning)

        return

    try:
        result = analyze_stock(
            symbol=bundle.primary.symbol,
            prices=bundle.primary.prices,
            info=bundle.primary.info,
            market_prices=(
                bundle.market.prices
                if bundle.market is not None
                else None
            ),
            sector_prices=(
                bundle.sector.prices
                if bundle.sector is not None
                else None
            ),
        )
    except Exception as exc:
        st.error("分析処理に失敗しました。")
        st.exception(exc)
        return

    info = bundle.primary.info
    company_name = (
        info.get("longName")
        or info.get("shortName")
        or bundle.primary.symbol
    )

    st.subheader(f"{company_name}（{bundle.primary.symbol}）")

    latest = result.latest
    close = latest.get("close")
    change = latest.get("change")
    change_percent = latest.get("change_percent")

    delta_text = None

    if change is not None and change_percent is not None:
        delta_text = (
            f"{change:+,.2f} "
            f"({change_percent:+.2f}%)"
        )

    columns = st.columns(6)

    columns[0].metric(
        "終値",
        format_number(close),
        delta=delta_text,
    )

    columns[1].metric(
        "RSI（14）",
        format_number(latest.get("rsi14"), 1),
    )

    columns[2].metric(
        "MACD",
        format_number(latest.get("macd"), 3),
    )

    columns[3].metric(
        "ATR（14）",
        format_number(latest.get("atr14")),
    )

    columns[4].metric(
        "サポート参考値",
        format_number(result.support),
    )

    columns[5].metric(
        "レジスタンス参考値",
        format_number(result.resistance),
    )

    if bundle.warnings or result.warnings:
        with st.expander(
            "データ・分析上の注意事項",
            expanded=False,
        ):
            for warning in dict.fromkeys(
                bundle.warnings + result.warnings
            ):
                st.warning(warning)

    tab_summary, tab_chart, tab_signals, tab_fundamentals, tab_data = (
        st.tabs(
            [
                "総合評価",
                "チャート",
                "シグナル",
                "企業情報",
                "分析データ",
            ]
        )
    )

    with tab_summary:
        render_score(result)

        st.divider()
        st.markdown("#### 相対パフォーマンス")

        relative_columns = st.columns(3)

        relative_columns[0].metric(
            "銘柄の約3か月リターン",
            format_number(
                latest.get("primary_return_3m"),
                1,
                "%",
            ),
        )

        relative_columns[1].metric(
            f"市場比（{bundle.market_symbol or 'なし'}）",
            format_number(
                result.market_relative_return,
                1,
                "pt",
            ),
        )

        relative_columns[2].metric(
            f"セクター比（{bundle.sector_symbol or 'なし'}）",
            format_number(
                result.sector_relative_return,
                1,
                "pt",
            ),
        )

        st.caption(
            "相対値は銘柄リターンから市場または"
            "セクターのリターンを差し引いた参考値です。"
        )

    with tab_chart:
        st.markdown("#### 直近約1年の価格・出来高・MACD")

        figure = create_chart(result)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with tab_signals:
        st.markdown("#### テクニカル・評価シグナル")

        if result.signals:
            for signal in result.signals:
                st.write(f"・{signal}")
        else:
            st.info("表示できるシグナルがありません。")

        st.markdown("#### ローソク足パターン")

        if result.candlestick_patterns:
            for pattern in result.candlestick_patterns:
                st.write(f"・{pattern}")
        else:
            st.write(
                "直近足に主要なローソク足パターンは"
                "検出されませんでした。"
            )

        st.warning(
            "パターンは直近の日足のみを対象とした簡易判定です。"
            "単独で将来の値動きを確定するものではありません。"
        )

    with tab_fundamentals:
        st.markdown("#### 企業情報・主要指標")

        if info:
            render_fundamentals(info)
        else:
            st.info(
                "企業情報を取得できなかったため、"
                "表示できるファンダメンタルデータがありません。"
            )

    with tab_data:
        st.markdown("#### 最新の分析データ")

        display_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA20",
            "SMA50",
            "SMA200",
            "RSI14",
            "MACD",
            "MACDSignal",
            "MACDHistogram",
            "K",
            "D",
            "J",
            "ATR14",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in result.frame.columns
        ]

        st.dataframe(
            result.frame[available_columns]
            .tail(250)
            .sort_index(ascending=False),
            use_container_width=True,
        )

        csv_data = result.frame.to_csv(
            index=True
        ).encode("utf-8-sig")

        st.download_button(
            "分析データをCSVでダウンロード",
            data=csv_data,
            file_name=f"{result.symbol}_analysis.csv",
            mime="text/csv",
        )

    st.divider()

    st.caption(
        "本アプリのスコア、支持線・抵抗線、テクニカルシグナルは"
        "統計的・機械的な参考情報です。投資成果を保証するものではなく、"
        "売買判断は価格変動、決算、流動性、為替、ニュースなども含めて"
        "総合的に検討してください。"
    )


if __name__ == "__main__":
    main()
