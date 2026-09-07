from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class AnalysisResult:
    symbol: str
    frame: pd.DataFrame
    score: float
    verdict: str
    components: dict[str, float] = field(default_factory=dict)
    component_maximums: dict[str, float] = field(default_factory=dict)
    latest: dict[str, Any] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    candlestick_patterns: list[str] = field(default_factory=list)
    support: Optional[float] = None
    resistance: Optional[float] = None
    market_relative_return: Optional[float] = None
    sector_relative_return: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


def _safe_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)

        if np.isfinite(converted):
            return converted
    except (TypeError, ValueError):
        pass

    return None


def _latest_value(
    series: pd.Series,
    default: Optional[float] = None,
) -> Optional[float]:
    if series is None or series.empty:
        return default

    value = _safe_float(series.iloc[-1])
    return default if value is None else value


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # 下落がなく平均損失がゼロの場合
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )

    # 値動きがない場合
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50,
    )

    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def calculate_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()

    df = prices.copy()

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "価格データに必要な列がありません: "
            + ", ".join(missing)
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    ).copy()

    df["Volume"] = df["Volume"].fillna(0)
    df = df.sort_index()

    df["Return"] = df["Close"].pct_change()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["EMA12"] = df["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    df["EMA26"] = df["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]

    df["MACDSignal"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["MACDHistogram"] = (
        df["MACD"] - df["MACDSignal"]
    )

    df["RSI14"] = _rsi(df["Close"], 14)
    df["ATR14"] = _atr(df, 14)

    lowest_low = df["Low"].rolling(9).min()
    highest_high = df["High"].rolling(9).max()

    price_range = (
        highest_high - lowest_low
    ).replace(0, np.nan)

    df["K"] = (
        100 * (df["Close"] - lowest_low) / price_range
    ).ewm(
        alpha=1 / 3,
        adjust=False,
    ).mean()

    df["D"] = df["K"].ewm(
        alpha=1 / 3,
        adjust=False,
    ).mean()

    df["J"] = 3 * df["K"] - 2 * df["D"]

    df["VolumeSMA20"] = df["Volume"].rolling(20).mean()
    df["OBV"] = _obv(df)
    df["OBVSMA10"] = df["OBV"].rolling(10).mean()

    rolling_std = df["Close"].rolling(20).std()

    df["BollingerUpper"] = df["SMA20"] + 2 * rolling_std
    df["BollingerLower"] = df["SMA20"] - 2 * rolling_std

    df["High20"] = df["High"].rolling(20).max()
    df["Low20"] = df["Low"].rolling(20).min()

    return df


def _period_return(
    prices: Optional[pd.DataFrame],
    periods: int = 63,
) -> Optional[float]:
    if prices is None or prices.empty:
        return None

    if "Close" not in prices.columns:
        return None

    close = pd.to_numeric(
        prices["Close"],
        errors="coerce",
    ).dropna()

    if len(close) < 2:
        return None

    start_position = max(0, len(close) - periods - 1)
    start_value = _safe_float(close.iloc[start_position])
    end_value = _safe_float(close.iloc[-1])

    if (
        start_value is None
        or end_value is None
        or start_value == 0
    ):
        return None

    return (end_value / start_value - 1) * 100


def _detect_candlestick_patterns(
    df: pd.DataFrame,
) -> list[str]:
    patterns: list[str] = []

    if len(df) < 2:
        return patterns

    current = df.iloc[-1]
    previous = df.iloc[-2]

    open_price = float(current["Open"])
    close_price = float(current["Close"])
    high_price = float(current["High"])
    low_price = float(current["Low"])

    body = abs(close_price - open_price)
    total_range = max(high_price - low_price, 1e-12)

    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price

    if body / total_range <= 0.1:
        patterns.append("十字線（方向感が拮抗）")

    if (
        lower_shadow >= body * 2
        and upper_shadow <= max(body, total_range * 0.1)
        and close_price >= open_price
    ):
        patterns.append("ハンマー型（下値否定の可能性）")

    if (
        upper_shadow >= body * 2
        and lower_shadow <= max(body, total_range * 0.1)
    ):
        patterns.append("上ヒゲ優勢（上値抵抗の可能性）")

    previous_open = float(previous["Open"])
    previous_close = float(previous["Close"])

    if (
        previous_close < previous_open
        and close_price > open_price
        and open_price <= previous_close
        and close_price >= previous_open
    ):
        patterns.append("強気包み足")

    if (
        previous_close > previous_open
        and close_price < open_price
        and open_price >= previous_close
        and close_price <= previous_open
    ):
        patterns.append("弱気包み足")

    return patterns


def _fundamental_score(
    info: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    最大20点。欠損項目は中立の2点とします。
    """
    score = 0.0
    notes: list[str] = []

    trailing_pe = _safe_float(info.get("trailingPE"))

    if trailing_pe is None:
        score += 2
    elif 0 < trailing_pe <= 25:
        score += 4
        notes.append("PERは評価上の許容範囲")
    elif trailing_pe <= 40:
        score += 2
    elif trailing_pe > 0:
        score += 1

    roe = _safe_float(info.get("returnOnEquity"))

    if roe is None:
        score += 2
    elif roe >= 0.15:
        score += 4
        notes.append("ROEが15%以上")
    elif roe >= 0.08:
        score += 3
    elif roe > 0:
        score += 1

    operating_margin = _safe_float(
        info.get("operatingMargins")
    )

    if operating_margin is None:
        score += 2
    elif operating_margin >= 0.15:
        score += 4
        notes.append("営業利益率が15%以上")
    elif operating_margin >= 0.08:
        score += 3
    elif operating_margin > 0:
        score += 1

    debt_to_equity = _safe_float(
        info.get("debtToEquity")
    )

    if debt_to_equity is None:
        score += 2
    elif debt_to_equity <= 50:
        score += 4
        notes.append("負債資本比率が比較的低い")
    elif debt_to_equity <= 100:
        score += 3
    elif debt_to_equity <= 200:
        score += 1

    revenue_growth = _safe_float(
        info.get("revenueGrowth")
    )

    if revenue_growth is None:
        score += 2
    elif revenue_growth >= 0.10:
        score += 4
        notes.append("売上高成長率が10%以上")
    elif revenue_growth > 0:
        score += 3
    elif revenue_growth > -0.05:
        score += 1

    return min(score, 20.0), notes


def _verdict(score: float) -> str:
    if score >= 75:
        return "強いポジティブ"
    if score >= 60:
        return "ポジティブ"
    if score >= 45:
        return "中立"
    if score >= 30:
        return "慎重"
    return "弱い"


def analyze_stock(
    symbol: str,
    prices: pd.DataFrame,
    info: Optional[dict[str, Any]] = None,
    market_prices: Optional[pd.DataFrame] = None,
    sector_prices: Optional[pd.DataFrame] = None,
) -> AnalysisResult:
    info = info or {}
    warnings: list[str] = []

    frame = calculate_indicators(prices)

    if frame.empty:
        raise ValueError("分析可能な価格データがありません。")

    if len(frame) < 50:
        warnings.append(
            "価格履歴が50営業日未満のため、"
            "中長期指標の信頼性が限定的です。"
        )
    elif len(frame) < 200:
        warnings.append(
            "価格履歴が200営業日未満のため、"
            "200日移動平均線は未確定です。"
        )

    latest_row = frame.iloc[-1]

    close = _safe_float(latest_row["Close"]) or 0.0
    previous_close = (
        _safe_float(frame["Close"].iloc[-2])
        if len(frame) >= 2
        else None
    )

    sma20 = _safe_float(latest_row["SMA20"])
    sma50 = _safe_float(latest_row["SMA50"])
    sma200 = _safe_float(latest_row["SMA200"])

    rsi = _safe_float(latest_row["RSI14"])
    macd = _safe_float(latest_row["MACD"])
    macd_signal = _safe_float(latest_row["MACDSignal"])

    k_value = _safe_float(latest_row["K"])
    d_value = _safe_float(latest_row["D"])
    j_value = _safe_float(latest_row["J"])

    volume = _safe_float(latest_row["Volume"]) or 0.0
    volume_average = _safe_float(
        latest_row["VolumeSMA20"]
    )

    obv = _safe_float(latest_row["OBV"])
    obv_average = _safe_float(latest_row["OBVSMA10"])

    # トレンド：最大30点
    trend_score = 0.0

    if sma20 is not None and close > sma20:
        trend_score += 5

    if sma50 is not None and close > sma50:
        trend_score += 6

    if sma200 is None:
        trend_score += 4
    elif close > sma200:
        trend_score += 8

    if sma20 is not None and sma50 is not None and sma20 > sma50:
        trend_score += 5

    if sma50 is None or sma200 is None:
        trend_score += 3
    elif sma50 > sma200:
        trend_score += 6

    trend_score = min(trend_score, 30)

    # モメンタム：最大25点
    momentum_score = 0.0

    if rsi is None:
        momentum_score += 4
    elif 45 <= rsi <= 65:
        momentum_score += 8
    elif 35 <= rsi <= 75:
        momentum_score += 5
    else:
        momentum_score += 1

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            momentum_score += 7

        if macd > 0:
            momentum_score += 4
    else:
        momentum_score += 5

    if k_value is not None and d_value is not None:
        if k_value > d_value:
            momentum_score += 3
    else:
        momentum_score += 1.5

    if j_value is None:
        momentum_score += 1.5
    elif 20 <= j_value <= 100:
        momentum_score += 3

    momentum_score = min(momentum_score, 25)

    # 出来高：最大10点
    volume_score = 0.0

    latest_return = _safe_float(latest_row["Return"])

    if volume_average is None or volume_average <= 0:
        volume_score += 5
    else:
        volume_ratio = volume / volume_average

        if volume_ratio >= 1.2 and (latest_return or 0) > 0:
            volume_score += 6
        elif volume_ratio >= 0.7:
            volume_score += 3
        else:
            volume_score += 1

    if obv is None or obv_average is None:
        volume_score += 2
    elif obv > obv_average:
        volume_score += 4

    volume_score = min(volume_score, 10)

    # 相対強度：最大15点
    primary_return = _period_return(frame, 63)
    market_return = _period_return(market_prices, 63)
    sector_return = _period_return(sector_prices, 63)

    market_relative_return: Optional[float] = None
    sector_relative_return: Optional[float] = None

    relative_score = 0.0

    if primary_return is None:
        relative_score += 7.5
    else:
        if market_return is None:
            relative_score += 4
        else:
            market_relative_return = primary_return - market_return

            if market_relative_return >= 5:
                relative_score += 8
            elif market_relative_return >= 0:
                relative_score += 6
            elif market_relative_return >= -5:
                relative_score += 3
            else:
                relative_score += 1

        if sector_return is None:
            relative_score += 3.5
        else:
            sector_relative_return = primary_return - sector_return

            if sector_relative_return >= 5:
                relative_score += 7
            elif sector_relative_return >= 0:
                relative_score += 5
            elif sector_relative_return >= -5:
                relative_score += 2
            else:
                relative_score += 0.5

    relative_score = min(relative_score, 15)

    fundamental_score, fundamental_notes = _fundamental_score(info)

    components = {
        "トレンド": round(trend_score, 1),
        "モメンタム": round(momentum_score, 1),
        "出来高": round(volume_score, 1),
        "相対強度": round(relative_score, 1),
        "ファンダメンタルズ": round(fundamental_score, 1),
    }

    component_maximums = {
        "トレンド": 30.0,
        "モメンタム": 25.0,
        "出来高": 10.0,
        "相対強度": 15.0,
        "ファンダメンタルズ": 20.0,
    }

    total_score = round(
        min(100.0, max(0.0, sum(components.values()))),
        1,
    )

    signals: list[str] = []

    if sma20 is not None:
        if close > sma20:
            signals.append("終値は20日移動平均線を上回っています。")
        else:
            signals.append("終値は20日移動平均線を下回っています。")

    if (
        sma20 is not None
        and sma50 is not None
        and sma200 is not None
    ):
        if sma20 > sma50 > sma200:
            signals.append(
                "短期・中期・長期移動平均線は上昇配列です。"
            )
        elif sma20 < sma50 < sma200:
            signals.append(
                "短期・中期・長期移動平均線は下降配列です。"
            )

    if rsi is not None:
        if rsi >= 70:
            signals.append(
                "RSIは70以上で、短期的な過熱に注意が必要です。"
            )
        elif rsi <= 30:
            signals.append(
                "RSIは30以下で、売られ過ぎ圏にあります。"
            )
        else:
            signals.append(
                "RSIは極端な過熱・売られ過ぎ圏にはありません。"
            )

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            signals.append(
                "MACDはシグナル線を上回っています。"
            )
        else:
            signals.append(
                "MACDはシグナル線を下回っています。"
            )

    if market_relative_return is not None:
        signals.append(
            "直近約3か月の市場相対リターンは"
            f"{market_relative_return:+.1f}ポイントです。"
        )

    if sector_relative_return is not None:
        signals.append(
            "直近約3か月のセクター相対リターンは"
            f"{sector_relative_return:+.1f}ポイントです。"
        )

    signals.extend(fundamental_notes)

    reference = (
        frame.iloc[:-1].tail(60)
        if len(frame) > 1
        else frame.tail(60)
    )

    support = _safe_float(reference["Low"].min())
    resistance = _safe_float(reference["High"].max())

    change = None
    change_percent = None

    if previous_close is not None and previous_close != 0:
        change = close - previous_close
        change_percent = (change / previous_close) * 100

    latest = {
        "date": frame.index[-1],
        "close": close,
        "change": change,
        "change_percent": change_percent,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_histogram": _safe_float(
            latest_row["MACDHistogram"]
        ),
        "k": k_value,
        "d": d_value,
        "j": j_value,
        "atr14": _safe_float(latest_row["ATR14"]),
        "volume": volume,
        "volume_average": volume_average,
        "primary_return_3m": primary_return,
    }

    return AnalysisResult(
        symbol=symbol,
        frame=frame,
        score=total_score,
        verdict=_verdict(total_score),
        components=components,
        component_maximums=component_maximums,
        latest=latest,
        signals=signals,
        candlestick_patterns=_detect_candlestick_patterns(frame),
        support=support,
        resistance=resistance,
        market_relative_return=market_relative_return,
        sector_relative_return=sector_relative_return,
        warnings=warnings,
    )
