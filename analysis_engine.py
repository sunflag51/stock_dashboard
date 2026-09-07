from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ScoreItem:
    category: str
    label: str
    points: int
    passed: bool
    available: bool
    detail: str = ""

    @property
    def earned(self) -> int:
        if self.available and self.passed:
            return self.points
        return 0


@dataclass
class ScoreResult:
    items: list[ScoreItem] = field(default_factory=list)
    earned: int = 0
    available: int = 0
    theoretical: int = 100
    normalized_score: int | None = None
    completeness: int = 0
    category_scores: dict[str, dict[str, int | None]] = field(
        default_factory=dict
    )
    status: str = ""
    status_color: str = "gray"


def safe_number(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def calculate_rsi_wilder(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder方式のRSI。
    一般的なチャートソフトに近い計算方式です。
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 上昇だけが続いた場合
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)

    # 値動きが全くない場合
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    return rsi


def add_indicators(source_df: pd.DataFrame) -> pd.DataFrame:
    df = source_df.copy()

    # 移動平均・ボリンジャーバンド
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["STD20"] = df["Close"].rolling(20).std(ddof=0)
    df["Upper"] = df["MA20"] + 2 * df["STD20"]
    df["Lower"] = df["MA20"] - 2 * df["STD20"]

    df["Vol_MA20"] = df["Volume"].rolling(20).mean()

    # 一目均衡表
    high9 = df["High"].rolling(9).max()
    low9 = df["Low"].rolling(9).min()
    df["Tenkan"] = (high9 + low9) / 2

    high26 = df["High"].rolling(26).max()
    low26 = df["Low"].rolling(26).min()
    df["Kijun"] = (high26 + low26) / 2

    df["SenkouA"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)

    high52 = df["High"].rolling(52).max()
    low52 = df["Low"].rolling(52).min()
    df["SenkouB"] = ((high52 + low52) / 2).shift(26)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # Wilder RSI
    df["RSI_9"] = calculate_rsi_wilder(df["Close"], 9)
    df["RSI_14"] = calculate_rsi_wilder(df["Close"], 14)

    # KDJ
    low_nine = df["Low"].rolling(9).min()
    high_nine = df["High"].rolling(9).max()
    denominator = (high_nine - low_nine).replace(0, np.nan)
    rsv = ((df["Close"] - low_nine) / denominator) * 100

    k_values = np.full(len(df), 50.0)
    d_values = np.full(len(df), 50.0)

    for i in range(1, len(df)):
        current_rsv = rsv.iloc[i]

        if pd.isna(current_rsv):
            k_values[i] = k_values[i - 1]
            d_values[i] = d_values[i - 1]
        else:
            k_values[i] = (
                (2 / 3) * k_values[i - 1]
                + (1 / 3) * current_rsv
            )
            d_values[i] = (
                (2 / 3) * d_values[i - 1]
                + (1 / 3) * k_values[i]
            )

    df["K"] = k_values
    df["D"] = d_values
    df["J"] = 3 * df["K"] - 2 * df["D"]

    # OBV
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (df["Volume"] * direction).cumsum()
    df["OBV_MA5"] = df["OBV"].rolling(5).mean()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # ATR
    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    return df


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    candle_range = (result["High"] - result["Low"]).replace(0, np.nan)
    body = (result["Close"] - result["Open"]).abs()

    lower_shadow = (
        result[["Open", "Close"]].min(axis=1) - result["Low"]
    )
    upper_shadow = (
        result["High"] - result[["Open", "Close"]].max(axis=1)
    )

    result["Hammer"] = (
        (body / candle_range <= 0.35)
        & (lower_shadow >= body * 2)
        & (upper_shadow <= candle_range * 0.25)
        & (
            result[["Open", "Close"]].max(axis=1)
            >= result["Low"] + candle_range * 0.60
        )
    )

    previous_open = result["Open"].shift(1)
    previous_close = result["Close"].shift(1)

    result["BullishEngulfing"] = (
        (previous_close < previous_open)
        & (result["Close"] > result["Open"])
        & (result["Open"] <= previous_close)
        & (result["Close"] >= previous_open)
    )

    return result


def detect_market_structure(df: pd.DataFrame) -> dict[str, Any]:
    """
    確定済みのスイング安値だけを使って、
    安値切り上げとネックライン突破を判定します。

    中央の安値を左右2日と比較するため、
    直近2本はスイング安値として確定しません。
    """
    result: dict[str, Any] = {
        "available": False,
        "higher_low": False,
        "recent_higher_low": False,
        "neckline_breakout": False,
        "first_low": None,
        "second_low": None,
        "neckline": None,
    }

    if len(df) < 30:
        return result

    low = df["Low"]

    swing_low = (
        (low < low.shift(1))
        & (low <= low.shift(2))
        & (low < low.shift(-1))
        & (low <= low.shift(-2))
    )

    # 直近2本は未来側の確認がないので除外
    swing_low.iloc[-2:] = False

    positions = np.flatnonzero(swing_low.fillna(False).to_numpy())

    # 古すぎるスイングは使わない
    positions = positions[positions >= max(0, len(df) - 120)]

    if len(positions) < 2:
        return result

    first_position = int(positions[-2])
    second_position = int(positions[-1])

    first_low = float(df["Low"].iloc[first_position])
    second_low = float(df["Low"].iloc[second_position])

    between_highs = df["High"].iloc[
        first_position:second_position + 1
    ]

    if between_highs.empty:
        return result

    neckline = float(between_highs.max())

    higher_low = second_low > first_low
    recent_higher_low = (
        higher_low
        and second_position >= len(df) - 35
    )

    close_above = df["Close"] > neckline
    breakout_cross = close_above & ~close_above.shift(1).fillna(False)
    recent_breakout = bool(breakout_cross.tail(5).any())

    result.update(
        {
            "available": True,
            "higher_low": higher_low,
            "recent_higher_low": recent_higher_low,
            "neckline_breakout": recent_higher_low and recent_breakout,
            "first_low": first_low,
            "second_low": second_low,
            "neckline": neckline,
        }
    )

    return result


def crossed_up_recently(
    left: pd.Series,
    right: pd.Series | float,
    lookback: int = 3,
) -> bool:
    if isinstance(right, pd.Series):
        crossed = (left > right) & (left.shift(1) <= right.shift(1))
    else:
        crossed = (left > right) & (left.shift(1) <= right)

    return bool(crossed.tail(lookback).fillna(False).any())


def _benchmark_items(
    frame: pd.DataFrame | None,
    category: str,
    label_prefix: str,
) -> list[ScoreItem]:
    if frame is None or frame.empty or len(frame) < 55:
        return [
            ScoreItem(
                category=category,
                label=f"{label_prefix}：50日移動平均",
                points=5,
                passed=False,
                available=False,
                detail="データ不足",
            ),
            ScoreItem(
                category=category,
                label=f"{label_prefix}：1か月騰落率",
                points=5,
                passed=False,
                available=False,
                detail="データ不足",
            ),
        ]

    close = frame["Close"]
    ma50 = close.rolling(50).mean().iloc[-1]
    current = close.iloc[-1]
    month_ago = close.iloc[-22]

    return [
        ScoreItem(
            category=category,
            label=f"{label_prefix}：株価が50日線より上",
            points=5,
            passed=bool(current > ma50),
            available=not pd.isna(ma50),
            detail=f"終値 {current:.2f} / MA50 {ma50:.2f}",
        ),
        ScoreItem(
            category=category,
            label=f"{label_prefix}：直近1か月が上昇",
            points=5,
            passed=bool(current > month_ago),
            available=not pd.isna(month_ago),
            detail=f"1か月騰落率 {(current / month_ago - 1) * 100:.1f}%",
        ),
    ]


def calculate_score(
    df: pd.DataFrame,
    info: dict[str, Any],
    market_df: pd.DataFrame | None,
    sector_df: pd.DataFrame | None,
    market_label: str = "市場",
    sector_label: str = "セクター",
) -> ScoreResult:
    items: list[ScoreItem] = []

    # =========================================================
    # 企業品質 20点
    # =========================================================
    revenue_growth = safe_number(info.get("revenueGrowth"))
    operating_margin = safe_number(info.get("operatingMargins"))
    operating_cf = safe_number(info.get("operatingCashflow"))
    free_cf = safe_number(info.get("freeCashflow"))
    trailing_pe = safe_number(info.get("trailingPE"))
    forward_pe = safe_number(info.get("forwardPE"))

    items.extend(
        [
            ScoreItem(
                "企業品質",
                "売上成長率がプラス",
                4,
                revenue_growth is not None and revenue_growth > 0,
                revenue_growth is not None,
                (
                    f"{revenue_growth * 100:.1f}%"
                    if revenue_growth is not None
                    else "取得不可"
                ),
            ),
            ScoreItem(
                "企業品質",
                "営業利益率が5%以上",
                4,
                operating_margin is not None and operating_margin >= 0.05,
                operating_margin is not None,
                (
                    f"{operating_margin * 100:.1f}%"
                    if operating_margin is not None
                    else "取得不可"
                ),
            ),
            ScoreItem(
                "企業品質",
                "営業キャッシュフローがプラス",
                4,
                operating_cf is not None and operating_cf > 0,
                operating_cf is not None,
                "プラス" if operating_cf and operating_cf > 0 else "取得不可またはマイナス",
            ),
            ScoreItem(
                "企業品質",
                "フリーキャッシュフローがプラス",
                4,
                free_cf is not None and free_cf > 0,
                free_cf is not None,
                "プラス" if free_cf and free_cf > 0 else "取得不可またはマイナス",
            ),
            ScoreItem(
                "企業品質",
                "予想PERが実績PERを下回る",
                4,
                (
                    trailing_pe is not None
                    and forward_pe is not None
                    and forward_pe > 0
                    and trailing_pe > forward_pe
                ),
                (
                    trailing_pe is not None
                    and forward_pe is not None
                    and trailing_pe > 0
                    and forward_pe > 0
                ),
                (
                    f"実績 {trailing_pe:.1f} / 予想 {forward_pe:.1f}"
                    if trailing_pe is not None and forward_pe is not None
                    else "取得不可"
                ),
            ),
        ]
    )

    # =========================================================
    # 市場環境 20点
    # =========================================================
    items.extend(
        _benchmark_items(
            market_df,
            "市場環境",
            market_label,
        )
    )
    items.extend(
        _benchmark_items(
            sector_df,
            "市場環境",
            sector_label,
        )
    )

    # =========================================================
    # 売られ過ぎ 20点
    # =========================================================
    recent10 = df.tail(10)
    recent60 = df.tail(60)

    rsi_available = recent10["RSI_14"].notna().any()
    rsi_oversold = (
        rsi_available
        and recent10["RSI_14"].min() <= 30
    )

    bb_available = recent10["Lower"].notna().any()
    bb_touch = (
        bb_available
        and bool((recent10["Low"] <= recent10["Lower"]).any())
    )

    sixty_high = recent60["High"].max()
    current_close = df["Close"].iloc[-1]
    drawdown = (
        current_close / sixty_high - 1
        if sixty_high and not pd.isna(sixty_high)
        else None
    )

    ma20_distance = (
        (df["Close"] / df["MA20"] - 1)
        .tail(10)
        .min()
    )

    items.extend(
        [
            ScoreItem(
                "売られ過ぎ",
                "直近10日でRSI(14)が30以下",
                6,
                rsi_oversold,
                rsi_available,
                (
                    f"最小RSI {recent10['RSI_14'].min():.1f}"
                    if rsi_available
                    else "計算不可"
                ),
            ),
            ScoreItem(
                "売られ過ぎ",
                "直近10日でボリンジャー-2σに到達",
                6,
                bb_touch,
                bb_available,
                "到達あり" if bb_touch else "到達なし",
            ),
            ScoreItem(
                "売られ過ぎ",
                "60日高値から10%以上下落",
                4,
                drawdown is not None and drawdown <= -0.10,
                drawdown is not None,
                (
                    f"{drawdown * 100:.1f}%"
                    if drawdown is not None
                    else "計算不可"
                ),
            ),
            ScoreItem(
                "売られ過ぎ",
                "直近10日で20日線から5%以上下方乖離",
                4,
                (
                    ma20_distance is not None
                    and not pd.isna(ma20_distance)
                    and ma20_distance <= -0.05
                ),
                ma20_distance is not None and not pd.isna(ma20_distance),
                (
                    f"最大下方乖離 {ma20_distance * 100:.1f}%"
                    if ma20_distance is not None
                    and not pd.isna(ma20_distance)
                    else "計算不可"
                ),
            ),
        ]
    )

    # =========================================================
    # 反転確認 40点
    # =========================================================
    rsi_cross = crossed_up_recently(
        df["RSI_14"],
        30,
        lookback=3,
    )

    macd_cross = crossed_up_recently(
        df["MACD"],
        df["Signal"],
        lookback=3,
    )

    kdj_cross_series = (
        (df["K"] > df["D"])
        & (df["K"].shift(1) <= df["D"].shift(1))
        & (df["K"] < 40)
    )
    kdj_cross = bool(kdj_cross_series.tail(3).fillna(False).any())

    recent_patterns = df.tail(3)
    pattern_detected = bool(
        recent_patterns["Hammer"].fillna(False).any()
        or recent_patterns["BullishEngulfing"].fillna(False).any()
    )

    volume_confirmation = bool(
        (
            (df["Close"] > df["Close"].shift(1))
            & (df["Volume"] > df["Vol_MA20"] * 1.5)
        )
        .tail(3)
        .fillna(False)
        .any()
    )

    obv_confirmation = bool(
        not pd.isna(df["OBV_MA5"].iloc[-1])
        and not pd.isna(df["OBV_MA20"].iloc[-1])
        and df["OBV_MA5"].iloc[-1] > df["OBV_MA20"].iloc[-1]
        and df["OBV_MA5"].iloc[-1] > df["OBV_MA5"].iloc[-5]
    )

    structure = detect_market_structure(df)

    ma20_reclaim = crossed_up_recently(
        df["Close"],
        df["MA20"],
        lookback=3,
    )

    items.extend(
        [
            ScoreItem(
                "反転確認",
                "RSIが30を上抜け",
                6,
                rsi_cross,
                df["RSI_14"].notna().sum() >= 15,
                f"現在 {df['RSI_14'].iloc[-1]:.1f}",
            ),
            ScoreItem(
                "反転確認",
                "MACDが直近3日でゴールデンクロス",
                6,
                macd_cross,
                df["Signal"].notna().sum() >= 30,
                (
                    f"MACD {df['MACD'].iloc[-1]:.3f} / "
                    f"Signal {df['Signal'].iloc[-1]:.3f}"
                ),
            ),
            ScoreItem(
                "反転確認",
                "KDJが低位でゴールデンクロス",
                4,
                kdj_cross,
                df["D"].notna().sum() >= 10,
                f"K {df['K'].iloc[-1]:.1f} / D {df['D'].iloc[-1]:.1f}",
            ),
            ScoreItem(
                "反転確認",
                "ハンマーまたは強気包み足",
                5,
                pattern_detected,
                len(df) >= 3,
                "直近3日を判定",
            ),
            ScoreItem(
                "反転確認",
                "上昇日に出来高が20日平均の1.5倍",
                5,
                volume_confirmation,
                df["Vol_MA20"].notna().sum() > 0,
                "直近3日を判定",
            ),
            ScoreItem(
                "反転確認",
                "OBVの短期傾向が上向き",
                4,
                obv_confirmation,
                df["OBV_MA20"].notna().sum() > 0,
                "OBVの5日平均と20日平均を比較",
            ),
            ScoreItem(
                "反転確認",
                "確定スイング安値が切り上がった",
                4,
                bool(structure["recent_higher_low"]),
                bool(structure["available"]),
                (
                    f"第1安値 {structure['first_low']:.2f} / "
                    f"第2安値 {structure['second_low']:.2f}"
                    if structure["available"]
                    else "確定スイング安値が不足"
                ),
            ),
            ScoreItem(
                "反転確認",
                "安値切り上げ後にネックライン突破",
                4,
                bool(structure["neckline_breakout"]),
                bool(structure["available"]),
                (
                    f"ネックライン {structure['neckline']:.2f}"
                    if structure["available"]
                    else "判定不可"
                ),
            ),
            ScoreItem(
                "反転確認",
                "終値が20日移動平均線を回復",
                2,
                ma20_reclaim,
                df["MA20"].notna().sum() > 0,
                "直近3日での上抜けを判定",
            ),
        ]
    )

    earned = sum(item.earned for item in items)
    available = sum(
        item.points for item in items if item.available
    )
    theoretical = sum(item.points for item in items)

    normalized_score = (
        round(earned / available * 100)
        if available > 0
        else None
    )

    completeness = (
        round(available / theoretical * 100)
        if theoretical > 0
        else 0
    )

    categories = ["企業品質", "市場環境", "売られ過ぎ", "反転確認"]
    category_scores: dict[str, dict[str, int | None]] = {}

    for category in categories:
        category_items = [
            item for item in items if item.category == category
        ]
        category_earned = sum(item.earned for item in category_items)
        category_available = sum(
            item.points for item in category_items if item.available
        )
        category_theoretical = sum(
            item.points for item in category_items
        )

        category_normalized = (
            round(category_earned / category_available * 100)
            if category_available > 0
            else None
        )

        category_scores[category] = {
            "earned": category_earned,
            "available": category_available,
            "theoretical": category_theoretical,
            "normalized": category_normalized,
        }

    oversold_score = (
        category_scores["売られ過ぎ"]["normalized"] or 0
    )
    reversal_score = (
        category_scores["反転確認"]["normalized"] or 0
    )

    if completeness < 60:
        status = "⚪ データ不足のため判定信頼度が低い"
        status_color = "gray"
    elif reversal_score >= 70:
        status = "🟢 反転確認材料が多い"
        status_color = "green"
    elif reversal_score >= 45:
        status = "🟡 初期反転を確認中"
        status_color = "orange"
    elif oversold_score >= 60:
        status = "🟠 売られ過ぎだが反転は未確認"
        status_color = "darkorange"
    else:
        status = "🔴 下降継続、または反転材料が不足"
        status_color = "red"

    return ScoreResult(
        items=items,
        earned=earned,
        available=available,
        theoretical=theoretical,
        normalized_score=normalized_score,
        completeness=completeness,
        category_scores=category_scores,
        status=status,
        status_color=status_color,
    )
