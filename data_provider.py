from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


REQUIRED_PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


@dataclass
class InstrumentData:
    symbol: str
    prices: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    dropped_incomplete_bar: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class MarketBundle:
    primary: InstrumentData
    market: Optional[InstrumentData]
    sector: Optional[InstrumentData]
    market_symbol: Optional[str]
    sector_symbol: Optional[str]
    warnings: list[str] = field(default_factory=list)


class MarketDataProvider(ABC):
    """
    yfinanceや将来追加するデータ取得元を統一するインターフェースです。
    """

    name: str = "unknown"

    @abstractmethod
    def get_instrument(
        self,
        symbol: str,
        period: str = "5y",
        include_info: bool = False,
        exclude_incomplete: bool = True,
    ) -> InstrumentData:
        raise NotImplementedError


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol or "").upper().strip()

    if not symbol:
        raise ValueError("銘柄コードを入力してください。")

    if len(symbol) > 30:
        raise ValueError("銘柄コードが長すぎます。")

    allowed_characters = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=_/"
    )

    if any(character not in allowed_characters for character in symbol):
        raise ValueError(
            "銘柄コードに使用できない文字が含まれています。"
        )

    return symbol


def get_benchmark_symbols(
    symbol: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    銘柄に対応する市場ベンチマークとセクターETFを返します。
    """
    symbol = normalize_symbol(symbol)

    if symbol.endswith(".T"):
        return "^N225", None

    sector_map = {
        # 半導体
        "NVDA": "SMH",
        "AMD": "SMH",
        "TSM": "SMH",
        "AVGO": "SMH",
        "INTC": "SMH",
        "QCOM": "SMH",
        "ARM": "SMH",
        "MU": "SMH",
        "ASML": "SMH",

        # 情報技術
        "AAPL": "XLK",
        "MSFT": "XLK",
        "CRM": "XLK",
        "ORCL": "XLK",
        "ADBE": "XLK",
        "IBM": "XLK",
        "ACN": "XLK",

        # コミュニケーション
        "GOOG": "XLC",
        "GOOGL": "XLC",
        "META": "XLC",
        "NFLX": "XLC",
        "DIS": "XLC",
        "TMUS": "XLC",
        "VZ": "XLC",
        "T": "XLC",

        # 一般消費財
        "AMZN": "XLY",
        "TSLA": "XLY",
        "NKE": "XLY",
        "HD": "XLY",
        "MCD": "XLY",
        "LOW": "XLY",
        "SBUX": "XLY",

        # 生活必需品
        "KO": "XLP",
        "PEP": "XLP",
        "PG": "XLP",
        "KDP": "XLP",
        "COST": "XLP",
        "WMT": "XLP",
        "TGT": "XLP",
        "BJ": "XLP",

        # 金融
        "V": "XLF",
        "MA": "XLF",
        "AXP": "XLF",
        "PYPL": "XLF",
        "JPM": "XLF",
        "BAC": "XLF",
        "GS": "XLF",
        "MS": "XLF",
        "WFC": "XLF",

        # ヘルスケア
        "ISRG": "XLV",
        "JNJ": "XLV",
        "UNH": "XLV",
        "MDT": "XLV",
        "SYK": "XLV",
        "PFE": "XLV",
        "MRK": "XLV",
        "LLY": "XLV",
        "ABBV": "XLV",

        # エネルギー
        "XOM": "XLE",
        "CVX": "XLE",
        "COP": "XLE",
        "SLB": "XLE",

        # 資本財
        "CAT": "XLI",
        "BA": "XLI",
        "GE": "XLI",
        "HON": "XLI",
        "UPS": "XLI",

        # 公益
        "NEE": "XLU",
        "DUK": "XLU",
        "SO": "XLU",

        # 不動産
        "AMT": "XLRE",
        "PLD": "XLRE",
        "EQIX": "XLRE",

        # 素材
        "LIN": "XLB",
        "FCX": "XLB",
        "NEM": "XLB",
        "DOW": "XLB",
    }

    return "SPY", sector_map.get(symbol)


def _default_timezone_for_symbol(symbol: str) -> str:
    if symbol.endswith(".T") or symbol == "^N225":
        return "Asia/Tokyo"

    if symbol.endswith(".L"):
        return "Europe/London"

    if symbol.endswith(".TO"):
        return "America/Toronto"

    if symbol.endswith(".AX"):
        return "Australia/Sydney"

    if symbol.endswith(".HK"):
        return "Asia/Hong_Kong"

    return "America/New_York"


def _market_close_time(timezone_name: str) -> time:
    close_times = {
        "America/New_York": time(16, 0),
        "America/Toronto": time(16, 0),
        "Asia/Tokyo": time(15, 30),
        "Europe/London": time(16, 30),
        "Europe/Berlin": time(17, 30),
        "Australia/Sydney": time(16, 0),
        "Asia/Hong_Kong": time(16, 0),
    }

    return close_times.get(timezone_name, time(16, 0))


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if not isinstance(result.columns, pd.MultiIndex):
        return result

    best_level = 0
    best_score = -1

    for level in range(result.columns.nlevels):
        values = {
            str(value)
            for value in result.columns.get_level_values(level)
        }
        score = len(values.intersection(REQUIRED_PRICE_COLUMNS))

        if score > best_score:
            best_score = score
            best_level = level

    result.columns = result.columns.get_level_values(best_level)
    result = result.loc[
        :,
        ~pd.Index(result.columns).duplicated(keep="last"),
    ]

    return result


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)

    result = _flatten_yfinance_columns(df)

    for column in REQUIRED_PRICE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[REQUIRED_PRICE_COLUMNS].copy()

    try:
        result.index = pd.to_datetime(result.index)
    except Exception:
        pass

    result = result[
        ~result.index.duplicated(keep="last")
    ].sort_index()

    for column in REQUIRED_PRICE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.replace([float("inf"), float("-inf")], pd.NA)
    result = result.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    valid_prices = (
        (result["Open"] > 0)
        & (result["High"] > 0)
        & (result["Low"] > 0)
        & (result["Close"] > 0)
        & (result["High"] >= result["Low"])
    )

    result = result.loc[valid_prices].copy()
    result["Volume"] = result["Volume"].fillna(0).clip(lower=0)

    return result


def _exclude_incomplete_daily_bar(
    df: pd.DataFrame,
    timezone_name: str,
    grace_minutes: int = 20,
) -> Tuple[pd.DataFrame, bool]:
    """
    当日の日足が市場終了前の場合に、その足を除外します。

    短縮取引日や臨時休場を完全に判定する取引所カレンダーではなく、
    通常の市場終了時刻を使った安全側の簡易判定です。
    """
    if df.empty:
        return df, False

    try:
        market_tz = ZoneInfo(timezone_name)
    except Exception:
        timezone_name = "America/New_York"
        market_tz = ZoneInfo(timezone_name)

    now_market = datetime.now(market_tz)
    last_timestamp = pd.Timestamp(df.index[-1])

    if last_timestamp.tzinfo is not None:
        last_date = last_timestamp.tz_convert(market_tz).date()
    else:
        last_date = last_timestamp.date()

    today_market = now_market.date()

    if last_date < today_market:
        return df, False

    if last_date > today_market:
        return df.iloc[:-1].copy(), True

    close_datetime = datetime.combine(
        today_market,
        _market_close_time(timezone_name),
        tzinfo=market_tz,
    )

    complete_after = close_datetime + timedelta(
        minutes=max(0, grace_minutes)
    )

    if now_market < complete_after:
        return df.iloc[:-1].copy(), True

    return df, False


class YahooFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def get_instrument(
        self,
        symbol: str,
        period: str = "5y",
        include_info: bool = False,
        exclude_incomplete: bool = True,
    ) -> InstrumentData:
        symbol = normalize_symbol(symbol)
        warnings: list[str] = []

        ticker = yf.Ticker(symbol)

        try:
            raw_df = ticker.history(
                period=period,
                interval="1d",
                auto_adjust=True,
                actions=False,
                repair=True,
                timeout=20,
            )
        except TypeError:
            # 一部のyfinanceバージョンで未対応の引数がある場合
            try:
                raw_df = ticker.history(
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{symbol}の価格データ取得に失敗しました: {exc}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{symbol}の価格データ取得に失敗しました: {exc}"
            ) from exc

        prices = _normalize_price_frame(raw_df)

        metadata: dict[str, Any] = {}

        try:
            metadata = ticker.get_history_metadata() or {}
        except Exception:
            warnings.append(
                f"{symbol}の市場メタデータを取得できませんでした。"
            )

        timezone_name = (
            metadata.get("exchangeTimezoneName")
            or metadata.get("timezone")
            or _default_timezone_for_symbol(symbol)
        )

        dropped = False

        if exclude_incomplete and not prices.empty:
            prices, dropped = _exclude_incomplete_daily_bar(
                prices=prices,
                timezone_name=str(timezone_name),
            )

        info: dict[str, Any] = {}

        if include_info:
            try:
                info = ticker.info or {}
            except Exception:
                warnings.append(
                    f"{symbol}の企業情報を取得できなかったため、"
                    "ファンダメンタルズは一部欠損扱いになります。"
                )

        if dropped:
            warnings.append(
                f"{symbol}の未確定と判断された当日足を除外しました。"
            )

        if prices.empty:
            warnings.append(
                f"{symbol}の有効な価格データがありません。"
            )
        elif len(prices) < 50:
            warnings.append(
                f"{symbol}の価格履歴が50営業日未満です。"
                "分析精度が低下する可能性があります。"
            )

        return InstrumentData(
            symbol=symbol,
            prices=prices,
            info=info,
            metadata=metadata,
            source=self.name,
            dropped_incomplete_bar=dropped,
            warnings=warnings,
        )


def create_provider(
    provider_name: str = "yfinance",
) -> MarketDataProvider:
    normalized = str(provider_name).lower().strip()

    if normalized == "yfinance":
        return YahooFinanceProvider()

    raise ValueError(
        f"未対応のデータプロバイダーです: {provider_name}"
    )


def load_market_bundle(
    provider_name: str,
    symbol: str,
) -> MarketBundle:
    symbol = normalize_symbol(symbol)
    provider = create_provider(provider_name)

    market_symbol, sector_symbol = get_benchmark_symbols(symbol)

    primary = provider.get_instrument(
        symbol=symbol,
        period="5y",
        include_info=True,
        exclude_incomplete=True,
    )

    warnings = list(primary.warnings)

    market: Optional[InstrumentData] = None

    if market_symbol and market_symbol != symbol:
        try:
            market = provider.get_instrument(
                symbol=market_symbol,
                period="1y",
                include_info=False,
                exclude_incomplete=True,
            )
            warnings.extend(market.warnings)
        except Exception as exc:
            warnings.append(
                f"市場データ（{market_symbol}）を"
                f"取得できませんでした: {exc}"
            )

    sector: Optional[InstrumentData] = None

    if sector_symbol and sector_symbol not in {symbol, market_symbol}:
        try:
            sector = provider.get_instrument(
                symbol=sector_symbol,
                period="1y",
                include_info=False,
                exclude_incomplete=True,
            )
            warnings.extend(sector.warnings)
        except Exception as exc:
            warnings.append(
                f"セクターデータ（{sector_symbol}）を"
                f"取得できませんでした: {exc}"
            )

    return MarketBundle(
        primary=primary,
        market=market,
        sector=sector,
        market_symbol=market_symbol,
        sector_symbol=sector_symbol,
        warnings=warnings,
    )
