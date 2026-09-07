from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any
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
    market: InstrumentData | None
    sector: InstrumentData | None
    market_symbol: str | None
    sector_symbol: str | None
    warnings: list[str] = field(default_factory=list)


class MarketDataProvider(ABC):
    """
    yfinance、Moomooなどのデータ取得元を統一するインターフェース。
    将来MoomooProviderを追加しても、app.py側の変更を少なくできます。
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


def get_benchmark_symbols(symbol: str) -> tuple[str | None, str | None]:
    """
    市場指数とセクターETFを返します。

    米国株:
        市場 = SPY
        セクター = 対応ETF

    日本株:
        市場 = ^N225
        セクター = None

    その他:
        市場 = SPY
        セクター = None
    """
    symbol = symbol.upper().strip()

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

        # ヘルスケア
        "ISRG": "XLV",
        "JNJ": "XLV",
        "UNH": "XLV",
        "MDT": "XLV",
        "SYK": "XLV",

        # コミュニケーション
        "GOOG": "XLC",
        "GOOGL": "XLC",
        "META": "XLC",
        "NFLX": "XLC",

        # 一般消費財
        "AMZN": "XLY",
        "TSLA": "XLY",
        "NKE": "XLY",
        "HD": "XLY",

        # 情報技術
        "AAPL": "XLK",
        "MSFT": "XLK",
        "CRM": "XLK",
        "ORCL": "XLK",

        # エネルギー
        "XOM": "XLE",
        "CVX": "XLE",

        # 資本財
        "CAT": "XLI",
        "BA": "XLI",

        # 公益
        "NEE": "XLU",
        "DUK": "XLU",

        # 不動産
        "AMT": "XLRE",
        "PLD": "XLRE",

        # 素材
        "LIN": "XLB",
        "FCX": "XLB",
    }

    return "SPY", sector_map.get(symbol)


def _market_close_time(timezone_name: str) -> time:
    """
    日足を確定扱いにする市場終了時刻。
    yfinanceの更新遅延を考慮し、別途20分の猶予を設けます。
    """
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


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)

    result = df.copy()

    # yfinance.download等でMultiIndexになった場合にも対応
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    for column in REQUIRED_PRICE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[REQUIRED_PRICE_COLUMNS]
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    for column in REQUIRED_PRICE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["Open", "High", "Low", "Close"])
    result["Volume"] = result["Volume"].fillna(0)

    return result


def _exclude_incomplete_daily_bar(
    df: pd.DataFrame,
    timezone_name: str,
    grace_minutes: int = 20,
) -> tuple[pd.DataFrame, bool]:
    """
    当日の日足が市場終了前なら除外します。

    市場終了後もyfinance側の反映に時間がかかる可能性があるため、
    grace_minutes分の猶予を設けます。
    """
    if df.empty:
        return df, False

    try:
        market_tz = ZoneInfo(timezone_name)
    except Exception:
        market_tz = ZoneInfo("America/New_York")
        timezone_name = "America/New_York"

    now_market = datetime.now(market_tz)
    last_timestamp = pd.Timestamp(df.index[-1])

    if last_timestamp.tzinfo is not None:
        last_date = last_timestamp.tz_convert(market_tz).date()
    else:
        last_date = last_timestamp.date()

    today_market = now_market.date()

    # 過去日なら確定足として扱う
    if last_date < today_market:
        return df, False

    # 将来日など、異常な日付の場合は安全のため除外
    if last_date > today_market:
        return df.iloc[:-1].copy(), True

    close_time = _market_close_time(timezone_name)
    close_datetime = datetime.combine(
        today_market,
        close_time,
        tzinfo=market_tz,
    )
    complete_after = close_datetime + timedelta(minutes=grace_minutes)

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
        symbol = symbol.upper().strip()
        warnings: list[str] = []

        ticker = yf.Ticker(symbol)

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

        prices = _normalize_price_frame(raw_df)

        metadata: dict[str, Any] = {}
        try:
            metadata = ticker.get_history_metadata() or {}
        except Exception as exc:
            warnings.append(
                f"{symbol}の市場メタデータを取得できませんでした。"
            )

        timezone_name = (
            metadata.get("exchangeTimezoneName")
            or metadata.get("timezone")
            or "America/New_York"
        )

        dropped = False
        if exclude_incomplete and not prices.empty:
            prices, dropped = _exclude_incomplete_daily_bar(
                prices,
                timezone_name=timezone_name,
            )

        info: dict[str, Any] = {}
        if include_info:
            try:
                info = ticker.info or {}
            except Exception:
                warnings.append(
                    f"{symbol}の企業情報を取得できなかったため、"
                    "ファンダメンタルズは欠損扱いになります。"
                )

        if prices.empty:
            warnings.append(f"{symbol}の有効な価格データがありません。")

        return InstrumentData(
            symbol=symbol,
            prices=prices,
            info=info,
            metadata=metadata,
            source=self.name,
            dropped_incomplete_bar=dropped,
            warnings=warnings,
        )


def create_provider(provider_name: str = "yfinance") -> MarketDataProvider:
    """
    将来ここにMoomooProviderを追加します。

    例:
        if provider_name == "moomoo":
            return MoomooProvider(...)
    """
    normalized = provider_name.lower().strip()

    if normalized == "yfinance":
        return YahooFinanceProvider()

    raise ValueError(f"未対応のデータプロバイダーです: {provider_name}")


def load_market_bundle(
    provider_name: str,
    symbol: str,
) -> MarketBundle:
    provider = create_provider(provider_name)
    market_symbol, sector_symbol = get_benchmark_symbols(symbol)

    primary = provider.get_instrument(
        symbol=symbol,
        period="5y",
        include_info=True,
        exclude_incomplete=True,
    )

    warnings = list(primary.warnings)

    market: InstrumentData | None = None
    if market_symbol:
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
                f"市場データ（{market_symbol}）を取得できませんでした: {exc}"
            )

    sector: InstrumentData | None = None
    if sector_symbol:
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
                f"セクターデータ（{sector_symbol}）を取得できませんでした: {exc}"
            )

    return MarketBundle(
        primary=primary,
        market=market,
        sector=sector,
        market_symbol=market_symbol,
        sector_symbol=sector_symbol,
        warnings=warnings,
    )
