from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Final, Any, Optional

import pandas as pd
import yfinance as yf


# ============================================================
# ログ設定
# ============================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# 定数
# ============================================================

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Open",
    "High",
    "Low",
    "Close",
)

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)


# ============================================================
# 戻り値
# ============================================================

@dataclass(frozen=True)
class InstrumentData:
    symbol: str
    provider_symbol: str
    prices: pd.DataFrame


# ============================================================
# 銘柄コード変換
# ============================================================

def normalize_yahoo_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise TypeError(
            f"銘柄コードは文字列で指定してください: {type(symbol).__name__}"
        )

    value = symbol.strip().upper()

    if not value:
        raise ValueError("銘柄コードが空です。")

    if value.endswith(".US"):
        return value[:-3]

    if value.endswith(".JP"):
        base = value[:-3]
        if not base:
            raise ValueError(f"無効な日本株コードです: {symbol!r}")
        return f"{base}.T"

    if value.endswith(".HK"):
        base = value[:-3]
        if base.isdigit():
            base = base.zfill(4)[-4:]
        return f"{base}.HK"

    return value


# ============================================================
# DataFrame正規化
# ============================================================

def _flatten_yfinance_columns(
    frame: pd.DataFrame,
    provider_symbol: str,
) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        result = frame.copy()
        result.columns = [str(column).strip() for column in result.columns]
        return result

    known_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }

    level_zero = {str(value).strip() for value in frame.columns.get_level_values(0)}
    level_one = {str(value).strip() for value in frame.columns.get_level_values(1)}

    result = frame.copy()

    if known_columns.intersection(level_zero):
        result.columns = [str(column[0]).strip() for column in result.columns]
        return result

    if known_columns.intersection(level_one):
        result.columns = [str(column[1]).strip() for column in result.columns]
        return result

    raise ValueError(
        f"{provider_symbol}の列構造を判定できませんでした。"
        f"columns={list(frame.columns)!r}"
    )


def _normalize_price_frame(
    frame: pd.DataFrame,
    provider_symbol: str,
) -> pd.DataFrame:
    if frame is None:
        raise ValueError(f"{provider_symbol}の価格データがNoneでした。")

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{provider_symbol}の取得結果がDataFrameではありません。")

    if frame.empty:
        raise ValueError(f"{provider_symbol}の価格データが空です。")

    result = _flatten_yfinance_columns(frame=frame, provider_symbol=provider_symbol)
    result = result.loc[:, ~result.columns.duplicated()].copy()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in result.columns]
    if missing_columns:
        raise KeyError(f"{provider_symbol}の価格データに必要な列がありません。")

    available_columns = [col for col in NUMERIC_COLUMNS if col in result.columns]
    result = result[available_columns].copy()

    for column in available_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=list(REQUIRED_COLUMNS), how="any")

    if result.empty:
        raise ValueError(f"{provider_symbol}は数値変換後に有効なデータが残りませんでした。")

    try:
        result.index = pd.to_datetime(result.index, errors="raise")
    except Exception as exc:
        raise ValueError(f"日付インデックス変換エラー: {exc}") from exc

    if isinstance(result.index, pd.DatetimeIndex):
        if result.index.tz is not None:
            result.index = result.index.tz_convert("UTC").tz_localize(None)

    result.index.name = "Date"
    result = result.loc[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    if result.empty:
        raise ValueError(f"{provider_symbol}の正規化後データが空です。")

    return result


def _interval_to_timedelta(interval: str) -> pd.Timedelta | None:
    interval_map: dict[str, pd.Timedelta] = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "1wk": pd.Timedelta(weeks=1),
    }
    return interval_map.get(interval.lower().strip())


def _exclude_incomplete_last_row(
    frame: pd.DataFrame,
    interval: str,
) -> pd.DataFrame:
    if len(frame) <= 1:
        return frame

    interval_delta = _interval_to_timedelta(interval)
    if interval_delta is None:
        return frame

    last_timestamp = pd.Timestamp(frame.index[-1])
    if last_timestamp.tzinfo is not None:
        last_timestamp = last_timestamp.tz_convert("UTC").tz_localize(None)

    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    expected_close = last_timestamp + interval_delta

    if now_utc < expected_close:
        return frame.iloc[:-1].copy()

    return frame


def _download_prices(
    provider_symbol: str,
    period: str,
    interval: str,
    auto_adjust: bool,
    timeout: int,
) -> pd.DataFrame:
    return yf.download(
        tickers=provider_symbol,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
        group_by="column",
        timeout=timeout,
    )


def get_instrument(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    exclude_incomplete: bool = True,
    auto_adjust: bool = False,
    retries: int = 3,
    retry_delay: float = 2.0,
    timeout: int = 20,
) -> InstrumentData:
    original_symbol = symbol.strip().upper()
    provider_symbol = normalize_yahoo_symbol(original_symbol)
    last_exception: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            raw_prices = _download_prices(provider_symbol, period, interval, auto_adjust, timeout)
            prices = _normalize_price_frame(raw_prices, provider_symbol)

            if exclude_incomplete:
                prices = _exclude_incomplete_last_row(prices, interval)

            return InstrumentData(symbol=original_symbol, provider_symbol=provider_symbol, prices=prices)

        except Exception as exc:
            last_exception = exc
            if attempt < retries:
                time.sleep(retry_delay * attempt)

    raise RuntimeError(f"{original_symbol}のデータ取得失敗: {last_exception}") from last_exception


def get_price_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    exclude_incomplete: bool = True,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    instrument = get_instrument(symbol, period, interval, exclude_incomplete, auto_adjust)
    return instrument.prices.copy()


# ============================================================
# アプリ（app.py）との連携用機能（ここから下を追記）
# ============================================================

@dataclass
class MarketData:
    symbol: str
    prices: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketBundle:
    primary: MarketData
    market: Optional[MarketData] = None
    sector: Optional[MarketData] = None
    market_symbol: Optional[str] = None
    sector_symbol: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

def normalize_symbol(symbol: str) -> str:
    """app.pyから呼ばれる関数。入力されたコードを大文字にする。"""
    if not symbol:
        raise ValueError("銘柄コードを入力してください。")
    return symbol.strip().upper()

def load_market_bundle(provider_name: str, symbol: str) -> MarketBundle:
    """
    app.pyから呼ばれるメインのデータ取得関数。
    上の堅牢な処理（get_instrument）を使って安全にデータを取得します。
    """
    warnings = []
    
    # 1. 銘柄の価格データ取得（2年分）
    try:
        instrument = get_instrument(
            symbol=symbol,
            period="2y", 
            interval="1d",
            exclude_incomplete=True,
            auto_adjust=False
        )
        primary_prices = instrument.prices
        provider_symbol = instrument.provider_symbol
    except Exception as e:
        primary_prices = pd.DataFrame()
        provider_symbol = normalize_yahoo_symbol(symbol)
        warnings.append(f"価格データの取得に失敗しました: {e}")

    # 2. 企業情報（ファンダメンタルズ）の取得
    try:
        ticker = yf.Ticker(provider_symbol)
        info = ticker.info
    except Exception as e:
        info = {}
        warnings.append(f"企業情報の取得に失敗しました: {e}")

    primary = MarketData(symbol=symbol, prices=primary_prices, info=info)

    # 3. 相対評価用の市場データの取得
    market_sym = "^N225" if provider_symbol.endswith(".T") else "^GSPC"
    market_name = "日経平均" if provider_symbol.endswith(".T") else "S&P500"
    
    try:
        market_instrument = get_instrument(
            symbol=market_sym,
            period="2y",
            interval="1d",
            exclude_incomplete=True,
            auto_adjust=False
        )
        market = MarketData(symbol=market_sym, prices=market_instrument.prices)
    except Exception as e:
        market = None
        warnings.append("比較用市場データの取得に失敗しました。")

    return MarketBundle(
        primary=primary,
        market=market,
        sector=None,
        market_symbol=market_name,
        sector_symbol=None,
        warnings=warnings
    )
