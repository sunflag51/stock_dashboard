from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Final

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
    """
    銘柄情報と価格データをまとめた戻り値。

    symbol:
        アプリで使用する元の銘柄コード。
        例: AAPL.US、7203.JP

    provider_symbol:
        Yahoo Financeへ送信した銘柄コード。
        例: AAPL、7203.T

    prices:
        正規化済みのOHLCVデータ。
    """

    symbol: str
    provider_symbol: str
    prices: pd.DataFrame


# ============================================================
# 銘柄コード変換
# ============================================================

def normalize_yahoo_symbol(symbol: str) -> str:
    """
    アプリ側の銘柄コードをYahoo Finance形式へ変換する。

    主な変換例:
        AAPL.US   -> AAPL
        7203.JP   -> 7203.T
         00700.HK   -> 0700.HK

    未対応のサフィックスは、そのまま返す。
    """

    if not isinstance(symbol, str):
        raise TypeError(
            f"銘柄コードは文字列で指定してください: {type(symbol).__name__}"
        )

    value = symbol.strip().upper()

    if not value:
        raise ValueError("銘柄コードが空です。")

    # 米国株
    if value.endswith(".US"):
        return value[:-3]

    # 日本株
    if value.endswith(".JP"):
        base = value[:-3]
        if not base:
            raise ValueError(f"無効な日本株コードです: {symbol!r}")
        return f"{base}.T"

    # 中国香港上場銘柄
    if value.endswith(".HK"):
        base = value[:-3]

        if base.isdigit():
            # Yahoo Financeでは通常4桁形式
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
    """
    yfinanceが返すMultiIndex列を通常の列へ変換する。

    yfinanceのバージョンや取得方法によって、次のような形式に
    なる場合がある。

        ('Close', 'AAPL')
        ('AAPL', 'Close')

    OHLCV名が含まれる階層を自動判定する。
    """

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

    level_zero = {
        str(value).strip()
        for value in frame.columns.get_level_values(0)
    }
    level_one = {
        str(value).strip()
        for value in frame.columns.get_level_values(1)
    }

    result = frame.copy()

    if known_columns.intersection(level_zero):
        result.columns = [
            str(column[0]).strip()
            for column in result.columns
        ]
        return result

    if known_columns.intersection(level_one):
        result.columns = [
            str(column[1]).strip()
            for column in result.columns
        ]
        return result

    raise ValueError(
        f"{provider_symbol}の列構造を判定できませんでした。"
        f"columns={list(frame.columns)!r}"
    )


def _normalize_price_frame(
    frame: pd.DataFrame,
    provider_symbol: str,
) -> pd.DataFrame:
    """
    Yahoo Financeの取得結果をアプリで扱いやすい形式へ正規化する。
    """

    if frame is None:
        raise ValueError(
            f"{provider_symbol}の価格データがNoneでした。"
        )

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{provider_symbol}の取得結果がDataFrameではありません。"
            f"type={type(frame).__name__}"
        )

    if frame.empty:
        raise ValueError(
            f"{provider_symbol}の価格データが空です。"
            "銘柄コード、期間、時間足を確認してください。"
        )

    result = _flatten_yfinance_columns(
        frame=frame,
        provider_symbol=provider_symbol,
    )

    # 同名列が重複した場合は最初の列を使用
    result = result.loc[:, ~result.columns.duplicated()].copy()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in result.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{provider_symbol}の価格データに必要な列がありません。"
            f"不足列={missing_columns}, "
            f"取得列={list(result.columns)}"
        )

    # 必要な列だけを残す
    available_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in result.columns
    ]

    result = result[available_columns].copy()

    # 数値変換
    for column in available_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # OHLCが欠損している行は除外
    result = result.dropna(
        subset=list(REQUIRED_COLUMNS),
        how="any",
    )

    if result.empty:
        raise ValueError(
            f"{provider_symbol}は数値変換後に有効な価格データが"
            "残りませんでした。"
        )

    # DatetimeIndexへ変換
    try:
        result.index = pd.to_datetime(
            result.index,
            errors="raise",
        )
    except Exception as exc:
        raise ValueError(
            f"{provider_symbol}の日付インデックスを変換できませんでした。"
            f"[{type(exc).__name__}]: {exc}"
        ) from exc

    # タイムゾーン情報を除去して比較しやすくする
    if isinstance(result.index, pd.DatetimeIndex):
        if result.index.tz is not None:
            result.index = result.index.tz_convert("UTC").tz_localize(None)

    result.index.name = "Date"

    # 重複日時を除外
    result = result.loc[
        ~result.index.duplicated(keep="last")
    ]

    result = result.sort_index()

    if result.empty:
        raise ValueError(
            f"{provider_symbol}の正規化後データが空です。"
        )

    return result


# ============================================================
# 未確定ローソク足の判定
# ============================================================

def _interval_to_timedelta(interval: str) -> pd.Timedelta | None:
    """
    yfinanceのintervalを、おおよその時間幅へ変換する。

    月足は月ごとの日数が異なるため、未確定判定を行わない。
    """

    interval_map: dict[str, pd.Timedelta] = {
        "1m": pd.Timedelta(minutes=1),
        "2m": pd.Timedelta(minutes=2),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "60m": pd.Timedelta(hours=1),
        "90m": pd.Timedelta(minutes=90),
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "5d": pd.Timedelta(days=5),
        "1wk": pd.Timedelta(weeks=1),
    }

    return interval_map.get(interval.lower().strip())


def _exclude_incomplete_last_row(
    frame: pd.DataFrame,
    interval: str,
) -> pd.DataFrame:
    """
    現在時刻から見て未確定と考えられる最後の行だけを除外する。

    単純に常時最終行を削除すると、過去期間の確定済みデータまで
    削除するため、終了時刻を使って判定する。
    """

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


# ============================================================
# Yahoo Finance取得
# ============================================================

def _download_prices(
    provider_symbol: str,
    period: str,
    interval: str,
    auto_adjust: bool,
    timeout: int,
) -> pd.DataFrame:
    """
    yfinance.download()を呼び出す。
    """

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
    """
    指定銘柄の価格データを取得する。

    Parameters
    ----------
    symbol:
        アプリ側の銘柄コード。
        例: AAPL.US、7203.JP

    period:
        yfinanceの取得期間。
        例: 1mo、3mo、6mo、1y、2y、5y、max

    interval:
        yfinanceの時間足。
        例: 1m、5m、1h、1d、1wk、1mo

    exclude_incomplete:
        Trueの場合、未確定と判断できる最終行を除外する。

    auto_adjust:
        Trueの場合、株式分割・配当を考慮した調整価格を使用する。

    retries:
        最大試行回数。

    retry_delay:
        再試行までの待機秒数。

    timeout:
        yfinanceへのリクエストタイムアウト秒数。

    Returns
    -------
    InstrumentData
        元の銘柄コード、データ提供元のコード、価格DataFrame。
    """

    original_symbol = symbol.strip().upper()
    provider_symbol = normalize_yahoo_symbol(original_symbol)

    if retries < 1:
        raise ValueError("retriesは1以上で指定してください。")

    if retry_delay < 0:
        raise ValueError("retry_delayは0以上で指定してください。")

    if timeout <= 0:
        raise ValueError("timeoutは1以上で指定してください。")

    last_exception: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "価格データを取得します: symbol=%s, provider_symbol=%s, "
                "period=%s, interval=%s, attempt=%d/%d",
                original_symbol,
                provider_symbol,
                period,
                interval,
                attempt,
                retries,
            )

            raw_prices = _download_prices(
                provider_symbol=provider_symbol,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                timeout=timeout,
            )

            prices = _normalize_price_frame(
                frame=raw_prices,
                provider_symbol=provider_symbol,
            )

            if exclude_incomplete:
                prices = _exclude_incomplete_last_row(
                    frame=prices,
                    interval=interval,
                )

            if prices.empty:
                raise ValueError(
                    f"{provider_symbol}は未確定データ除外後に"
                    "価格データが空になりました。"
                )

            logger.info(
                "価格データの取得に成功しました: symbol=%s, rows=%d",
                original_symbol,
                len(prices),
            )

            return InstrumentData(
                symbol=original_symbol,
                provider_symbol=provider_symbol,
                prices=prices,
            )

        except Exception as exc:
            last_exception = exc

            logger.exception(
                "価格データ取得に失敗しました: "
                "symbol=%s, provider_symbol=%s, attempt=%d/%d",
                original_symbol,
                provider_symbol,
                attempt,
                retries,
            )

            if attempt < retries:
                time.sleep(retry_delay * attempt)

    assert last_exception is not None

    raise RuntimeError(
        f"{original_symbol}の価格データ取得に失敗しました。"
        f"Yahoo Financeコード={provider_symbol}, "
        f"period={period}, interval={interval}, "
        f"原因=[{type(last_exception).__name__}]: "
        f"{last_exception!r}"
    ) from last_exception


# ============================================================
# DataFrameだけ必要な場合の簡易関数
# ============================================================

def get_price_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    exclude_incomplete: bool = True,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    InstrumentDataではなく価格DataFrameだけを返す。
    """

    instrument = get_instrument(
        symbol=symbol,
        period=period,
        interval=interval,
        exclude_incomplete=exclude_incomplete,
        auto_adjust=auto_adjust,
    )

    return instrument.prices.copy()
