from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from cfevals.benchmarks.base import TimeSeriesBenchmark, TimeSeriesDataset, TimeSeriesPoint
from cfevals.context import ContextEvent

_REQUIRED_COLUMNS = ["Date", "Stock_symbol", "Publisher", "Article_title", "Lsa_summary"]


@dataclass
class FNSPIDNewsVolumeBenchmark(TimeSeriesBenchmark):
    dataset_name: str = "sabareesh88/FNSPID_nasdaq"
    split: str = "train"
    symbol: str = "AAPL"
    max_rows: int = 50_000
    min_points: int = 120
    start_date: str | None = None
    end_date: str | None = None
    cache_dir: str | None = None
    streaming: bool = True  # Deprecated: retained for backward-compatible registry args.
    max_scan_rows_multiplier: int = 10

    def load(self) -> TimeSeriesDataset:
        _validate_config(self)
        cache_dir = self.cache_dir or os.environ.get("CFEVALS_CACHE")

        symbol = self.symbol.upper()
        start_boundary = _parse_boundary(self.start_date) if self.start_date else None
        end_boundary = _parse_boundary(self.end_date) if self.end_date else None
        day_counts: dict[datetime, float] = {}
        context_events: list[ContextEvent] = []
        matched = 0
        scan_cap = self.max_rows * self.max_scan_rows_multiplier
        scanned = 0
        scanned_shards = 0
        for shard in _list_parquet_files(self.dataset_name, self.split):
            local_path = _download_parquet_file(self.dataset_name, shard, cache_dir)
            scanned_shards += 1
            for row in _iter_parquet_rows(local_path):
                if scanned >= scan_cap or matched >= self.max_rows:
                    break
                scanned += 1
                if str(row.get("Stock_symbol") or "").upper() != symbol:
                    continue

                timestamp = _parse_timestamp(row.get("Date"))
                if timestamp is None:
                    continue
                if start_boundary and timestamp < start_boundary:
                    continue
                if end_boundary and timestamp > end_boundary:
                    continue

                text = _compose_text(row)
                if not text:
                    continue

                day = datetime(timestamp.year, timestamp.month, timestamp.day)
                day_counts[day] = day_counts.get(day, 0.0) + 1.0
                context_events.append(
                    ContextEvent(
                        event_time=timestamp,
                        available_at=timestamp,
                        text=text,
                        modality="text",
                        source=str(row.get("Publisher") or "fnspid"),
                        metadata={"symbol": symbol},
                    )
                )
                matched += 1
            if scanned >= scan_cap or matched >= self.max_rows:
                break

        if not day_counts:
            raise ValueError(
                f"FNSPID benchmark produced no usable rows for symbol {symbol!r}. "
                "Try a different symbol/date range or increase max_rows/max_scan_rows_multiplier. "
                f"Scanned rows={scanned}, matched rows={matched}, scanned_shards={scanned_shards}."
            )

        first_day = min(day_counts.keys())
        last_day = max(day_counts.keys())
        index = pd.date_range(start=first_day, end=last_day, freq="D")
        points = [
            TimeSeriesPoint(
                timestamp=ts.to_pydatetime(),
                value=float(day_counts.get(ts.to_pydatetime(), 0.0)),
                features={"weekday": float(ts.weekday())},
            )
            for ts in index
        ]

        if len(points) < self.min_points:
            raise ValueError(
                f"FNSPID benchmark has only {len(points)} points for symbol {symbol!r}, "
                f"need at least {self.min_points}. Increase max_rows/max_scan_rows_multiplier or relax filters. "
                f"Scanned rows={scanned}, matched rows={matched}, scanned_shards={scanned_shards}."
            )

        metadata: dict[str, Any] = {
            "dataset": self.dataset_name,
            "split": self.split,
            "symbol": symbol,
            "matched_rows": matched,
            "scanned_rows": scanned,
            "scanned_shards": scanned_shards,
            "context_event_count": len(context_events),
            "task": "forecast_daily_news_volume",
        }
        return TimeSeriesDataset(
            points=points,
            context_events=context_events,
            frequency="D",
            metadata=metadata,
        )


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(None).to_pydatetime()


def _parse_boundary(value: str) -> datetime:
    parsed = pd.to_datetime(value, utc=True, errors="raise")
    return parsed.tz_convert(None).to_pydatetime()


def _compose_text(row: dict[str, Any]) -> str:
    title = str(row.get("Article_title") or "").strip()
    summary = str(row.get("Lsa_summary") or "").strip()
    parts = [part for part in [title, summary] if part and part.upper() != "N/A"]
    return " | ".join(parts)


def _validate_config(config: FNSPIDNewsVolumeBenchmark) -> None:
    if not config.dataset_name:
        raise ValueError("dataset_name must be set for FNSPID benchmark")
    if not config.split:
        raise ValueError("split must be set for FNSPID benchmark")
    if not config.symbol:
        raise ValueError("symbol must be set for FNSPID benchmark")
    if config.max_rows <= 0:
        raise ValueError("max_rows must be > 0 for FNSPID benchmark")
    if config.min_points <= 0:
        raise ValueError("min_points must be > 0 for FNSPID benchmark")
    if config.max_scan_rows_multiplier <= 0:
        raise ValueError("max_scan_rows_multiplier must be > 0 for FNSPID benchmark")
    if config.start_date and config.end_date:
        start = _parse_boundary(config.start_date)
        end = _parse_boundary(config.end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date for FNSPID benchmark")


def _list_parquet_files(dataset_name: str, split: str) -> list[str]:
    try:
        files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to list files for FNSPID dataset {dataset_name!r}. "
            "Check dataset availability/network credentials."
        ) from exc
    prefix = f"data/{split}-"
    shards = sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))
    if shards:
        return shards
    split_names = sorted({path.split("/", 1)[1].split("-", 1)[0] for path in files if path.startswith("data/")})
    raise ValueError(
        f"FNSPID dataset {dataset_name!r} does not contain split {split!r}. "
        f"Available splits inferred from parquet files: {split_names}"
    )


def _download_parquet_file(dataset_name: str, path: str, cache_dir: str | None) -> str:
    try:
        return hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=path,
            cache_dir=cache_dir,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download parquet shard {path!r} from dataset {dataset_name!r}. "
            "Check network/cache settings."
        ) from exc


def _iter_parquet_rows(path: str):
    try:
        frame = pd.read_parquet(path, columns=_REQUIRED_COLUMNS)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to read FNSPID parquet shard {path!r} with required columns {_REQUIRED_COLUMNS}."
        ) from exc

    missing = sorted(set(_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Parquet shard {path!r} is missing required columns: {missing}")

    normalized = frame.fillna("")
    for row in normalized.itertuples(index=False):
        yield {
            "Date": row.Date,
            "Stock_symbol": row.Stock_symbol,
            "Publisher": row.Publisher,
            "Article_title": row.Article_title,
            "Lsa_summary": row.Lsa_summary,
        }


# TODO(P0): Add point-in-time market data joins (prices/fundamentals) with explicit
# publication timestamps to forecast economically meaningful targets (returns/volatility).
# TODO(P1): Add panel mode (many symbols) with grouped/purged CV to prevent cross-asset leakage.
# TODO(P2): Add article deduplication and semantic clustering to reduce redundant context volume.
