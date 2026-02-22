from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from utils import compute_features, normalize_headers

LabelFn = Callable[[pd.DataFrame], Sequence]
SourceLike = Union[str, Path]
SourceListLike = Sequence[SourceLike]


class KlineStreamer:
    """Unified kline streamer.

    Supports:
    - Directory of CSVs
    - Single CSV
    - In-memory iterable of dicts (live feed)

    Applies `compute_features` once per chunk and yields tuples:
        (window: np.ndarray [T,F], label_or_return, metadata_dict)
    where metadata includes basic OHLC and optional timestamp.
    """

    def __init__(
        self,
        source: Union[SourceLike, SourceListLike, Iterable[dict]],
        *,
        window_size: int,
        feature_cols: Optional[List[str]] = None,
        chunksize: int = 300_000,
        overlap_rows: Optional[int] = None,
        label_fn: Optional[LabelFn] = None,
    ) -> None:
        self.source = source
        self.window = int(window_size)
        self.feature_cols = None if feature_cols is None else list(feature_cols)
        self.chunksize = int(chunksize)
        self.overlap = int(overlap_rows) if overlap_rows is not None else max(self.window + 5, 2000)
        self.label_fn = label_fn

    # ----------------------------
    # Public iterator
    # ----------------------------
    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[object], dict]]:
        buf: Deque[np.ndarray] = deque(maxlen=self.window)
        for df in self._iter_feature_frames():
            labels = None
            if self.label_fn is not None:
                labels = list(self.label_fn(df))
                if len(labels) != len(df):
                    raise ValueError("label_fn must return one label per row")

            if self.feature_cols is None:
                num_df = df.select_dtypes(include=[np.number])
                if num_df.shape[1] == 0:
                    raise ValueError("No numeric columns available for features.")
                feat_mat = num_df.to_numpy(dtype=np.float32, copy=False)
            else:
                feat_mat = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            for idx in range(len(df)):
                buf.append(feat_mat[idx])
                if len(buf) < self.window:
                    continue
                window = np.stack(list(buf), axis=0)
                label = labels[idx] if labels is not None else None
                row = df.iloc[idx]
                meta = {
                    "open": float(row.get("open", np.nan)),
                    "high": float(row.get("high", np.nan)),
                    "low": float(row.get("low", np.nan)),
                    "close": float(row.get("close", np.nan)),
                    "volume": float(row.get("volume", np.nan)),
                    "atr": float(row.get("atr_14", np.nan)) if "atr_14" in df.columns else None,
                    "timestamp": row.get("timestamp", None),
                }
                yield window, label, meta

    # ----------------------------
    # Frame iteration
    # ----------------------------
    def _iter_feature_frames(self) -> Iterator[pd.DataFrame]:
        yielded = 0
        num_files = 0

        files: Optional[List[Path]] = None

        # Case A: directory or single file
        if isinstance(self.source, (str, Path)):
            files = self._list_csvs(Path(self.source))

        # Case B: explicit list/tuple of CSV paths
        elif isinstance(self.source, Sequence) and not isinstance(self.source, (bytes, bytearray)):
            # treat as list of CSV paths if elements look like paths (str/Path)
            if len(self.source) > 0 and all(isinstance(x, (str, Path)) for x in self.source):
                files = [Path(x) for x in self.source]
            else:
                files = None

        if files is not None:
            num_files = len(files)
            tail_raw: Optional[pd.DataFrame] = None
            for path in files:
                try:
                    for chunk in pd.read_csv(path, chunksize=self.chunksize):
                        chunk = normalize_headers(chunk)
                        raw = pd.concat([tail_raw, chunk], ignore_index=True) if tail_raw is not None else chunk
                        feat = compute_features(raw)
                        if len(feat) > self.overlap:
                            out = feat.iloc[self.overlap:].reset_index(drop=True)
                            tail_raw = raw.iloc[-self.overlap:].reset_index(drop=True)
                            yield out
                            yielded += 1
                        else:
                            tail_raw = raw.reset_index(drop=True)
                except Exception as e:
                    print(f"[KlineStreamer] ERROR reading {path}: {e}")
                    continue
        else:
            # live iterable of dicts
            history: List[dict[str, Any]] = []
            for bar in self.source:
                try:
                    if not isinstance(bar, dict):
                        continue
                    history.append(bar)
                    df = pd.DataFrame(history)
                    if len(df) < self.window:
                        continue
                    feat = compute_features(df)
                    yield feat.tail(self.window).reset_index(drop=True)
                    yielded += 1
                except Exception as e:
                    print(f"[KlineStreamer] ERROR in live stream: {e}")
                    continue
        if yielded == 0:
            print(
                f"[KlineStreamer] yielded=0. files={num_files} window={self.window} "
                f"chunksize={self.chunksize} feature_cols={self.feature_cols}"
            )

    @staticmethod
    def _list_csvs(path: Path) -> List[Path]:
        if path.is_dir():
            files = sorted(path.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files found in directory: {path}")
            return files
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file or directory, got: {path}")
        return [path]
