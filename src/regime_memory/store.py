"""Regime vector store: k-NN lookup of historical embedded windows.

Two implementations live here:

* :class:`NaiveRegimeStore` — pure numpy, brute-force cosine similarity. This
  is the **primary** path because ``faiss-cpu`` is not in
  :mod:`requirements.txt`. For the regime sizes E4 expects (≤100k windows of
  ~144 floats each) brute force is sub-millisecond per query and saves us a
  hard build dependency.
* :class:`RegimeStore` — FAISS-backed (``IndexFlatIP`` over L2-normalized
  vectors so inner product equals cosine similarity). Constructor raises
  :class:`ImportError` if ``faiss`` is unavailable, with a hint to use
  :class:`NaiveRegimeStore`. This is the same correctness contract — the
  FAISS path is just a perf upgrade for callers who have FAISS installed.

Both stores normalize embeddings to L2 unit norm before storing so the
similarity score returned by :meth:`query` is bounded in ``[-1, 1]`` (cosine
similarity, ``1.0`` = identical direction, ``0.0`` = orthogonal, ``-1.0`` =
opposite). The factory :func:`make_regime_store` picks FAISS when available
and falls back to the numpy store otherwise.

Persistence
-----------
:class:`NaiveRegimeStore` saves to a single ``.npz`` file containing the
stacked embedding matrix plus a JSON-serialized list of metadata records.
The format is intentionally simple — one file, no side cars — so
``RegimeLookup`` consumers can copy regime stores around with ``cp`` and not
worry about index/metadata getting out of sync. :class:`RegimeStore` (FAISS)
also persists as a single ``.npz`` because writing the FAISS index to a
sibling file would create the exact synchronization hazard we're avoiding.
On load it rebuilds the FAISS index from the saved vectors — that costs O(N)
on store open but eliminates a class of bugs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from regime_memory.encoder import RegimeWindow

# --- FAISS optional import --------------------------------------------------

try:  # pragma: no cover - exercised via unittest.skipUnless gating
    import faiss  # type: ignore[import-not-found]

    _HAS_FAISS = True
except Exception:  # pragma: no cover - the no-faiss branch IS the test case
    faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False


# --- helpers ----------------------------------------------------------------

_EPS = 1e-12


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` rescaled to unit L2 norm.

    Works on 1-D and 2-D inputs (rows are normalized independently for 2-D).
    A row whose norm is below :data:`_EPS` is left untouched (returning all
    zeros) — the alternative would be NaN propagation, which silently breaks
    the cosine-similarity contract downstream.
    """

    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm < _EPS:
            return arr
        return arr / norm
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms < _EPS, 1.0, norms)
    out = arr / safe
    # Rows that were near-zero get zeroed out explicitly; their similarity
    # against any query is 0.0 which is the correct semantics.
    out[(norms < _EPS).reshape(-1)] = 0.0
    return out


def _window_metadata_dict(window: RegimeWindow) -> dict:
    """Serialize a :class:`RegimeWindow` as a JSON-safe dict.

    We serialize *everything except* the embedding (which lives in the npz
    matrix alongside) so on load we can reconstruct the original window.
    """

    return {
        "symbol": window.symbol,
        "window_end_utc": window.window_end_utc,
        "bars": window.bars,
        "metadata": dict(window.metadata),
    }


def _window_from_metadata(meta: dict, embedding: List[float]) -> RegimeWindow:
    return RegimeWindow(
        symbol=str(meta["symbol"]),
        window_end_utc=str(meta["window_end_utc"]),
        bars=int(meta["bars"]),
        embedding=list(embedding),
        metadata={str(k): float(v) for k, v in dict(meta.get("metadata", {})).items()},
    )


# --- naive (numpy) store ----------------------------------------------------


class NaiveRegimeStore:
    """Brute-force cosine k-NN store backed by a numpy matrix.

    For the sizes this project cares about (≤100k windows × 144 dims)
    ``np.dot`` on a contiguous (N, D) matrix returns all similarities in
    well under a millisecond on commodity hardware — using FAISS for that
    workload is a wash, and skipping it removes a build dependency.

    The store is append-only at the API level; callers who want to evict
    stale windows should rebuild a new store from the surviving subset.
    """

    def __init__(self, *, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim!r}")
        self.dim: int = int(dim)
        # Vectors are stored already-normalized so query() can do a plain
        # matmul against the normalized query vector.
        self._vectors: List[np.ndarray] = []
        self._windows: List[RegimeWindow] = []

    # -- writes --------------------------------------------------------------

    def add(self, window: RegimeWindow) -> None:
        """Append a single window. Embedding is L2-normalized in place.

        Raises
        ------
        ValueError
            If the embedding's length doesn't match :attr:`dim`.
        """

        if len(window.embedding) != self.dim:
            raise ValueError(
                "embedding dim mismatch: store expects "
                f"{self.dim}, got {len(window.embedding)} for window "
                f"{window.symbol!r} @ {window.window_end_utc!r}"
            )
        vec = _l2_normalize(np.asarray(window.embedding, dtype=np.float32))
        self._vectors.append(vec)
        self._windows.append(window)

    def add_many(self, windows: List[RegimeWindow]) -> None:
        for w in windows:
            self.add(w)

    # -- reads ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def query(
        self,
        embedding: List[float],
        k: int = 10,
    ) -> List[Tuple[RegimeWindow, float]]:
        """Return the top-``k`` (window, cosine_similarity) pairs.

        Returns ``[]`` if the store is empty. Similarities are clamped to
        ``[-1.0, 1.0]`` to absorb the tiny float drift that ``np.dot`` can
        produce on already-normalized vectors.
        """

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        if not self._vectors:
            return []
        if len(embedding) != self.dim:
            raise ValueError(
                f"query embedding dim mismatch: store expects {self.dim}, "
                f"got {len(embedding)}"
            )
        q = _l2_normalize(np.asarray(embedding, dtype=np.float32))
        mat = np.stack(self._vectors, axis=0)  # (N, D)
        sims = mat @ q  # (N,)
        sims = np.clip(sims, -1.0, 1.0)
        # ``argpartition`` is O(N) and good enough; we then sort the top-k
        # slice descending so ranking is deterministic.
        n = sims.shape[0]
        kk = min(k, n)
        if kk == n:
            top_idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, kk - 1)[:kk]
            top_idx = part[np.argsort(-sims[part])]
        return [(self._windows[int(i)], float(sims[int(i)])) for i in top_idx]

    # -- persistence ---------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the store to ``path`` as a single ``.npz``.

        Layout (npz members):
        * ``vectors`` — (N, D) float32, already L2-normalized
        * ``metadata_json`` — UTF-8 bytes of a JSON list of metadata records
        * ``dim`` — scalar int, for round-trip validation
        """

        path = Path(path)
        if self._vectors:
            mat = np.stack(self._vectors, axis=0).astype(np.float32, copy=False)
        else:
            mat = np.zeros((0, self.dim), dtype=np.float32)
        meta_records = [_window_metadata_dict(w) for w in self._windows]
        meta_bytes = json.dumps(meta_records).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=mat,
            metadata_json=np.asarray(meta_bytes),
            dim=np.int64(self.dim),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NaiveRegimeStore":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        dim = int(data["dim"].item())
        store = cls(dim=dim)
        meta_bytes = data["metadata_json"].tobytes()
        meta_records = json.loads(meta_bytes.decode("utf-8"))
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        if vectors.shape[0] != len(meta_records):
            raise ValueError(
                f"corrupt store at {path!s}: vectors={vectors.shape[0]} "
                f"vs metadata={len(meta_records)}"
            )
        for vec, meta in zip(vectors, meta_records):
            window = _window_from_metadata(meta, vec.astype(float).tolist())
            # Reuse add() so the dim guard fires on round-trip drift.
            store.add(window)
        return store


# --- FAISS-backed store -----------------------------------------------------


class RegimeStore:
    """FAISS-backed k-NN store. Falls back to :class:`NaiveRegimeStore` semantics.

    Constructor raises :class:`ImportError` if ``faiss-cpu`` is not installed,
    so callers should prefer :func:`make_regime_store` which transparently
    picks the right backend.

    The on-disk format is the **same** ``.npz`` as :class:`NaiveRegimeStore`
    so a store written by either backend is readable by either backend.
    Persisting the FAISS index directly would force callers to keep two files
    in sync, which the project doesn't need.
    """

    def __init__(
        self,
        *,
        dim: int,
        faiss_index_path: Optional[Path] = None,  # accepted for API symmetry
    ) -> None:
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is not installed; install it or use "
                "regime_memory.store.NaiveRegimeStore (or call "
                "regime_memory.store.make_regime_store(...) which picks "
                "automatically)"
            )
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim!r}")
        self.dim: int = int(dim)
        # IndexFlatIP over L2-normalized vectors == cosine similarity.
        self._index = faiss.IndexFlatIP(self.dim)  # type: ignore[union-attr]
        self._windows: List[RegimeWindow] = []
        # ``faiss_index_path`` is stored for symmetry with the original brief
        # but isn't used (we persist via .npz, not native FAISS files).
        self.faiss_index_path = faiss_index_path

    # -- writes --------------------------------------------------------------

    def add(self, window: RegimeWindow) -> None:
        if len(window.embedding) != self.dim:
            raise ValueError(
                "embedding dim mismatch: store expects "
                f"{self.dim}, got {len(window.embedding)} for window "
                f"{window.symbol!r} @ {window.window_end_utc!r}"
            )
        vec = _l2_normalize(np.asarray(window.embedding, dtype=np.float32))
        self._index.add(vec.reshape(1, -1))  # type: ignore[union-attr]
        self._windows.append(window)

    def add_many(self, windows: List[RegimeWindow]) -> None:
        if not windows:
            return
        mat_list: List[np.ndarray] = []
        for w in windows:
            if len(w.embedding) != self.dim:
                raise ValueError(
                    "embedding dim mismatch: store expects "
                    f"{self.dim}, got {len(w.embedding)} for window "
                    f"{w.symbol!r} @ {w.window_end_utc!r}"
                )
            mat_list.append(np.asarray(w.embedding, dtype=np.float32))
        mat = _l2_normalize(np.stack(mat_list, axis=0))
        self._index.add(mat)  # type: ignore[union-attr]
        self._windows.extend(windows)

    # -- reads ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def query(
        self,
        embedding: List[float],
        k: int = 10,
    ) -> List[Tuple[RegimeWindow, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        if not self._windows:
            return []
        if len(embedding) != self.dim:
            raise ValueError(
                f"query embedding dim mismatch: store expects {self.dim}, "
                f"got {len(embedding)}"
            )
        q = _l2_normalize(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        kk = min(k, len(self._windows))
        sims, idxs = self._index.search(q, kk)  # type: ignore[union-attr]
        sims = np.clip(sims[0], -1.0, 1.0)
        out: List[Tuple[RegimeWindow, float]] = []
        for sim, idx in zip(sims, idxs[0]):
            if int(idx) < 0:  # FAISS returns -1 for under-populated slots
                continue
            out.append((self._windows[int(idx)], float(sim)))
        return out

    # -- persistence ---------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        # Reconstruct the (N, D) matrix from FAISS storage. ``reconstruct_n``
        # is O(N) but the alternative (writing native FAISS index) loses the
        # single-file persistence guarantee.
        n = len(self._windows)
        if n > 0:
            mat = np.zeros((n, self.dim), dtype=np.float32)
            for i in range(n):
                mat[i] = self._index.reconstruct(i)  # type: ignore[union-attr]
        else:
            mat = np.zeros((0, self.dim), dtype=np.float32)
        meta_records = [_window_metadata_dict(w) for w in self._windows]
        meta_bytes = json.dumps(meta_records).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=mat,
            metadata_json=np.asarray(meta_bytes),
            dim=np.int64(self.dim),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RegimeStore":
        if not _HAS_FAISS:
            raise ImportError(
                "faiss-cpu is not installed; cannot load a RegimeStore. Use "
                "NaiveRegimeStore.load() — the on-disk format is identical."
            )
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        dim = int(data["dim"].item())
        store = cls(dim=dim)
        meta_bytes = data["metadata_json"].tobytes()
        meta_records = json.loads(meta_bytes.decode("utf-8"))
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        if vectors.shape[0] != len(meta_records):
            raise ValueError(
                f"corrupt store at {path!s}: vectors={vectors.shape[0]} "
                f"vs metadata={len(meta_records)}"
            )
        windows = [
            _window_from_metadata(meta, vec.astype(float).tolist())
            for vec, meta in zip(vectors, meta_records)
        ]
        store.add_many(windows)
        return store


# --- factory ----------------------------------------------------------------


def make_regime_store(
    *,
    dim: int,
    prefer_faiss: bool = True,
) -> Union[RegimeStore, NaiveRegimeStore]:
    """Pick the right backend for the current environment.

    Returns :class:`RegimeStore` when FAISS is importable and
    ``prefer_faiss`` is True; otherwise :class:`NaiveRegimeStore`. Existing
    callers can switch between backends transparently because both share the
    same public API.
    """

    if prefer_faiss and _HAS_FAISS:
        return RegimeStore(dim=dim)
    return NaiveRegimeStore(dim=dim)


# Keep dataclass `asdict` import alive if we later expose it for debugging;
# the import is also a hint to future maintainers that metadata is dict-like.
_ = asdict  # noqa: F841  — silence unused import warning


__all__ = [
    "NaiveRegimeStore",
    "RegimeStore",
    "make_regime_store",
]
