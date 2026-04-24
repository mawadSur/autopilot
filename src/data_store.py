#!/usr/bin/env python3
"""
HDF5 Data Store - Consolidated storage for market data.

Replaces scattered .npz files with organized HDF5 storage:
- Single file for all training data
- Fast queries by symbol/time range
- Built-in compression
- Metadata storage
- Memory efficient
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

os.environ['H5PY_DISABLE_FILE_LOCKING'] = '1'
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')


class HDF5DataStore:
    """
    HDF5-based storage for processed market data.

    Structure:
    /symbol/interval/data (dataset)
    /symbol/interval/metadata (attributes)
    """

    def __init__(self, store_path: str = "market_data_store.h5"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(exist_ok=True)

    def store_chunk(
        self,
        symbol: str,
        interval: str,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store a data chunk with optional metadata."""
        with h5py.File(self.store_path, 'a') as f:
            # Create or replace group path
            group_path = f"{symbol}/{interval}"
            if group_path in f:
                del f[group_path]
            grp = f.create_group(group_path)

            if isinstance(data, dict):
                for dataset_name, dataset_value in data.items():
                    grp.create_dataset(
                        dataset_name,
                        data=dataset_value,
                        compression='gzip',
                        compression_opts=6,
                        chunks=True,
                    )
            else:
                grp.create_dataset(
                    'data',
                    data=data,
                    compression='gzip',
                    compression_opts=6,
                    chunks=True,
                )

            # Store metadata as attributes
            if metadata:
                for key, value in metadata.items():
                    grp.attrs[key] = value

    def load_chunk(self, symbol: str, interval: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Load data chunk and metadata."""
        with h5py.File(self.store_path, 'r') as f:
            grp = f[f"{symbol}/{interval}"]
            data = {name: grp[name][:] for name in grp.keys()}

            # Load metadata
            metadata = dict(grp.attrs)

            return data, metadata

    def list_symbols(self) -> List[str]:
        """List all stored symbols."""
        if not self.store_path.exists():
            return []

        with h5py.File(self.store_path, 'r') as f:
            return list(f.keys())

    def list_intervals(self, symbol: str) -> List[str]:
        """List intervals for a symbol."""
        if not self.store_path.exists():
            return []

        with h5py.File(self.store_path, 'r') as f:
            if symbol not in f:
                return []
            return list(f[symbol].keys())

    def get_info(self) -> Dict:
        """Get store information."""
        if not self.store_path.exists():
            return {"file_exists": False}

        info = {"file_exists": True, "file_size_mb": self.store_path.stat().st_size / (1024**2)}

        with h5py.File(self.store_path, 'r') as f:
            symbols = {}
            total_chunks = 0

            for symbol in f.keys():
                intervals = {}
                for interval in f[symbol].keys():
                    grp = f[f"{symbol}/{interval}"]
                    data_shape = grp['data'].shape
                    intervals[interval] = {
                        "shape": data_shape,
                        "dtype": str(grp['data'].dtype),
                        "metadata": dict(grp.attrs)
                    }
                    total_chunks += 1

                symbols[symbol] = intervals

            info.update({
                "symbols": symbols,
                "total_chunks": total_chunks
            })

        return info


class DataMigration:
    """Migrate existing .npz files to HDF5 store."""

    def __init__(self, npz_dir: str = "labeled_chunks",
                 hdf5_path: str = "market_data_store.h5"):
        self.npz_dir = Path(npz_dir)
        self.store = HDF5DataStore(hdf5_path)

    def migrate_all_chunks(self) -> Dict[str, int]:
        """Migrate all .npz files to HDF5."""
        if not self.npz_dir.exists():
            print(f"Directory {self.npz_dir} not found")
            return {}

        npz_files = list(self.npz_dir.glob("batch_*.npz"))
        print(f"Found {len(npz_files)} .npz files to migrate")

        migrated = 0
        skipped = 0

        for npz_file in tqdm(npz_files, desc="Migrating chunks"):
            try:
                batch_num = int(npz_file.stem.split('_')[1])
                symbol = "ETHUSDT"
                interval = "1m"
                group_path = f"{symbol}/{interval}_batch_{batch_num}"
                
                with h5py.File(self.store.store_path, 'r') as f:
                    if group_path in f:
                        print(f"Skipping batch_{batch_num}: already exists in HDF5")
                        skipped += 1
                        continue
                
                data = np.load(npz_file)

                # Assume structure: X, y arrays
                if 'X' in data and 'y' in data:
                    X, y = data['X'], data['y']

                    # Extract batch number for metadata
                    batch_num = int(npz_file.stem.split('_')[1])

                    symbol = "ETHUSDT"  # Default, can be made configurable
                    interval = "1m"

                    metadata = {
                        "batch_number": batch_num,
                        "original_file": str(npz_file.name),
                        "X_shape": X.shape,
                        "y_shape": y.shape,
                        "feature_count": X.shape[-1] if len(X.shape) > 1 else 1,
                        "migrated_at": pd.Timestamp.now().isoformat(),
                    }

                    self.store.store_chunk(
                        symbol,
                        f"{interval}_batch_{batch_num}",
                        {"X": X, "y": y},
                        metadata,
                    )
                    migrated += 1

                else:
                    print(f"Skipping {npz_file}: unexpected structure {list(data.keys())}")
                    skipped += 1

            except Exception as e:
                print(f"Error migrating {npz_file}: {e}")
                skipped += 1

        return {"migrated": migrated, "skipped": skipped}


def main():
    """Command-line interface for data store operations."""
    import argparse

    parser = argparse.ArgumentParser(description="HDF5 Data Store Manager")
    parser.add_argument("action", choices=["migrate", "info", "list"],
                       help="Action to perform")
    parser.add_argument("--npz-dir", default="labeled_chunks",
                       help="Directory containing .npz files")
    parser.add_argument("--store-path", default="market_data_store.h5",
                       help="HDF5 store path")

    args = parser.parse_args()

    store = HDF5DataStore(args.store_path)

    if args.action == "migrate":
        migrator = DataMigration(args.npz_dir, args.store_path)
        result = migrator.migrate_all_chunks()
        print(f"Migration complete: {result['migrated']} migrated, {result['skipped']} skipped")

    elif args.action == "info":
        info = store.get_info()
        print("Store Information:")
        print(f"  File exists: {info['file_exists']}")
        if info['file_exists']:
                    print(f"  File size (MB): {info['file_size_mb']:.2f}")

if __name__ == "__main__":
    main()