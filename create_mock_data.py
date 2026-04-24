import h5py
import numpy as np
import os

def create_mock_hdf5(path="test_market_data.h5"):
    if os.path.exists(path):
        os.remove(path)
    
    with h5py.File(path, 'w') as f:
        grp = f.create_group("ETHUSDT/1m/candles")
        
        # Create dummy data
        n_rows = 2000
        data = np.zeros(n_rows, dtype=[
            ('timestamp', 'f8'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('volume', 'f8'),
            ('close_time', 'f8'),
            ('trades', 'f8'),
            ('logged_at', 'f8')
        ])
        
        data['timestamp'] = np.arange(n_rows)
        data['open'] = 2000.0 + np.random.randn(n_rows)
        data['high'] = data['open'] + 10.0
        data['low'] = data['open'] - 10.0
        data['close'] = data['open'] + 2.0
        data['volume'] = 100.0
        
        dset = grp.create_dataset("candles", data=data)
        dset.attrs["columns"] = list(data.dtype.names)
    
    print(f"✓ Created mock HDF5 at {path}")

if __name__ == "__main__":
    create_mock_hdf5()
