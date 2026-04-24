import numpy as np
from pathlib import Path
path = Path('labeled_chunks/batch_0.npz')
with np.load(path) as data:
    print('keys', list(data.keys()))
    for k in data.keys():
        arr = data[k]
        print(k, arr.shape, arr.dtype)
        if arr.ndim <= 2:
            print(arr[:2])
        else:
            print('ndim', arr.ndim, 'shape', arr.shape)
