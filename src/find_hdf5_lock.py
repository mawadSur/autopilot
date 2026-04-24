import pathlib
import psutil

path = pathlib.Path('market_data_store.h5').absolute()
print('path', path)
for proc in psutil.process_iter(['pid', 'name']):
    try:
        for f in proc.open_files():
            if pathlib.Path(f.path).samefile(path):
                print(proc.pid, proc.name(), f.path)
                raise SystemExit
    except Exception:
        continue
print('no lock owner')
