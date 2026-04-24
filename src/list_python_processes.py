import psutil
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    if proc.info['name'] and proc.info['name'].lower().startswith('python'):
        print(proc.info['pid'], proc.info['name'], proc.info['cmdline'])
