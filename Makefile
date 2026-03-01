reinstall:
\tpip uninstall -y protobuf tensorflow tensorflow-cpu
\tpip install -U pip
\tpip install protobuf==5.28.3
\tpip install -r requirements.txt
