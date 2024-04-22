import MetaTrader5 as mt5

def connect(login, password, server, path):
    if not mt5.initialize(path=path, login=login, password=password, server=server):
        print("initialize() failed, error code =", mt5.last_error())
        return False
    return True

def disconnect():
    mt5.shutdown()

def check_connection():
    return mt5.terminal_info() is not None
