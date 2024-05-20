import MetaTrader5 as mt5
# import MetaTrader4 as mt4
import win32com.client
import time

def connect(login, password, server, path, mt_version):
    if mt_version == 5:
        if not mt5.initialize(path=path, login=login, password=password, server=server):
            print("initialize() failed for MT5, error code =", mt5.last_error())
            return False
        return True
    # elif mt_version == 4:
    #     try:
    #         mt4_client = mt4.MT4()
    #         mt4_client.Connect(server, login, password, "")
    #         return True
    #     except Exception as e:
    #         print("initialize() failed for MT4, error =", e)
    #         return False
    else:
        raise ValueError("Invalid MetaTrader version. Please specify 4 or 5.")

def disconnect(mt_version):
    if mt_version == 5:
        mt5.shutdown()
    # elif mt_version == 4:
    #     try:
    #         mt4_client = mt4.MT4()
    #         mt4_client.Disconnect()
    #     except Exception as e:
    #         print("disconnect() failed for MT4, error =", e)

def check_connection(mt_version):
    if mt_version == 5:
        return mt5.terminal_info() is not None
    # elif mt_version == 4:
    #     try:
    #         mt4_client = mt4.MT4()
    #         return mt4_client.IsConnected()
    #     except Exception as e:
    #         print("check_connection() failed for MT4, error =", e)
    #         return False
    # else:
    #     raise ValueError("Invalid MetaTrader version. Please specify 4 or 5.")

def initialize_mt5(mt5_path):
    # Initialize MetaTrader 5
    if not mt5.initialize(path=mt5_path):
        print("Failed to initialize MetaTrader 5 terminal.")
        mt5.shutdown()
        return False
    else:
        print("MetaTrader 5 terminal initialized successfully.")
        return True

def shutdown_mt5():
    # Shutdown MetaTrader 5
    mt5.shutdown()
    print("MetaTrader 5 connection gracefully shut down.")
