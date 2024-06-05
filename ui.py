# import tkinter as tk
# from tkinter import ttk
# from config import Config
# import ast

# def run_ui(run_backtest_func, run_live_trading_func):
#     window = tk.Tk()
#     window.title("Wavy Tunnel Bot")
#     window.geometry("600x400")
#     window.withdraw()

#     def update_config():
#         config_values = {}
#         for attr, entry in config_entries.items():
#             value = entry.get()
#             if value != "":
#                 try:
#                     config_values[attr] = ast.literal_eval(value)
#                 except (ValueError, SyntaxError):
#                     config_values[attr] = value

#         for attr, value in config_values.items():
#             setattr(Config, attr, value)

#         config_window.destroy()
#         mode_window.deiconify()

#     def open_config_window():
#         mode_window.withdraw()
#         config_window.deiconify()

#     def run_backtest():
#         mode_window.destroy()
#         window.destroy()
#         run_backtest_func()

#     def run_live_trading():
#         mode_window.destroy()
#         window.destroy()
#         run_live_trading_func()

#     # Mode Selection Window
#     mode_window = tk.Toplevel(window)
#     mode_window.title("Select Mode")
#     mode_window.geometry("600x400")
#     mode_window.withdraw()

#     mode_label = ttk.Label(mode_window, text="Select Mode", font=("Arial", 16))
#     mode_label.pack(pady=20)

#     backtest_button = ttk.Button(mode_window, text="Run Backtesting", command=run_backtest)
#     backtest_button.pack(pady=10)

#     live_trading_button = ttk.Button(mode_window, text="Run Live Trading", command=run_live_trading)
#     live_trading_button.pack(pady=10)

#     config_button = ttk.Button(mode_window, text="Configuration Settings", command=open_config_window)
#     config_button.pack(pady=10)

#     # Configuration Settings Window
#     config_window = tk.Toplevel(window)
#     config_window.title("Configuration Settings")
#     config_window.geometry("600x400")

#     config_label = ttk.Label(config_window, text="Configuration Settings", font=("Arial", 16))
#     config_label.pack(pady=20)

#     config_canvas = tk.Canvas(config_window)
#     config_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#     scrollbar = ttk.Scrollbar(config_window, orient=tk.VERTICAL, command=config_canvas.yview)
#     scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

#     config_canvas.configure(yscrollcommand=scrollbar.set)
#     config_canvas.bind('<Configure>', lambda e: config_canvas.configure(scrollregion=config_canvas.bbox('all')))

#     config_frame = ttk.Frame(config_canvas)
#     config_canvas.create_window((0, 0), window=config_frame, anchor='nw')

#     config_entries = {}
#     for attr, value in Config.__dict__.items():
#         if not callable(value) and not attr.startswith("__"):
#             attr_label = ttk.Label(config_frame, text=attr)
#             attr_label.grid(row=len(config_entries), column=0, padx=5, pady=5, sticky="e")

#             attr_entry = ttk.Entry(config_frame, width=40)
#             attr_entry.insert(0, str(value))
#             attr_entry.grid(row=len(config_entries), column=1, padx=5, pady=5)

#             config_entries[attr] = attr_entry

#     update_button = ttk.Button(config_window, text="Update Settings", command=update_config)
#     update_button.pack(pady=10)

#     config_window.deiconify()

#     window.mainloop()

import tkinter as tk
from tkinter import ttk
from config import Config
import ast

def run_ui(run_backtest_func, run_live_trading_func):
    window = tk.Tk()
    window.title("Wavy Tunnel Bot")
    window.geometry("600x400")
    window.withdraw()

    def update_config():
        config_values = {}
        for attr, entry in config_entries.items():
            value = entry.get()
            if value != "":
                try:
                    config_values[attr] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    config_values[attr] = value

        for attr, value in config_values.items():
            setattr(Config, attr, value)

        config_window.destroy()
        mode_window.deiconify()

    def open_config_window():
        mode_window.withdraw()
        config_window.deiconify()

    def run_backtest():
        mode_window.destroy()
        window.destroy()
        run_backtest_func()

    def run_live_trading():
        mode_window.destroy()
        window.destroy()
        run_live_trading_func()

    def open_mode_window():
        welcome_window.destroy()
        mode_window.deiconify()

    # Apply a theme
    style = ttk.Style()
    style.theme_use('clam')

    # Welcome Screen Window
    welcome_window = tk.Toplevel(window)
    welcome_window.title("Welcome")
    welcome_window.geometry("600x400")

    welcome_label = ttk.Label(welcome_window, text="Welcome to the Wavy Tunnel Bot", font=("Arial", 20))
    welcome_label.pack(pady=50)

    proceed_button = ttk.Button(welcome_window, text="Proceed", command=open_mode_window)
    proceed_button.pack(pady=20)

    # Mode Selection Window
    mode_window = tk.Toplevel(window)
    mode_window.title("Select Mode")
    mode_window.geometry("600x400")
    mode_window.withdraw()

    mode_label = ttk.Label(mode_window, text="Select Mode", font=("Arial", 16))
    mode_label.pack(pady=20)

    backtest_button = ttk.Button(mode_window, text="Run Backtesting", command=run_backtest)
    backtest_button.pack(pady=10)

    live_trading_button = ttk.Button(mode_window, text="Run Live Trading", command=run_live_trading)
    live_trading_button.pack(pady=10)

    config_button = ttk.Button(mode_window, text="Configuration Settings", command=open_config_window)
    config_button.pack(pady=10)

    # Configuration Settings Window
    config_window = tk.Toplevel(window)
    config_window.title("Configuration Settings")
    config_window.geometry("600x400")
    config_window.withdraw()

    config_label = ttk.Label(config_window, text="Configuration Settings", font=("Arial", 16))
    config_label.pack(pady=20)

    config_canvas = tk.Canvas(config_window)
    config_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(config_window, orient=tk.VERTICAL, command=config_canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    config_canvas.configure(yscrollcommand=scrollbar.set)
    config_canvas.bind('<Configure>', lambda e: config_canvas.configure(scrollregion=config_canvas.bbox('all')))

    config_frame = ttk.Frame(config_canvas)
    config_canvas.create_window((0, 0), window=config_frame, anchor='nw')

    config_entries = {}
    for attr, value in Config.__dict__.items():
        if not callable(value) and not attr.startswith("__") and attr not in ["validate", "log_config"]:
            attr_label = ttk.Label(config_frame, text=attr)
            attr_label.grid(row=len(config_entries), column=0, padx=5, pady=5, sticky="e")

            attr_entry = ttk.Entry(config_frame, width=40)
            attr_entry.insert(0, str(value))
            attr_entry.grid(row=len(config_entries), column=1, padx=5, pady=5)

            config_entries[attr] = attr_entry

    update_button = ttk.Button(config_window, text="Update Settings", command=update_config)
    update_button.pack(pady=10)

    window.mainloop()
