import tkinter as tk
from tkinter import ttk
from config import Config
import ast
import os
import threading
import logging

class LogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END)

def run_ui(run_backtest_func, run_live_trading_func, clear_log_file, open_log_file):
    window = tk.Tk()
    window.title("Wavy Tunnel Bot")
    window.geometry("800x600")

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

        mode_window.deiconify()
        config_window.withdraw()

    def open_config_window():
        mode_window.withdraw()
        config_window.deiconify()

    def run_backtest():
        log_text.configure(state='normal')
        log_text.delete(1.0, tk.END)
        log_text.configure(state='disabled')
        clear_log_file()
        log_text.insert(tk.END, "Backtesting running...\n")
        threading.Thread(target=run_backtest_thread).start()

    def run_backtest_thread():
        run_backtest_func()
        log_text.insert(tk.END, "Backtesting completed.\n")
        open_log_file()

    def run_live_trading():
        log_text.configure(state='normal')
        log_text.delete(1.0, tk.END)
        log_text.configure(state='disabled')
        clear_log_file()
        log_text.insert(tk.END, "Live trading running...\n")
        threading.Thread(target=run_live_trading_thread).start()

    def run_live_trading_thread():
        run_live_trading_func()
        log_text.insert(tk.END, "Live trading completed.\n")
        open_log_file()

    def clear_log():
        clear_log_file()
        log_clear_label.config(text="Log file cleared.")

    def open_log():
        open_log_file()

    # Apply a theme
    style = ttk.Style()
    style.theme_use('clam')

    # Main Mode Selection Window
    mode_window = window
    mode_window.title("Wavy Tunnel Bot")
    mode_window.geometry("800x600")

    mode_label = ttk.Label(mode_window, text="Select Mode", font=("Arial", 16))
    mode_label.pack(pady=20)

    backtest_button = ttk.Button(mode_window, text="Run Backtesting", command=run_backtest)
    backtest_button.pack(pady=10)

    live_trading_button = ttk.Button(mode_window, text="Run Live Trading", command=run_live_trading)
    live_trading_button.pack(pady=10)

    config_button = ttk.Button(mode_window, text="Configuration Settings", command=open_config_window)
    config_button.pack(pady=10)

    clear_log_button = ttk.Button(mode_window, text="Clear Log File", command=clear_log)
    clear_log_button.pack(pady=10)

    log_clear_label = ttk.Label(mode_window, text="")
    log_clear_label.pack(pady=10)

    open_log_button = ttk.Button(mode_window, text="Open Log File", command=open_log)
    open_log_button.pack(pady=10)

    # Terminal-like window for logs
    log_frame = ttk.Frame(mode_window)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    log_text = tk.Text(log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    log_scrollbar = ttk.Scrollbar(log_frame, command=log_text.yview)
    log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    log_text.config(yscrollcommand=log_scrollbar.set)

    # Add the log handler
    handler = LogHandler(log_text)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Configuration Settings Window
    config_window = tk.Toplevel(window)
    config_window.title("Configuration Settings")
    config_window.geometry("800x600")
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
