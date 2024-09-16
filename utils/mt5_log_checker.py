import os
import time
import logging
import threading
import re
from glob import glob

log_monitoring_thread = None
log_monitoring_active = False

def find_latest_log_file(log_directory):
    log_files = glob(os.path.join(log_directory, '*.log'))
    if not log_files:
        return None
    latest_log_file = max(log_files, key=os.path.getmtime)
    return latest_log_file

def monitor_logs(log_directory, check_interval=1):  # Reduced interval to 1 second
    latest_log_file = find_latest_log_file(log_directory)
    if not latest_log_file:
        logging.warning("No log files found in the specified directory.")
        return

    with open(latest_log_file, 'r') as log_file:
        log_file.seek(0, os.SEEK_END)  # Move to the end of the file

        while log_monitoring_active:
            line = log_file.readline()
            if not line:
                time.sleep(check_interval)
                continue

            # Check for specific error patterns
            if "order_send" in line.lower() or "error" in line.lower():
                logging.error(f"MT5 Log Alert: {line.strip()}")
            elif re.search(r'\bTRADE_RETCODE\b', line):
                logging.warning(f"MT5 Trade Warning: {line.strip()}")

def start_log_checking(log_directory="C:\\Users\\16198\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\logs"):
    #C:\\Users\16198\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\logs
    #C:\\Users\Owner\\AppData\\Roaming\\MetaQuotes\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\logs
    global log_monitoring_thread, log_monitoring_active
    log_monitoring_active = True
    log_monitoring_thread = threading.Thread(target=monitor_logs, args=(log_directory,))
    log_monitoring_thread.daemon = True  # Make the thread a daemon thread
    log_monitoring_thread.start()
    logging.info("Started MT5 log monitoring.")

def stop_log_checking():
    global log_monitoring_active
    log_monitoring_active = False
    if log_monitoring_thread:
        log_monitoring_thread.join()
    logging.info("Stopped MT5 log monitoring.")
    logging.shutdown()  # Ensure logging is properly closed

