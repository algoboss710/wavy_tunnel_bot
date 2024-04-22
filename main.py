import signal
from scheduler import setup_schedule, run_scheduled_tasks
from utils.error_handling import handle_error, critical_error

def signal_handler(signum, frame):
    critical_error("Signal received, shutting down", f"Signal handler triggered with signal: {signum}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        setup_schedule()  # Set up the scheduler
        run_scheduled_tasks()  # Start the scheduling loop
    except Exception as e:
        handle_error(e, "Failed to start the scheduling loop")
