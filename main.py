import signal
from scheduler import setup_schedule, run_scheduled_tasks
from utils.error_handling import handle_error, critical_error
from strategy.trade_logic import calculate_position_size, entry_long, entry_short, exit_trade
from utils.data_validation import validate_data, sanitize_data
from config import Config
from metatrader.connection import initialize_mt5
from metatrader.trade_management import get_open_positions, should_exit_position

def signal_handler(signum, frame):
    critical_error("Signal received, shutting down", f"Signal handler triggered with signal: {signum}")

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        setup_schedule()
        
        if initialize_mt5():
            account_info = get_account_info()
            account_balance = account_info["balance"]

            for symbol in Config.SYMBOLS:
                data = get_historical_data(symbol, Config.MT5_TIMEFRAME, start_time, end_time)
                sanitized_data = sanitize_data(data)

                if validate_data(sanitized_data, TradeRequestSchema.schema()):
                    signal = generate_trade_signal(sanitized_data, period, deviation)
                    price = sanitized_data["close"].iloc[-1]
                    stop_loss_pips = 20
                    take_profit_pips = 40
                    deviation = 10
                    magic = 12345
                    comment = "Tunnel Strategy"

                    position_size = calculate_position_size(account_balance, Config.RISK_PER_TRADE, stop_loss_pips, Config.PIP_VALUE)

                    if signal == 'BUY':
                        sl = price - (stop_loss_pips * Config.PIP_VALUE)
                        tp = price + (take_profit_pips * Config.PIP_VALUE)
                        entry_long(symbol, position_size, price, sl, tp, deviation, magic, comment)
                    elif signal == 'SELL':
                        sl = price + (stop_loss_pips * Config.PIP_VALUE)
                        tp = price - (take_profit_pips * Config.PIP_VALUE)
                        entry_short(symbol, position_size, price, sl, tp, deviation, magic, comment)
                else:
                    logging.error("Invalid input data")

            open_positions = get_open_positions()
            for position in open_positions:
                if should_exit_position(position):
                    exit_trade(position.ticket)

        run_scheduled_tasks()

    except Exception as e:
        handle_error(e, "Failed to execute trading logic or validate/sanitize input data")

if __name__ == "__main__":
    main()