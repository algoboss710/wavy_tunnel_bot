import MetaTrader5 as mt5
from config import Config
import logging

def place_order(symbol, order_type, volume, price=None, sl=None, tp=None):
    try:
        order = mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order,
            "price": mt5.symbol_info_tick(symbol).ask if order_type == 'buy' else mt5.symbol_info_tick(symbol).bid,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 234000,
            "comment": "python script order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result.comment if result else 'Order failed'
    except Exception as e:
        return f'Order failed: {str(e)}'

def close_position(ticket):
    try:
        position = mt5.positions_get(ticket=ticket)
        if position:
            result = mt5.Close(ticket)
            return result.comment if result else 'Close failed'
        return 'Position not found'
    except Exception as e:
        return f'Close failed: {str(e)}'

def modify_order(ticket, sl=None, tp=None):
    try:
        result = mt5.order_check(ticket)
        if result and result.type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "ticket": ticket,
                "sl": sl,
                "tp": tp
            }
            result = mt5.order_send(request)
            return result.comment if result else 'Modify failed'
        return 'Order not found'
    except Exception as e:
        return f'Modify failed: {str(e)}'

def execute_trade(trade_request, is_backtest=False):
    if is_backtest:
        # For backtest, we just log the trade and return a success result
        logging.info(f"Backtest: Executing trade - {trade_request}")
        return True

    attempt = 0
    retries = 4
    delay = 6
    while attempt <= retries:
        try:
            logging.debug(f"Attempt {attempt + 1} to execute trade with request: {trade_request}")
            if not ensure_symbol_subscription(trade_request['symbol']):
                logging.error(f"Failed to subscribe to symbol {trade_request['symbol']}")
                return None
            if not check_broker_connection() or not check_market_open():
                logging.error("Trade execution aborted due to connection issues or market being closed.")
                return None
            logging.info(f"Placing order with price: {trade_request['price']}")
            result = mt5.order_send(trade_request)
            if result is None:
                logging.error(f"Failed to place order: mt5.order_send returned None. Trade Request: {trade_request}")
                raise ValueError("mt5.order_send returned None.")
            logging.info(f"Order response received at {datetime.now()}: {result}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.debug(f"Trade executed successfully: {result}")
                return result
            elif result.retcode == 10021:
                logging.warning(f"Failed to execute trade due to 'No prices' error. Attempt {attempt + 1} of {retries + 1}")
                attempt += 1
                if attempt <= retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"Failed to execute trade after {retries + 1} attempts due to 'No prices' error.")
            else:
                logging.error(f"Failed to execute trade: {result.retcode} - {result.comment}")
                return None
        except Exception as e:
            handle_error(e, f"Exception occurred during trade execution attempt {attempt + 1}")
            attempt += 1
            if attempt <= retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed to execute trade after {retries + 1} attempts due to an exception.")
                return None
    if Config.ENABLE_PENDING_ORDER_FALLBACK:
        logging.info("Attempting to place a pending order as a fallback...")
        return place_pending_order(trade_request)
    return None