import MetaTrader5 as mt5

def place_order(symbol, order_type, volume, price=None, sl=None, tp=None):
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

def close_position(ticket):
    position = mt5.positions_get(ticket=ticket)
    if position:
        result = mt5.Close(ticket)
        return result.comment if result else 'Close failed'
    return 'Position not found'

def modify_order(ticket, sl=None, tp=None):
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
