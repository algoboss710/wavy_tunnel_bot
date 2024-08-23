import logging

# Get the logger for this module
logger = logging.getLogger(__name__)

def calculate_trade_costs(symbol, position_size, pip_value, volatility, slippage_pips, spread_pips, commission_per_lot):
    logger.debug(f"Calculating costs for {symbol}: position_size={position_size}, pip_value={pip_value}, volatility={volatility}")
    
    slippage_cost = slippage_pips * pip_value * position_size
    spread_cost = spread_pips * pip_value * position_size
    commission = commission_per_lot * (position_size / 100000)  # Assuming standard lot size
    
    logger.debug(f"Calculated costs: slippage={slippage_cost}, spread={spread_cost}, commission={commission}")
    
    return {
        'slippage': slippage_cost,
        'spread': spread_cost,
        'commission': commission
    }