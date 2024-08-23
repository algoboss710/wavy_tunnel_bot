def calculate_trade_costs(symbol, position_size, pip_value, volatility, slippage_pips, spread_pips, commission_per_lot):
    # Convert position size to lots (assuming standard lot size of 100,000)
    lots = position_size / 100000

    slippage_cost = slippage_pips * pip_value * position_size
    spread_cost = spread_pips * pip_value * position_size
    commission = commission_per_lot * lots

    return {
        'slippage': slippage_cost,
        'spread': spread_cost,
        'commission': commission
    }