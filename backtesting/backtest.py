import pandas as pd
from metatrader.data_retrieval import get_historical_data
from strategy.tunnel_strategy import generate_trade_signal, execute_trade, manage_position, calculate_position_size
from utils.plotting import plot_backtest_results

def run_backtest(symbol, start_date, end_date, timeframe, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day):
    # Retrieve historical data
    data = get_historical_data(symbol, timeframe, start_date, end_date)

    # Initialize variables
    balance = initial_balance
    trades = []

    # Backtesting loop
    for i in range(len(data)):
        # Calculate indicators and generate trading signals
        signal = generate_trade_signal(data.iloc[:i+1], period=20, deviation_factor=2.0)

        if signal == 'BUY':
            # Simulate trade entry
            trade = {
                'entry_time': data.iloc[i]['time'],
                'entry_price': data.iloc[i]['close'],
                'volume': calculate_position_size(balance, risk_percent)
            }
            trades.append(trade)
            execute_trade(trade)

        elif signal == 'SELL':
            # Simulate trade exit
            if trades:
                trade = trades[-1]
                trade['exit_time'] = data.iloc[i]['time']
                trade['exit_price'] = data.iloc[i]['close']
                trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['volume']
                balance += trade['profit']
                execute_trade(trade)

        manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

    # Calculate performance metrics
    total_profit = balance - initial_balance
    num_trades = len(trades)
    win_rate = sum(1 for trade in trades if trade['profit'] > 0) / num_trades
    max_drawdown = calculate_max_drawdown(trades, initial_balance)

    # Print performance metrics
    print(f"Total Profit: {total_profit:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}")

    # Plot backtest results
    plot_backtest_results(data, trades)

def calculate_max_drawdown(trades, initial_balance):
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0

    for trade in trades:
        balance += trade['profit']
        max_balance = max(max_balance, balance)
        drawdown = max_balance - balance
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown