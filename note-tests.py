import logging
import pandas as pd
import numpy as np
import pstats
from io import StringIO
from strategy.tunnel_strategy import (
    calculate_position_size, detect_peaks_and_dips, manage_position,
    check_entry_conditions, calculate_ema, execute_trade
)
from metatrader.data_retrieval import get_data
import cProfile
import MetaTrader5 as mt5
from config import Config

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_backtest(symbol, initial_balance, risk_percent, min_take_profit, max_loss_per_day,
                 starting_equity, stop_loss_pips, pip_value, start_date=None, end_date=None,
                 timeframe=mt5.TIMEFRAME_H1, max_trades_per_day=None, slippage=0,
                 transaction_cost=0, enable_profiling=False, data=None):
    """
    Run a backtest for a given symbol with historical data.
    """
    # Initialize the profiler if profiling is enabled
    pr = cProfile.Profile() if enable_profiling else None
    if enable_profiling:
        pr.enable()

    # Check for zero or negative initial balance
    if initial_balance <= 0:
        raise ValueError("Initial balance must be greater than zero.")

    # Check for zero risk percentage
    if risk_percent == 0:
        raise ValueError("Risk percentage cannot be zero.")

    try:
        balance = initial_balance
        trades = []
        trades_today = 0
        peak_type = 21

        # Use provided data if available, otherwise fetch it
        if data is None:
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date must be provided if data is not supplied")
            data = get_data(symbol, mode='backtest', start_date=start_date, end_date=end_date, timeframe=timeframe)
        else:
            # If data is provided, extract start_date and end_date from it
            start_date = data['time'].min()
            end_date = data['time'].max()

        if data is None or data.empty:
            logger.error(f"No historical data available for {symbol}")
            return None

        current_day = data.iloc[0]['time'].date()

        data = data.copy()  # Make a copy to avoid modifying the original DataFrame

        # Handle missing values by interpolation
        data['high'] = data['high'].interpolate(method='linear')
        data['close'] = data['close'].interpolate(method='linear')
        data['low'] = data['low'].interpolate(method='linear')

        # Check if the DataFrame has enough rows for EMA calculation
        if len(data) < 200:
            raise ValueError("Not enough data to calculate required EMAs. Ensure data has at least 200 rows.")

        # Log the data length before EMA calculation
        logger.debug(f"Data length for 'high': {len(data['high'])}, 'low': {len(data['low'])}, 'close': {len(data['close'])}")

        # Calculate EMAs
        data.loc[:, 'wavy_h'] = calculate_ema(data['high'], 34)
        data.loc[:, 'wavy_c'] = calculate_ema(data['close'], 34)
        data.loc[:, 'wavy_l'] = calculate_ema(data['low'], 34)
        data.loc[:, 'tunnel1'] = calculate_ema(data['close'], 144)
        data.loc[:, 'tunnel2'] = calculate_ema(data['close'], 169)
        data.loc[:, 'long_term_ema'] = calculate_ema(data['close'], 200)

        # Peak and Dip detection
        peaks, dips = detect_peaks_and_dips(data, peak_type)

        # Loop through the data
        for i in range(200, len(data)):  # Start from 200 to ensure all indicators are calculated
            row = data.iloc[i]
            if row['time'].date() != current_day:
                current_day = row['time'].date()
                trades_today = 0
                logger.info(f"New trading day: {current_day}, resetting daily counters.")

            if max_trades_per_day is not None and trades_today >= max_trades_per_day:
                logger.info(f"Reached max trades per day: {max_trades_per_day}, skipping further trades for {current_day}.")
                continue

            buy_condition, sell_condition = check_entry_conditions(row, peaks, dips, symbol)

            if not buy_condition and not sell_condition:
                logger.debug(f"No trade signal generated for {row['time']}.")
                continue

            try:
                position_size = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
            except ZeroDivisionError as e:
                logger.warning(f"Zero division error while calculating position size: {e}")
                continue

            std_dev = data['close'].rolling(window=20).std().iloc[i]

            if buy_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
                trade = {
                    'entry_time': row['time'],
                    'entry_price': row['close'],
                    'volume': position_size,
                    'symbol': symbol,
                    'action': 'BUY',
                    'sl': row['close'] - (1.5 * std_dev),
                    'tp': row['close'] + (2 * std_dev),
                    'profit': 0  # Initialize profit to 0
                }
                execute_trade(trade)
                trades.append(trade)
                trades_today += 1
                logger.info(f"Executed BUY trade at {trade['entry_time']}, price: {trade['entry_price']}, volume: {trade['volume']}.")

            elif sell_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
                trade = {
                    'entry_time': row['time'],
                    'entry_price': row['close'],
                    'volume': position_size,
                    'symbol': symbol,
                    'action': 'SELL',
                    'sl': row['close'] + (1.5 * std_dev),
                    'tp': row['close'] - (2 * std_dev),
                    'profit': 0  # Initialize profit to 0
                }
                execute_trade(trade)
                trades.append(trade)
                trades_today += 1
                logger.info(f"Executed SELL trade at {trade['entry_time']}, price: {trade['entry_price']}, volume: {trade['volume']}.")

            manage_position(symbol, min_take_profit, max_loss_per_day, starting_equity, max_trades_per_day)

            for trade in trades:
                exit_price = trade['tp'] if trade['action'] == 'BUY' else trade['sl']
                if trade['action'] == 'BUY':
                    trade['profit'] = (exit_price - trade['entry_price']) * trade['volume'] - slippage - transaction_cost
                else:
                    trade['profit'] = (trade['entry_price'] - exit_price) * trade['volume'] - slippage - transaction_cost
                logger.info(f"Trade closed at {trade['entry_time']}, action: {trade['action']}, profit: {trade['profit']}.")

            total_profit = sum(trade.get('profit', 0) for trade in trades)
            num_trades = len(trades)
            win_rate = sum(1 for trade in trades if trade.get('profit', 0) > 0) / num_trades if num_trades > 0 else 0
            max_drawdown = calculate_max_drawdown(trades, initial_balance)

            final_balance = balance + total_profit

            logger.info(f"Backtest completed. Total Profit: {total_profit}, Final Balance: {final_balance}, Number of Trades: {num_trades}, Win Rate: {win_rate}, Max Drawdown: {max_drawdown}.")

            return {
                'total_profit': total_profit,
                'final_balance': final_balance,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'total_slippage_costs': len(trades) * slippage,
                'total_transaction_costs': len(trades) * transaction_cost
            }

    finally:
        if enable_profiling and pr:
            pr.disable()
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
            print(s.getvalue())

def calculate_max_drawdown(trades, initial_balance):
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0

    for trade in trades:
        if 'profit' in trade:
            balance += trade['profit']
            max_balance = max(max_balance, balance)
            drawdown = (max_balance - balance) / max_balance
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown