import logging
import pandas as pd
import numpy as np
import pstats
from io import StringIO
from strategy.tunnel_strategy import generate_trade_signal, execute_trade, calculate_position_size, detect_peaks_and_dips, manage_position, check_entry_conditions, check_secondary_entry_conditions
from metatrader.indicators import calculate_ema
from config import Config
import cProfile

# Initialize the logger
logger = logging.getLogger(__name__)


def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day, starting_equity, stop_loss_pips, pip_value, max_trades_per_day=None, slippage=0, transaction_cost=0, enable_profiling=False):
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
        current_day = data.iloc[0]['time'].date()
        peak_type = 21

        data = data.copy()  # Make a copy to avoid modifying the original DataFrame

        # Handle missing values by interpolation
        data['high'] = data['high'].interpolate(method='linear')
        data['close'] = data['close'].interpolate(method='linear')
        data['low'] = data['low'].interpolate(method='linear')

        # Check if the DataFrame has enough rows for EMA calculation
        if len(data) < 200:
            raise ValueError(
                "Not enough data to calculate required EMAs. Ensure data has at least 200 rows.")

        # Log the data length before EMA calculation
        logger.debug(
            f"Data length for 'high': {len(data['high'])}, 'low': {len(data['low'])}, 'close': {len(data['close'])}")

        # Calculate EMAs
        data.loc[:, 'wavy_h'] = calculate_ema(data['high'], 34)
        data.loc[:, 'wavy_c'] = calculate_ema(data['close'], 34)
        data.loc[:, 'wavy_l'] = calculate_ema(data['low'], 34)
        data.loc[:, 'tunnel1'] = calculate_ema(data['close'], 144)
        data.loc[:, 'tunnel2'] = calculate_ema(data['close'], 169)
        data.loc[:, 'long_term_ema'] = calculate_ema(data['close'], 200)

        # Peak and Dip detection
        peaks, dips = detect_peaks_and_dips(data, peak_type)

        logger.info(
            f"Starting backtest for {symbol} with initial balance: {initial_balance}")
        logger.info(
            f"Secondary strategy enabled: {Config.ENABLE_SECONDARY_STRATEGY}")

        # Loop through the data
        for i in range(34, len(data)):
            row = data.iloc[i]
            if row['time'].date() != current_day:
                current_day = row['time'].date()
                trades_today = 0
                daily_loss = 0
                logger.info(
                    f"New trading day: {current_day}, resetting daily counters.")

            if max_trades_per_day is not None and trades_today >= max_trades_per_day:
                logger.info(
                    f"Reached max trades per day: {max_trades_per_day}, skipping further trades for {current_day}.")
                continue

            primary_buy, primary_sell = check_entry_conditions(
                row, peaks, dips, symbol)
            logger.debug(
                f"Primary conditions at {row['time']}: Buy={primary_buy}, Sell={primary_sell}")

            if Config.ENABLE_SECONDARY_STRATEGY:
                secondary_buy, secondary_sell = check_secondary_entry_conditions(
                    row, symbol)
                logger.debug(
                    f"Secondary conditions at {row['time']}: Buy={secondary_buy}, Sell={secondary_sell}")
            else:
                secondary_buy, secondary_sell = False, False

            buy_condition = primary_buy or (
                Config.ENABLE_SECONDARY_STRATEGY and secondary_buy)
            sell_condition = primary_sell or (
                Config.ENABLE_SECONDARY_STRATEGY and secondary_sell)

            logger.debug(
                f"Final conditions at {row['time']}: Buy={buy_condition}, Sell={sell_condition}")

            if not (buy_condition or sell_condition):
                logger.debug(f"No trade signal generated for {row['time']}.")
                continue

            try:
                position_size = calculate_position_size(
                    balance, risk_percent, stop_loss_pips, pip_value)
            except ZeroDivisionError as e:
                logger.warning(
                    f"Zero division error while calculating position size: {e}")
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
                    'profit': 0,  # Initialize profit to 0
                    'type': 'primary' if primary_buy else 'secondary'
                }
                if execute_trade(trade, is_backtest=True):
                    trades.append(trade)
                    trades_today += 1
                    logger.info(f"Executed {trade['type'].upper()} BUY trade at {trade['entry_time']}, price: {trade['entry_price']}, volume: {trade['volume']}.")
                else:
                    logger.warning(f"Failed to execute {trade['type'].upper()} BUY trade at {trade['entry_time']}")

            elif sell_condition and (max_trades_per_day is None or trades_today < max_trades_per_day):
                trade = {
                'entry_time': row['time'],
                'entry_price': row['close'],
                'volume': position_size,
                'symbol': symbol,
                'action': 'SELL',
                'sl': row['close'] + (1.5 * std_dev),
                'tp': row['close'] - (2 * std_dev),
                'profit': 0,  # Initialize profit to 0
                'type': 'primary' if primary_sell else 'secondary'
                }
                if execute_trade(trade, is_backtest=True):
                    trades.append(trade)
                    trades_today += 1
                    logger.info(
                    f"Executed {trade['type'].upper()} SELL trade at {trade['entry_time']}, price: {trade['entry_price']}, volume: {trade['volume']}.")
                else:
                    logger.warning(f"Failed to execute {trade['type'].upper()} SELL trade at {trade['entry_time']}")

            manage_position(symbol, min_take_profit, max_loss_per_day,
                            starting_equity, max_trades_per_day)

        # Calculate profits/losses for each trade
        for trade in trades:
            exit_price = trade['tp'] if trade['action'] == 'BUY' else trade['sl']
            if trade['action'] == 'BUY':
                trade['profit'] = (exit_price - trade['entry_price']) * \
                    trade['volume'] - slippage - transaction_cost
            else:
                trade['profit'] = (trade['entry_price'] - exit_price) * \
                    trade['volume'] - slippage - transaction_cost
            logger.info(
                f"Trade closed at {trade['entry_time']}, action: {trade['action']}, type: {trade['type']}, profit: {trade['profit']}.")

        total_profit = sum(trade.get('profit', 0) for trade in trades)
        num_trades = len(trades)
        win_rate = sum(1 for trade in trades if trade.get(
            'profit', 0) > 0) / num_trades if num_trades > 0 else 0
        max_drawdown = calculate_max_drawdown(trades, initial_balance)

        final_balance = balance + total_profit

        logger.info(f"Backtest completed for {symbol}.")
        logger.info(f"Total Profit: {total_profit}")
        logger.info(f"Final Balance: {final_balance}")
        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(
            f"Total primary trades: {sum(1 for trade in trades if trade['type'] == 'primary')}")
        logger.info(
            f"Total secondary trades: {sum(1 for trade in trades if trade['type'] == 'secondary')}")

        return {
            'total_profit': total_profit,
            'final_balance': final_balance,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'buy_condition': buy_condition,
            'sell_condition': sell_condition,
            'trades': trades,
            'total_slippage_costs': len(trades) * slippage,
            'total_transaction_costs': len(trades) * transaction_cost
        }

    finally:
        if enable_profiling and pr:
            pr.disable()
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(
                pstats.SortKey.CUMULATIVE)
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
            drawdown = max_balance - balance
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
