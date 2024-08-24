import logging
import pandas as pd
import numpy as np
import pstats
from io import StringIO
from strategy.tunnel_strategy import generate_trade_signal, calculate_position_size, detect_peaks_and_dips, manage_position, check_entry_conditions
from metatrader.indicators import calculate_ema
from metatrader.trade_management import execute_trade
from utils.cost_calculation import calculate_trade_costs
import cProfile

logger = logging.getLogger(__name__)

def log_trade_details(trade, costs, balance, logger):
    logger.debug(f"Trade Details:")
    logger.debug(f"  Action: {trade['action']}")
    logger.debug(f"  Entry Time: {trade['entry_time']}")
    logger.debug(f"  Entry Price: {trade['entry_price']:.5f}")
    logger.debug(f"  Adjusted Entry Price: {trade['adjusted_entry_price']:.5f}")
    logger.debug(f"  Volume: {trade['volume']:.2f}")
    logger.debug(f"  Stop Loss: {trade['sl']:.5f}")
    logger.debug(f"  Take Profit: {trade['tp']:.5f}")
    logger.debug(f"  Costs:")
    logger.debug(f"    Slippage: {costs['slippage']:.5f}")
    logger.debug(f"    Spread: {costs['spread']:.5f}")
    logger.debug(f"    Commission: {costs['commission']:.5f}")
    logger.debug(f"  Total Cost: {sum(costs.values()):.5f}")
    logger.debug(f"  Balance After Trade: {balance:.2f}")

def log_trade_closure(trade, exit_price, raw_profit, costs, net_profit, balance, logger):
    logger.debug(f"Trade Closure:")
    logger.debug(f"  Exit Time: {trade['exit_time']}")
    logger.debug(f"  Exit Price: {exit_price:.5f}")
    logger.debug(f"  Raw Profit: {raw_profit:.5f}")
    logger.debug(f"  Exit Costs:")
    logger.debug(f"    Slippage: {costs['slippage']:.5f}")
    logger.debug(f"    Spread: {costs['spread']:.5f}")
    logger.debug(f"    Commission: {costs['commission']:.5f}")
    logger.debug(f"  Total Exit Cost: {sum(costs.values()):.5f}")
    logger.debug(f"  Net Profit: {net_profit:.5f}")
    logger.debug(f"  Balance After Closure: {balance:.2f}")

def run_backtest(symbol, data, initial_balance, risk_percent, min_take_profit, max_loss_per_day,
                 starting_equity, stop_loss_pips, pip_value, max_trades_per_day=None,
                 slippage_pips=1, spread_pips=2, commission_per_lot=7, enable_profiling=False):
    logger.debug("Starting run_backtest function")
    pr = cProfile.Profile() if enable_profiling else None
    if enable_profiling:
        pr.enable()

    try:
        balance = initial_balance
        trades = []
        trades_today = 0
        current_day = data.iloc[0]['time'].date()
        peak_balance = initial_balance
        max_drawdown = 0

        data = data.copy()
        data['high'] = data['high'].interpolate(method='linear')
        data['close'] = data['close'].interpolate(method='linear')
        data['low'] = data['low'].interpolate(method='linear')

        if len(data) < 200:
            raise ValueError("Not enough data to calculate required EMAs. Ensure data has at least 200 rows.")

        logger.debug(f"Data length for 'high': {len(data['high'])}, 'low': {len(data['low'])}, 'close': {len(data['close'])}")

        data.loc[:, 'wavy_h'] = calculate_ema(data['high'], 34)
        data.loc[:, 'wavy_c'] = calculate_ema(data['close'], 34)
        data.loc[:, 'wavy_l'] = calculate_ema(data['low'], 34)
        data.loc[:, 'tunnel1'] = calculate_ema(data['close'], 144)
        data.loc[:, 'tunnel2'] = calculate_ema(data['close'], 169)
        data.loc[:, 'long_term_ema'] = calculate_ema(data['close'], 200)

        peaks, dips = detect_peaks_and_dips(data, 21)

        for i in range(34, len(data)):
            row = data.iloc[i]
            logger.debug(f"Processing data point {i}: {row['time']}")

            if row['time'].date() != current_day:
                current_day = row['time'].date()
                trades_today = 0
                logger.info(f"New trading day: {current_day}, resetting daily counters.")

            if max_trades_per_day is not None and trades_today >= max_trades_per_day:
                logger.info(f"Reached max trades per day: {max_trades_per_day}, skipping further trades for {current_day}.")
                continue

            buy_condition, sell_condition = check_entry_conditions(row, peaks, dips, symbol)
            logger.debug(f"Entry conditions - Buy: {buy_condition}, Sell: {sell_condition}")

            volatility = np.std(data['close'].iloc[i-20:i+1])

            if buy_condition or sell_condition:
                logger.debug("Trade condition met, calculating position size and costs")
                try:
                    position_size = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
                    position_size_units = position_size * 100000  # Convert lots to units
                except Exception as e:
                    logger.warning(f"Error calculating position size: {e}")
                    continue

                entry_costs = calculate_trade_costs(symbol, position_size_units, pip_value, volatility,
                                                    slippage_pips, spread_pips, commission_per_lot)

                adjusted_entry_price = row['close'] + (entry_costs['slippage'] + entry_costs['spread']) / position_size_units if buy_condition else row['close'] - (entry_costs['slippage'] + entry_costs['spread']) / position_size_units

                # Calculate stop loss and take profit based on adjusted entry price
                min_sl_distance = 5 * pip_value  # Minimum 5 pips distance for stop loss
                if buy_condition:
                    sl = min(adjusted_entry_price - (stop_loss_pips * pip_value), adjusted_entry_price - min_sl_distance)
                    tp = adjusted_entry_price + (2 * stop_loss_pips * pip_value)
                else:
                    sl = max(adjusted_entry_price + (stop_loss_pips * pip_value), adjusted_entry_price + min_sl_distance)
                    tp = adjusted_entry_price - (2 * stop_loss_pips * pip_value)

                trade = {
                    'entry_time': row['time'],
                    'entry_price': row['close'],
                    'adjusted_entry_price': adjusted_entry_price,
                    'volume': position_size,  # This is now in lots
                    'symbol': symbol,
                    'action': 'BUY' if buy_condition else 'SELL',
                    'sl': sl,
                    'tp': tp,
                    'entry_costs': entry_costs,
                    'profit': -sum(entry_costs.values())
                }

                trades.append(trade)
                trades_today += 1

                balance -= sum(entry_costs.values())

                logger.debug(f"Trade executed - Entry: {row['close']}, Adjusted Entry: {adjusted_entry_price}, SL: {trade['sl']}, TP: {trade['tp']}")
                log_trade_details(trade, entry_costs, balance, logger)

            for trade in trades:
                if trade.get('exit_time') is None:
                    exit_price = None
                    if trade['action'] == 'BUY':
                        if row['low'] <= trade['sl']:
                            exit_price = trade['sl']
                        elif row['high'] >= trade['tp']:
                            exit_price = trade['tp']
                    else:  # SELL
                        if row['high'] >= trade['sl']:
                            exit_price = trade['sl']
                        elif row['low'] <= trade['tp']:
                            exit_price = trade['tp']

                    if exit_price:
                        trade['exit_time'] = row['time']
                        trade['exit_price'] = exit_price

                        exit_costs = calculate_trade_costs(symbol, trade['volume'] * 100000, pip_value, volatility,
                                                           slippage_pips, spread_pips, commission_per_lot)

                        raw_profit = (exit_price - trade['adjusted_entry_price']) * trade['volume'] * 100000 if trade['action'] == 'BUY' else (trade['adjusted_entry_price'] - exit_price) * trade['volume'] * 100000
                        net_profit = raw_profit - sum(exit_costs.values())

                        trade['exit_costs'] = exit_costs
                        trade['profit'] = net_profit
                        trade['is_win'] = net_profit > 0

                        balance += net_profit

                        logger.debug(f"Trade closed - Exit: {exit_price}, Raw Profit: {raw_profit}, Net Profit: {net_profit}")
                        log_trade_closure(trade, exit_price, raw_profit, exit_costs, net_profit, balance, logger)

                        peak_balance = max(peak_balance, balance)
                        current_drawdown = peak_balance - balance
                        max_drawdown = max(max_drawdown, current_drawdown)

        total_profit = sum(trade['profit'] for trade in trades)
        num_trades = len(trades)
        win_rate = sum(1 for trade in trades if trade.get('is_win', False)) / num_trades if num_trades > 0 else 0

        logger.info(f"Backtest completed with {num_trades} trades, win rate: {win_rate:.2%}, total profit: {total_profit:.2f}")

        return {
            'total_profit': total_profit,
            'final_balance': balance,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_slippage_costs': sum(trade['entry_costs']['slippage'] + trade['exit_costs']['slippage'] for trade in trades),
            'total_spread_costs': sum(trade['entry_costs']['spread'] + trade['exit_costs']['spread'] for trade in trades),
            'total_commissions': sum(trade['entry_costs']['commission'] + trade['exit_costs']['commission'] for trade in trades),
            'total_transaction_costs': sum(sum(trade['entry_costs'].values()) + sum(trade['exit_costs'].values()) for trade in trades)
        }

    except Exception as e:
        logger.error(f"An error occurred during backtesting: {str(e)}")
        return None

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
            drawdown = max_balance - balance
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')