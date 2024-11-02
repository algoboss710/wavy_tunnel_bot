import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
import os
import concurrent.futures
import time
import gc
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pickle
import sys

# Set up global logging with both file and console handlers
def setup_global_logging(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_global_logging()

class OptimizationResult:
    def __init__(self, config: Dict, summary: Dict, timeframe: str):
        self.config = config
        self.summary = summary
        self.timeframe = timeframe
        self.net_profit = summary.get('net_profit', 0)
        self.total_trades = (
            summary.get('trades_executed_long', 0) +
            summary.get('trades_executed_short', 0) +
            summary.get('trades_executed_second_long', 0) +
            summary.get('trades_executed_second_short', 0)
        )
        self.win_rate = summary.get('win_rate', 0)
        self.max_drawdown = summary.get('max_drawdown', 0)
        self.profit_factor = summary.get('profit_factor', 0)
        self.avg_trade = summary.get('average_trade', 0)
        self.sharpe_ratio = summary.get('sharpe_ratio', 0)
        self.recovery_factor = summary.get('recovery_factor', 0)
        self.expectancy = summary.get('expectancy', 0)

    def __str__(self):
        return (f"TimeFrame: {self.timeframe}, "
                f"Net Profit: ${self.net_profit:,.2f}, "
                f"Win Rate: {self.win_rate:.1f}%")

class Trade:
    def __init__(self, entry_price: float, entry_time: datetime,
                 direction: str, lot_size: float, stop_loss: float = None):
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.direction = direction
        self.lot_size = lot_size
        self.stop_loss = stop_loss
        self.take_profits = []
        self.status = 'open'
        self.profit = 0
        self.remaining_quantity = lot_size

        logging.debug(
            f"New trade created: {direction} at {entry_price:.5f}, "
            f"lot_size: {lot_size}, stop_loss: {stop_loss:.5f}"
        )

    def add_take_profit(self, price: float, quantity: float):
        self.take_profits.append((price, quantity))
        logging.debug(f"Added TP level: Price {price:.5f}, Quantity {quantity}")

    def close_trade(self, exit_price: float, exit_time: datetime, quantity: float):
        self.exit_price = exit_price
        self.exit_time = exit_time

        if self.direction == 'long':
            self.profit += (exit_price - self.entry_price) * quantity * 100000
        else:
            self.profit += (self.entry_price - exit_price) * quantity * 100000

        self.remaining_quantity -= quantity

        if self.remaining_quantity <= 0:
            self.status = 'closed'

        logging.debug(
            f"Trade closed: {self.direction} Exit: {exit_price:.5f}, "
            f"Profit: {self.profit:.2f}, Remaining Qty: {self.remaining_quantity}"
        )

class DataCache:
    def __init__(self):
        self._cache = {}
        self._min_required_data = {
            "M15": 200,
            "H1": 200,
            "H4": 200,
            "D1": 250  # Increased for daily timeframe
        }
        logging.info("DataCache initialized")

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            logging.debug(f"Cache hit for {key}")
            return self._cache[key]
        logging.debug(f"Cache miss for {key}")
        return None

    def set(self, key: str, data: np.ndarray):
        timeframe = key.split('_')[1]  # Extract timeframe from key
        if len(data) < self._min_required_data[timeframe]:
            logging.warning(
                f"Insufficient data for {timeframe}. "
                f"Got {len(data)} bars, need {self._min_required_data[timeframe]}"
            )
        self._cache[key] = data
        logging.debug(f"Cached data for {key}: {len(data)} records")

    def clear(self):
        self._cache.clear()
        gc.collect()  # Force garbage collection
        logging.debug("Cache cleared and memory freed")

    def save_to_disk(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self._cache, f)
        logging.info(f"Cache saved to {filepath}")

    def load_from_disk(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self._cache = pickle.load(f)
            logging.info(f"Cache loaded from {filepath}")

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average with validation"""
    min_required = period * 2  # Require at least 2x period length for better accuracy
    if len(data) < min_required:
        logging.warning(
            f"Data length ({len(data)}) is less than minimum required for EMA{period} ({min_required})"
        )
        return pd.Series(np.nan, index=data.index)

    ema = data.ewm(span=period, adjust=False).mean()
    logging.debug(f"Calculated EMA{period}, data points: {len(ema)}")
    return ema

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index with validation"""
    min_required = period * 2
    if len(data) < min_required:
        logging.warning(
            f"Data length ({len(data)}) is less than minimum required for RSI{period} ({min_required})"
        )
        return pd.Series(np.nan, index=data.index)

    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    logging.debug(f"Calculated RSI{period}, data points: {len(rsi)}")
    return rsi

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range with validation"""
    min_required = period * 2
    if len(high) < min_required:
        logging.warning(
            f"Data length ({len(high)}) is less than minimum required for ATR{period} ({min_required})"
        )
        return pd.Series(np.nan, index=high.index)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    logging.debug(f"Calculated ATR{period}, data points: {len(atr)}")
    return atr

def identify_peaks(prices: pd.Series, lookback: int, threshold: float) -> pd.Series:
    """Identify price peaks with validation"""
    min_required = lookback * 2 + 1
    if len(prices) < min_required:
        logging.warning(
            f"Data length ({len(prices)}) is insufficient for peak detection. Need {min_required}"
        )
        return pd.Series(0, index=prices.index)

    peaks = pd.Series(0, index=prices.index)
    price_array = prices.values
    peak_count = 0

    for i in range(lookback, len(prices) - lookback):
        is_peak = True
        current_price = price_array[i]

        # Check previous prices
        for j in range(1, lookback + 1):
            if current_price <= price_array[i-j] * (1 + threshold):
                is_peak = False
                break

        # Check next prices if still potential peak
        if is_peak:
            for j in range(1, lookback + 1):
                if current_price <= price_array[i+j] * (1 + threshold):
                    is_peak = False
                    break

        if is_peak:
            peaks.iloc[i] = 1
            peak_count += 1

    logging.debug(f"Identified {peak_count} peaks (lookback: {lookback}, threshold: {threshold:.4f})")
    return peaks

def identify_dips(prices: pd.Series, lookback: int, threshold: float) -> pd.Series:
    """Identify price dips with validation"""
    min_required = lookback * 2 + 1
    if len(prices) < min_required:
        logging.warning(
            f"Data length ({len(prices)}) is insufficient for dip detection. Need {min_required}"
        )
        return pd.Series(0, index=prices.index)

    dips = pd.Series(0, index=prices.index)
    price_array = prices.values
    dip_count = 0

    for i in range(lookback, len(prices) - lookback):
        is_dip = True
        current_price = price_array[i]

        # Check previous prices
        for j in range(1, lookback + 1):
            if current_price >= price_array[i-j] * (1 - threshold):
                is_dip = False
                break

        # Check next prices if still potential dip
        if is_dip:
            for j in range(1, lookback + 1):
                if current_price >= price_array[i+j] * (1 - threshold):
                    is_dip = False
                    break

        if is_dip:
            dips.iloc[i] = 1
            dip_count += 1

    logging.debug(f"Identified {dip_count} dips (lookback: {lookback}, threshold: {threshold:.4f})")
    return dips

def calculate_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Calculate all technical indicators needed for the strategy"""
    logging.info("Starting indicator calculations")
    initial_len = len(df)

    if initial_len == 0:
        logging.error("Empty dataframe provided for indicator calculation")
        return df

    try:
        # Validate minimum data requirements
        min_required = max(
            config['tunnel_ema2'] * 2,  # Largest EMA period * 2
            config['peak_dip_lookback'] * 2 + 1,  # For peak/dip detection
            config['rsi_period'] * 2  # For RSI calculation
        )

        if initial_len < min_required:
            logging.error(f"Insufficient data for calculations. Need {min_required}, got {initial_len}")
            return df

        # EMAs
        df['wavy_ema'] = calculate_ema(df['close'], config['wavy_ema'])
        df['tunnel_ema1'] = calculate_ema(df['close'], config['tunnel_ema1'])
        df['tunnel_ema2'] = calculate_ema(df['close'], config['tunnel_ema2'])

        # RSI
        df['rsi'] = calculate_rsi(df['close'], config['rsi_period'])

        # ATR
        df['atr'] = calculate_atr(
            df['high'],
            df['low'],
            df['close'],
            config['atr_period']
        )

        # Peaks and Dips
        lookback = config['peak_dip_lookback']
        threshold = config['peak_dip_threshold']

        df['is_peak'] = identify_peaks(df['close'], lookback, threshold)
        df['is_dip'] = identify_dips(df['close'], lookback, threshold)

        # Log indicator statistics
        peaks_count = df['is_peak'].sum()
        dips_count = df['is_dip'].sum()
        nan_count = df.isna().sum().sum()

        logging.info(f"""
        Indicator Calculation Summary:
        - Initial bars: {initial_len}
        - Peaks identified: {peaks_count}
        - Dips identified: {dips_count}
        - NaN values: {nan_count}
        - Lookback period: {lookback}
        - Threshold: {threshold}
        """)

        return df

    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        raise

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, config: Dict):
        self.data = data.copy()
        self.config = config
        self.trades = []
        self.current_trade = None
        self.equity_curve = []
        self.total_equity = 0
        self.start_time = time.time()

        # Validate configuration
        if not self.validate_config():
            raise ValueError("Invalid configuration provided")

        logging.info(f"""
        BacktestEngine initialized:
        - Initial data points: {len(data)}
        - Date range: {data.index[0]} to {data.index[-1]}
        - Timeframe: {config['timeframe']}
        - Configuration: {json.dumps(config, indent=2)}
        """)

    def validate_config(self) -> bool:
        """Validate backtest configuration"""
        try:
            required_fields = [
                'wavy_ema', 'tunnel_ema1', 'tunnel_ema2',
                'peak_dip_lookback', 'peak_dip_threshold',
                'tp_levels', 'tp_quantities'
            ]

            for field in required_fields:
                if field not in self.config:
                    logging.error(f"Missing required field: {field}")
                    return False

            # Validate take profit levels
            if not all(0 < x < 1 for x in self.config['tp_quantities']):
                logging.error("Take profit quantities must be between 0 and 1")
                return False

            if abs(sum(self.config['tp_quantities']) - 1.0) > 0.0001:
                logging.error("Take profit quantities must sum to 1")
                return False

            return True

        except Exception as e:
            logging.error(f"Config validation failed: {str(e)}")
            return False

    def run_backtest(self) -> Dict:
        """Run the complete backtest"""
        try:
            logging.info("Starting backtest")

            if len(self.data) == 0:
                logging.error("Empty dataset provided for backtest")
                return self._empty_results()

            # Initialize indicators
            self.data = calculate_indicators(self.data, self.config)

            # Remove initial NaN values from indicator calculations
            lookback = max(
                self.config['peak_dip_lookback'],
                self.config['tunnel_ema2'],
                self.config['rsi_period']
            )

            initial_len = len(self.data)
            self.data = self.data.iloc[lookback:].copy()
            self.data = self.data.dropna()  # Remove any remaining NaN values

            if len(self.data) < lookback:
                logging.error(f"Insufficient data after preparation. Remaining bars: {len(self.data)}")
                return self._empty_results()

            logging.info(f"""
            Data preparation complete:
            - Original bars: {initial_len}
            - Usable bars after indicator calculation: {len(self.data)}
            - Removed bars: {initial_len - len(self.data)}
            - Final date range: {self.data.index[0]} to {self.data.index[-1]}
            """)

            # Run through each bar
            for i in range(1, len(self.data)):
                try:
                    self.process_bar(i)
                    self.equity_curve.append(self.total_equity)

                    # Log progress every 1000 bars
                    if i % 1000 == 0:
                        elapsed_time = time.time() - self.start_time
                        bars_per_second = i / elapsed_time
                        logging.info(f"""
                        Progress Update:
                        - Processed {i}/{len(self.data)} bars ({i/len(self.data)*100:.1f}%)
                        - Current equity: ${self.total_equity:.2f}
                        - Processing speed: {bars_per_second:.0f} bars/second
                        """)

                except Exception as e:
                    logging.error(f"Error processing bar {i}: {str(e)}")
                    raise

            # Close any remaining trades
            if self.current_trade and self.current_trade.status == 'open':
                logging.info("Closing remaining open trade at end of backtest")
                self.close_trade(len(self.data)-1, 'final_close')

            # Calculate and return results
            results = self.calculate_results()

            # Log performance metrics
            elapsed_time = time.time() - self.start_time
            total_bars = len(self.data)
            bars_per_second = total_bars / elapsed_time

            logging.info(f"""
            Backtest completed:
            - Total time: {elapsed_time:.2f} seconds
            - Processing speed: {bars_per_second:.0f} bars/second
            - Total bars processed: {total_bars}
            - Final results: {json.dumps(results, indent=2)}
            """)

            return results

        except Exception as e:
            logging.error(f"Backtest failed: {str(e)}")
            return self._empty_results()

    def process_bar(self, i: int):
        """Process each price bar"""
        try:
            # Check existing trade
            if self.current_trade and self.current_trade.status == 'open':
                self.check_exit_conditions(i)

            # Check for new trade if no current trade
            elif not self.current_trade or self.current_trade.status == 'closed':
                self.check_entry_conditions(i)

        except Exception as e:
            logging.error(f"Error processing bar {i}: {str(e)}")
            raise

    def check_entry_conditions(self, i: int):
        """Check and execute entry conditions"""
        try:
            row = self.data.iloc[i]
            prev_row = self.data.iloc[i-1]

            # Log conditions check periodically
            if i % 500 == 0:
                logging.debug(f"""
                Entry Conditions Check (Bar {i}):
                Close: {row['close']}
                EMAs: {row['wavy_ema']:.2f}/{row['tunnel_ema1']:.2f}/{row['tunnel_ema2']:.2f}
                RSI: {row['rsi']:.2f}
                Previous Peak/Dip: {prev_row['is_peak']}/{prev_row['is_dip']}
                """)

            # Long entry conditions
            long_conditions = (
                row['close'] > row['tunnel_ema1'] and
                row['close'] > row['tunnel_ema2'] and
                row['wavy_ema'] > row['tunnel_ema1'] and
                prev_row['is_dip'] == 1 and
                (not self.config['apply_rsi_filter'] or row['rsi'] < self.config['rsi_upper'])
            )

            # Short entry conditions
            short_conditions = (
                row['close'] < row['tunnel_ema1'] and
                row['close'] < row['tunnel_ema2'] and
                row['wavy_ema'] < row['tunnel_ema1'] and
                prev_row['is_peak'] == 1 and
                (not self.config['apply_rsi_filter'] or row['rsi'] > self.config['rsi_lower'])
            )

            if long_conditions:
                logging.info(f"""
                Long Entry Signal (Bar {i}):
                Close: {row['close']}
                EMAs: {row['wavy_ema']:.2f}/{row['tunnel_ema1']:.2f}/{row['tunnel_ema2']:.2f}
                RSI: {row['rsi']:.2f}
                Previous Dip: {prev_row['is_dip']}
                """)
                self.enter_trade(i, 'long')

            elif short_conditions:
                logging.info(f"""
                Short Entry Signal (Bar {i}):
                Close: {row['close']}
                EMAs: {row['wavy_ema']:.2f}/{row['tunnel_ema1']:.2f}/{row['tunnel_ema2']:.2f}
                RSI: {row['rsi']:.2f}
                Previous Peak: {prev_row['is_peak']}
                """)
                self.enter_trade(i, 'short')

        except Exception as e:
            logging.error(f"Error checking entry conditions: {str(e)}")
            raise
    def enter_trade(self, i: int, direction: str):
        """Enter a new trade"""
        try:
            entry_price = self.data.iloc[i]['close']
            entry_time = self.data.index[i]

            # Calculate stop loss and take profits
            atr = self.data.iloc[i]['atr']
            stop_distance = atr * self.config['tp_atr_multiplier']

            stop_loss = entry_price - stop_distance if direction == 'long' else entry_price + stop_distance

            # Create new trade
            self.current_trade = Trade(
                entry_price=entry_price,
                entry_time=entry_time,
                direction=direction,
                lot_size=self.config['lot_size'],
                stop_loss=stop_loss
            )

            # Set take profit levels
            for level, quantity in zip(self.config['tp_levels'], self.config['tp_quantities']):
                tp_distance = atr * self.config['tp_atr_multiplier'] * level
                tp_price = entry_price + tp_distance if direction == 'long' else entry_price - tp_distance
                self.current_trade.add_take_profit(tp_price, quantity * self.config['lot_size'])

            self.trades.append(self.current_trade)

            logging.info(f"""
            New Trade Opened:
            Direction: {direction}
            Entry Price: {entry_price:.5f}
            Stop Loss: {stop_loss:.5f}
            ATR: {atr:.5f}
            Take Profit Levels: {[f"{tp[0]:.5f}" for tp in self.current_trade.take_profits]}
            """)

        except Exception as e:
            logging.error(f"Error entering trade: {str(e)}")
            raise

    def check_exit_conditions(self, i: int):
        """Check and execute exit conditions"""
        try:
            current_price = self.data.iloc[i]['close']
            current_time = self.data.index[i]

            # Check stop loss
            if self.current_trade.direction == 'long' and current_price <= self.current_trade.stop_loss:
                logging.info(f"""
                Stop Loss Hit (Long):
                Entry: {self.current_trade.entry_price:.5f}
                Exit: {current_price:.5f}
                Stop Level: {self.current_trade.stop_loss:.5f}
                """)
                self.close_trade(i, 'stop_loss')

            elif self.current_trade.direction == 'short' and current_price >= self.current_trade.stop_loss:
                logging.info(f"""
                Stop Loss Hit (Short):
                Entry: {self.current_trade.entry_price:.5f}
                Exit: {current_price:.5f}
                Stop Level: {self.current_trade.stop_loss:.5f}
                """)
                self.close_trade(i, 'stop_loss')

            # Check take profits
            if self.current_trade.take_profits:
                remaining_tps = []
                for tp_price, tp_quantity in self.current_trade.take_profits:
                    if (self.current_trade.direction == 'long' and current_price >= tp_price) or \
                       (self.current_trade.direction == 'short' and current_price <= tp_price):
                        self.partial_close(i, tp_price, tp_quantity, 'take_profit')
                    else:
                        remaining_tps.append((tp_price, tp_quantity))
                self.current_trade.take_profits = remaining_tps

        except Exception as e:
            logging.error(f"Error checking exit conditions: {str(e)}")
            raise

    def partial_close(self, i: int, exit_price: float, quantity: float, reason: str):
        """Close part of a trade"""
        try:
            self.current_trade.close_trade(
                exit_price=exit_price,
                exit_time=self.data.index[i],
                quantity=quantity
            )

            # Update total equity
            trade_profit = (exit_price - self.current_trade.entry_price) * quantity * 100000 \
                if self.current_trade.direction == 'long' \
                else (self.current_trade.entry_price - exit_price) * quantity * 100000

            self.total_equity += trade_profit

            logging.info(f"""
            Partial Close:
            Exit Price: {exit_price:.5f}
            Quantity: {quantity}
            Profit: ${trade_profit:.2f}
            Total Equity: ${self.total_equity:.2f}
            Remaining Quantity: {self.current_trade.remaining_quantity}
            """)

        except Exception as e:
            logging.error(f"Error in partial close: {str(e)}")
            raise

    def close_trade(self, i: int, reason: str):
        """Close entire trade"""
        try:
            exit_price = self.data.iloc[i]['close']
            remaining_quantity = self.current_trade.remaining_quantity

            self.current_trade.close_trade(
                exit_price=exit_price,
                exit_time=self.data.index[i],
                quantity=remaining_quantity
            )

            # Update total equity
            trade_profit = (exit_price - self.current_trade.entry_price) * remaining_quantity * 100000 \
                if self.current_trade.direction == 'long' \
                else (self.current_trade.entry_price - exit_price) * remaining_quantity * 100000

            self.total_equity += trade_profit

            logging.info(f"""
            Trade Closed ({reason}):
            Direction: {self.current_trade.direction}
            Entry Price: {self.current_trade.entry_price:.5f}
            Exit Price: {exit_price:.5f}
            Quantity: {remaining_quantity}
            Profit: ${trade_profit:.2f}
            Total Equity: ${self.total_equity:.2f}
            """)

        except Exception as e:
            logging.error(f"Error closing trade: {str(e)}")
            raise

    def calculate_results(self) -> Dict:
        """Calculate backtest results"""
        try:
            if not self.trades:
                logging.warning("No trades executed during backtest")
                return self._empty_results()

            profits = [trade.profit for trade in self.trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]

            total_trades = len(self.trades)
            winning_trades_count = len(winning_trades)
            total_profit = sum(profits)
            max_drawdown = self._calculate_max_drawdown()

            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')

            results = {
                'net_profit': total_profit,
                'total_trades': total_trades,
                'win_rate': (winning_trades_count / total_trades * 100) if total_trades > 0 else 0,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'average_trade': total_profit / total_trades if total_trades > 0 else 0,
                'sharpe_ratio': self._calculate_sharpe_ratio(profits),
                'recovery_factor': abs(total_profit / max_drawdown) if max_drawdown != 0 else float('inf'),
                'expectancy': self._calculate_expectancy(winning_trades, losing_trades),
                'trades_executed_long': len([t for t in self.trades if t.direction == 'long']),
                'trades_executed_short': len([t for t in self.trades if t.direction == 'short']),
                'average_win': avg_win,
                'average_loss': avg_loss,
                'largest_win': max(winning_trades) if winning_trades else 0,
                'largest_loss': min(losing_trades) if losing_trades else 0,
                'consecutive_wins': self._calculate_max_consecutive(profits, True),
                'consecutive_losses': self._calculate_max_consecutive(profits, False)
            }

            logging.info(f"Results calculated successfully: {json.dumps(results, indent=2)}")
            return results

        except Exception as e:
            logging.error(f"Error calculating results: {str(e)}")
            return self._empty_results()

    def _calculate_max_consecutive(self, profits: List[float], count_wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_streak = current_streak = 0
        for profit in profits:
            if (profit > 0 and count_wins) or (profit < 0 and not count_wins):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        try:
            if not self.equity_curve:
                return 0

            peak = self.equity_curve[0]
            max_dd = 0

            for value in self.equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100 if peak != 0 else 0
                max_dd = max(max_dd, dd)

            return max_dd

        except Exception as e:
            logging.error(f"Error calculating max drawdown: {str(e)}")
            return 0

    def _calculate_sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not profits:
                return 0

            returns = pd.Series(profits)
            excess_returns = returns.mean() - risk_free_rate/252  # Daily risk-free rate
            std_dev = returns.std()

            if std_dev == 0:
                return 0

            return excess_returns / std_dev * np.sqrt(252)  # Annualized

        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0

    def _calculate_expectancy(self, winning_trades: List[float], losing_trades: List[float]) -> float:
        """Calculate system expectancy"""
        try:
            total_trades = len(winning_trades) + len(losing_trades)
            if total_trades == 0:
                return 0

            win_probability = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0

            if avg_loss == 0:
                return 0

            return (win_probability * avg_win / avg_loss) - (1 - win_probability)

        except Exception as e:
            logging.error(f"Error calculating expectancy: {str(e)}")
            return 0

    def _empty_results(self) -> Dict:
        """Return empty results when no trades were taken"""
        return {
            'net_profit': 0,
            'total_trades': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'average_trade': 0,
            'sharpe_ratio': 0,
            'recovery_factor': 0,
            'expectancy': 0,
            'trades_executed_long': 0,
            'trades_executed_short': 0,
            'average_win': 0,
            'average_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }

def get_start_date(timeframe: str, end_date: datetime) -> datetime:
    """Calculate appropriate start date based on timeframe"""
    required_bars = 169 + 50  # Largest EMA period + buffer

    if timeframe == "M15":
        return end_date - timedelta(days=5)  # More than enough for M15
    elif timeframe == "H1":
        return end_date - timedelta(days=14)  # About 2 weeks
    elif timeframe == "H4":
        return end_date - timedelta(days=45)  # About 1.5 months
    elif timeframe == "D1":
        return end_date - timedelta(days=250)  # About 8 months
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

class MultiTimeframeOptimizer:
    def __init__(self,
                 start_date: datetime,
                 end_date: datetime,
                 symbol: str = "XAUUSD",
                 base_path: str = "optimization_results"):

        self.end_date = end_date
        self.symbol = symbol
        self.base_path = Path(base_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = self.base_path / self.run_id
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Set up file logging
        file_handler = logging.FileHandler(self.results_path / "optimization.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)

        self.data_cache = DataCache()

        # Define timeframes and their properties
        self.timeframes = {
            "M15": {"mt5_tf": mt5.TIMEFRAME_M15, "lookback_multiplier": 4},
            "H1": {"mt5_tf": mt5.TIMEFRAME_H1, "lookback_multiplier": 1},
            "H4": {"mt5_tf": mt5.TIMEFRAME_H4, "lookback_multiplier": 0.25},
            "D1": {"mt5_tf": mt5.TIMEFRAME_D1, "lookback_multiplier": 0.0416}
        }

        # Initialize MT5 with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not mt5.initialize():
                    if attempt == max_retries - 1:
                        raise Exception("MetaTrader5 initialization failed")
                    time.sleep(1)
                    continue
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"MetaTrader5 initialization failed after {max_retries} attempts: {str(e)}")
                time.sleep(1)

        # Set dynamic start dates for each timeframe
        self.start_dates = {
            tf: get_start_date(tf, end_date) for tf in self.timeframes
        }

        logging.info(f"""
        Optimizer Initialized:
        Symbol: {symbol}
        End Date: {end_date.date()}
        Start Dates:
        {chr(10).join(f'- {tf}: {date.date()}' for tf, date in self.start_dates.items())}
        Results Path: {self.results_path}
        """)

    def get_parameter_ranges(self, timeframe: str, phase: str = "initial") -> Dict:
        """Get parameter ranges based on timeframe and optimization phase"""
        multiplier = self.timeframes[timeframe]["lookback_multiplier"]

        if phase == "initial":
            return {
                'lookback_values': [
                    int(50 * multiplier),
                    int(100 * multiplier),
                    int(150 * multiplier)
                ],
                'threshold_values': [0.001, 0.002, 0.003],
                'tp_combinations': [
                    {"levels": [0.2, 0.4, 0.6, 0.8], "quantities": [0.6, 0.15, 0.15, 0.1]},
                    {"levels": [0.15, 0.3, 0.45, 0.6], "quantities": [0.4, 0.3, 0.2, 0.1]},
                    {"levels": [0.25, 0.5, 0.75, 1.0], "quantities": [0.7, 0.1, 0.1, 0.1]}
                ],
                'atr_multiplier_values': [1.5, 2.5]
            }
        else:  # fine-tuning phase
            return self.current_fine_tuning_ranges

    def generate_configurations(self, timeframe: str, parameter_ranges: Dict) -> List[Dict]:
        """Generate parameter combinations for testing with validation"""
        configs = []
        max_lookback = int(200 * self.timeframes[timeframe]["lookback_multiplier"])

        for lookback in parameter_ranges['lookback_values']:
            # Validate lookback isn't too large for timeframe
            if lookback > max_lookback:
                logging.warning(f"Skipping invalid lookback {lookback} for {timeframe}")
                continue

            for threshold in parameter_ranges['threshold_values']:
                for tp_combo in parameter_ranges['tp_combinations']:
                    # Validate take profit configuration
                    if not self._validate_tp_config(tp_combo):
                        continue

                    for atr_mult in parameter_ranges['atr_multiplier_values']:
                        config = {
                            "currency_pair": self.symbol,
                            "timeframe": timeframe,
                            "wavy_ema": 34,
                            "tunnel_ema1": 144,
                            "tunnel_ema2": 169,
                            "atr_period": 14,
                            "rsi_period": 14,
                            "rsi_upper": 70,
                            "rsi_lower": 30,
                            "apply_rsi_filter": True,
                            "lot_size": 0.01,
                            "enable_second_strategy": True,
                            "max_allow_into_zone": 0.25,
                            "peak_dip_lookback": lookback,
                            "peak_dip_threshold": threshold,
                            "tp_levels": tp_combo["levels"],
                            "tp_quantities": tp_combo["quantities"],
                            "tp_atr_multiplier": atr_mult
                        }
                        configs.append(config)

        logging.info(f"Generated {len(configs)} valid configurations for {timeframe}")
        return configs

    def _validate_tp_config(self, tp_combo: Dict) -> bool:
        """Validate take profit configuration"""
        try:
            levels = tp_combo["levels"]
            quantities = tp_combo["quantities"]

            if len(levels) != len(quantities):
                logging.error("Mismatched lengths in TP configuration")
                return False

            if not all(0 < x < 1 for x in quantities):
                logging.error("Invalid TP quantities (must be between 0 and 1)")
                return False

            if abs(sum(quantities) - 1.0) > 0.0001:
                logging.error("TP quantities must sum to 1")
                return False

            if not all(levels[i] < levels[i+1] for i in range(len(levels)-1)):
                logging.error("TP levels must be in ascending order")
                return False

            return True

        except Exception as e:
            logging.error(f"TP config validation failed: {str(e)}")
            return False
    def run_optimization_phase(self, timeframe: str, phase: str) -> List[OptimizationResult]:
        """Run a single optimization phase with enhanced error handling and progress tracking"""
        try:
            parameter_ranges = self.get_parameter_ranges(timeframe, phase)
            configs = self.generate_configurations(timeframe, parameter_ranges)
            results = []

            if not configs:
                logging.warning(f"No valid configurations generated for {timeframe}")
                return results

            logging.info(f"""
            Starting {phase} phase for {timeframe}:
            Configurations to test: {len(configs)}
            Parameter ranges: {json.dumps(parameter_ranges, indent=2)}
            """)

            # Get data for this timeframe with retry logic
            market_data = self.get_cached_data(timeframe)
            if market_data is None or len(market_data) == 0:
                logging.error(f"Failed to get valid market data for {timeframe}")
                return results

            # Run parallel optimization with enhanced progress tracking
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_config = {
                    executor.submit(
                        self.run_single_backtest,
                        config,
                        market_data
                    ): config for config in configs
                }

                completed = 0
                successful = 0
                failed = 0
                start_time = time.time()

                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        summary = future.result()
                        if summary:
                            results.append(OptimizationResult(config, summary, timeframe))
                            successful += 1
                        else:
                            failed += 1

                        completed += 1
                        if completed % 10 == 0:
                            elapsed = time.time() - start_time
                            remaining = (len(configs) - completed) * (elapsed / completed)
                            success_rate = (successful / completed * 100) if completed > 0 else 0

                            logging.info(f"""
                            Progress Update ({timeframe} - {phase}):
                            Completed: {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)
                            Successful: {successful} ({success_rate:.1f}%)
                            Failed: {failed}
                            Elapsed Time: {elapsed/60:.1f} minutes
                            Estimated Remaining: {remaining/60:.1f} minutes
                            """)

                    except Exception as e:
                        failed += 1
                        logging.error(f"""
                        Backtest failed:
                        Config: {json.dumps(config, indent=2)}
                        Error: {str(e)}
                        """)

            logging.info(f"""
            Phase Complete:
            Timeframe: {timeframe}
            Phase: {phase}
            Total Configurations: {len(configs)}
            Successful: {successful}
            Failed: {failed}
            Success Rate: {(successful/len(configs)*100):.1f}%
            """)

            return results

        except Exception as e:
            logging.error(f"Optimization phase failed: {str(e)}")
            return []

    def run_phased_optimization(self):
        """Run multi-timeframe phased optimization with enhanced error handling"""
        all_results = {}
        total_timeframes = len(self.timeframes)
        completed_timeframes = 0

        try:
            for timeframe in self.timeframes:
                completed_timeframes += 1
                logging.info(f"""
                Starting optimization for {timeframe} ({completed_timeframes}/{total_timeframes})
                """)

                # Phase 1: Initial broad optimization
                logging.info(f"Running initial phase for {timeframe}")
                phase1_results = self.run_optimization_phase(timeframe, "initial")

                if not phase1_results:
                    logging.warning(f"No valid results for {timeframe} initial phase")
                    continue

                # Get top performers from phase 1
                top_configs = self.get_top_performers(phase1_results, n=3)

                # Phase 2: Fine-tuning around best performers
                logging.info(f"Running fine-tuning phase for {timeframe}")
                phase2_results = []

                for idx, base_config in enumerate(top_configs, 1):
                    logging.info(f"Fine-tuning around config {idx}/3")
                    fine_tuning_ranges = self.generate_fine_tuning_ranges(base_config)
                    self.current_fine_tuning_ranges = fine_tuning_ranges
                    results = self.run_optimization_phase(timeframe, "fine_tuning")
                    phase2_results.extend(results)

                if phase2_results:
                    # Store results for this timeframe
                    all_results[timeframe] = {
                        'initial_phase': phase1_results,
                        'fine_tuning_phase': phase2_results,
                        'best_config': self.get_top_performers(phase2_results, n=1)[0]
                    }

                    # Save intermediate results
                    self.save_timeframe_results(timeframe, all_results[timeframe])

                logging.info(f"Completed optimization for {timeframe}")

            # Save and display final results
            if all_results:
                self.save_results(all_results)
                self.display_results(all_results)
                self.summarize_optimization(all_results)
            else:
                logging.error("No valid results for any timeframe")

        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
        finally:
            self.cleanup()

        return all_results

    def cleanup(self):
        """Clean up resources"""
        try:
            self.data_cache.clear()
            gc.collect()
            logging.info("Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Cleanup failed: {str(e)}")

    def save_timeframe_results(self, timeframe: str, results: Dict):
        """Save results for individual timeframe"""
        try:
            timeframe_path = self.results_path / f"{timeframe}_results.json"
            with open(timeframe_path, 'w') as f:
                json.dump(results, f, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
            logging.info(f"Saved results for {timeframe} to {timeframe_path}")
        except Exception as e:
            logging.error(f"Failed to save timeframe results: {str(e)}")

    def summarize_optimization(self, results: Dict):
        """Print summary statistics for optimization run"""
        try:
            total_configs = 0
            total_trades = 0
            best_profit = float('-inf')
            best_timeframe = None

            print("\n=== OPTIMIZATION SUMMARY ===")

            for timeframe, data in results.items():
                configs = len(data.get('initial_phase', [])) + len(data.get('fine_tuning_phase', []))
                total_configs += configs

                if 'best_config' in data:
                    result = next((r for r in data['fine_tuning_phase']
                                if r.config == data['best_config']), None)
                    if result:
                        if result.net_profit > best_profit:
                            best_profit = result.net_profit
                            best_timeframe = timeframe

                        print(f"\n{timeframe} Results:")
                        print(f"Net Profit: ${result.net_profit:,.2f}")
                        print(f"Win Rate: {result.win_rate:.1f}%")
                        print(f"Total Trades: {result.total_trades}")

            print(f"\nOverall Statistics:")
            print(f"Total Configurations Tested: {total_configs}")
            print(f"Best Performing Timeframe: {best_timeframe}")
            print(f"Best Net Profit: ${best_profit:,.2f}")

        except Exception as e:
            logging.error(f"Failed to generate summary: {str(e)}")

def main():
    try:
        # Set date range for optimization
        end_date = datetime.now()
        symbol = "XAUUSD"

        logging.info(f"""
        Starting optimization:
        Symbol: {symbol}
        End Date: {end_date.date()}
        """)

        optimizer = MultiTimeframeOptimizer(
            start_date=None,  # Will be set dynamically per timeframe
            end_date=end_date,
            symbol=symbol
        )

        results = optimizer.run_phased_optimization()

        if results:
            logging.info("Optimization completed successfully")
        else:
            logging.warning("Optimization completed but no valid results were obtained")

    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}", exc_info=True)
        print(f"Optimization failed: {str(e)}")
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")

if __name__ == "__main__":
    main()