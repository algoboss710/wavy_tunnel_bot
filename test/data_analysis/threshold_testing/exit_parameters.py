import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
from pathlib import Path
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from itertools import product
import time
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Emergency stop file path
STOP_FILE = "stop_optimization.txt"

# Global configurations
SYMBOL = "XAUUSD"
OPTIMIZATION_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]

# Optimized entry parameters for XAUUSD
TIMEFRAME_ENTRY_PARAMS = {
    "M5": {
        'wavy_period': 25,
        'tunnel_period1': 280,
        'tunnel_period2': 300,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.45000000000000007
    },
    "M15": {
        'wavy_period': 30,
        'tunnel_period1': 180,
        'tunnel_period2': 320,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.45000000000000007
    },
    "M30": {
        'wavy_period': 20,
        'tunnel_period1': 300,
        'tunnel_period2': 320,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.30000000000000004
    },
    "H1": {
        'wavy_period': 20,
        'tunnel_period1': 180,
        'tunnel_period2': 280,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.45000000000000007
    },
    "H4": {
        'wavy_period': 50,
        'tunnel_period1': 160,
        'tunnel_period2': 300,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.20000000000000004
    },
    "D1": {
        'wavy_period': 50,
        'tunnel_period1': 160,
        'tunnel_period2': 160,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.1
    }
}

def test_mt5_connection():
    """Test MT5 connection and basic functionality"""
    print("\nTesting MT5 Connection...")
    
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return False
        
    print("MT5 Package Version:", mt5.__version__)
    print("Terminal Info:", mt5.terminal_info())
    
    # Test symbol info
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is not None:
        print(f"Symbol: {SYMBOL}")
        print(f"Points: {symbol_info.point}")
        print(f"Digits: {symbol_info.digits}")
        print(f"Trade Mode: {symbol_info.trade_mode}")
    else:
        print(f"Failed to get symbol info for {SYMBOL}")
        return False
    
    account_info = mt5.account_info()
    if account_info is not None:
        print("Connected to account:", account_info.login)
        print("Server:", account_info.server)
    else:
        print("Failed to get account info")
        return False
    
    return True

class OptimizationLogger:
    """Enhanced logging system for optimization process"""
    def __init__(self, base_path: Path, symbol: str, timeframe: str):
        self.base_path = base_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.log_path = base_path / f"{symbol}_{timeframe}"
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Setup loggers
        self.setup_loggers()

    def setup_loggers(self):
        # Main optimization logger
        self.main_logger = self._setup_logger('main', 'optimization.log')
        # Progress logger
        self.progress_logger = self._setup_logger('progress', 'progress.log')
        # Results logger
        self.results_logger = self._setup_logger('results', 'results.log')
        # Debug logger
        self.debug_logger = self._setup_logger('debug', 'debug.log')
        # Trades logger
        self.trades_logger = self._setup_logger('trades', 'trades.log')

    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(f"{self.symbol}_{self.timeframe}_{name}")
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_path / filename)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def log_params(self, params: Dict):
        self.main_logger.info(f"Starting optimization with parameters: {params}")

    def log_progress(self, current: int, total: int, elapsed_time: float, remaining_time: float,
                    best_result: Optional[Dict] = None):
        progress = (current / total) * 100
        msg = (f"Progress: {current}/{total} ({progress:.2f}%) - "
               f"Elapsed: {elapsed_time:.2f}s - "
               f"Remaining: {remaining_time:.2f}s")
        
        if best_result:
            msg += (f"\nBest so far - Trades: {best_result['total_trades']}, "
                   f"Win Rate: {best_result['win_rate']:.2f}%, "
                   f"Profit: {best_result['avg_profit']*100:.2f}%")
        
        self.progress_logger.info(msg)

    def log_debug(self, message: str):
        self.debug_logger.debug(message)

    def log_results(self, results: Dict):
        self.results_logger.info(f"Optimization Results:\n{results}")

class WavyTunnelExitOptimizer:
    def __init__(self, symbol: str, timeframe: str, start_date: datetime, 
                 end_date: datetime, entry_params: Dict, 
                 base_path: str = "exit_optimization_results"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.entry_params = entry_params
        self.base_path = Path(base_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Get symbol info and store necessary properties
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Could not get symbol info for {symbol}")
        
        # Store only the necessary properties
        self.point = symbol_info.point
        self.digits = symbol_info.digits
        self.trade_mode = symbol_info.trade_mode

        # Initialize logger
        self.logger = OptimizationLogger(self.base_path, symbol, timeframe)

        # Parameter ranges for optimization
        self.param_ranges = self._setup_param_ranges()

    def _setup_param_ranges(self) -> Dict:
        """Define parameter ranges with significantly reduced combinations"""
        return {
            # Take profit distribution parameters - minimal steps
            'tp1_lot_percent': [40, 50, 60],        # 3 values
            'tp2_lot_percent': [20, 25, 30],        # 3 values
            'tp3_lot_percent': [10, 15],            # 2 values
            'tp4_lot_percent': [10, 15],            # 2 values
            
            # Take profit distances calibrated for XAUUSD
            'tp1_weight': np.array([0.001, 0.002, 0.003]),  # 3 values
            'tp2_weight': np.array([0.004, 0.005, 0.006]),  # 3 values
            'tp3_weight': np.array([0.007, 0.008, 0.009]),  # 3 values
            'tp4_weight': np.array([0.010, 0.011, 0.012]),  # 3 values
            
            # Stop loss parameters - minimal values
            'wave_cross_buffer': np.array([0.001, 0.002]),  # 2 values
            'tunnel_touch_buffer': np.array([0.001, 0.002]) # 2 values
        }

    def _validate_params(self, params: Dict) -> bool:
        """Validate parameter combinations with stricter rules"""
        try:
            # Check lot percentages sum to 100%
            lot_sum = (params['tp1_lot_percent'] + params['tp2_lot_percent'] + 
                      params['tp3_lot_percent'] + params['tp4_lot_percent'])
            if lot_sum != 100:
                return False

            # Validate take profit weights are properly ordered
            weights = [
                params['tp1_weight'],
                params['tp2_weight'],
                params['tp3_weight'],
                params['tp4_weight']
            ]
            
            # Check ascending order with minimum spacing
            if not all(weights[i] + 0.001 < weights[i+1] for i in range(len(weights)-1)):
                return False

            return True

        except Exception as e:
            self.logger.debug_logger.error(f"Parameter validation error: {str(e)}")
            return False

    def test_data_and_signals(self, start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Test function to verify data fetching and signal generation"""
        print("\nRunning diagnostic test...")
        
        # Test data fetching
        print("1. Testing MT5 data fetch...")
        data = self._get_market_data()
        if data is not None:
            print(f"✓ Successfully fetched {len(data)} bars of data")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            print("\nSample data:")
            print(data.head())
            self.logger.debug_logger.info(f"Data fetch successful. Shape: {data.shape}")
        else:
            print("✗ Failed to fetch data")
            self.logger.debug_logger.error("Data fetch failed")
            return None, None, None, None
        
        # Test signal generation
        print("\n2. Testing signal generation...")
        long_signals, short_signals, is_primary = self.generate_signals(data)
        
        print(f"Long signals found: {long_signals.sum()}")
        print(f"Short signals found: {short_signals.sum()}")
        print(f"Primary signals found: {is_primary.sum()}")
        
        self.logger.debug_logger.info(
            f"Signal generation test - Longs: {long_signals.sum()}, "
            f"Shorts: {short_signals.sum()}, Primary: {is_primary.sum()}"
        )
        
        if long_signals.sum() > 0 or short_signals.sum() > 0:
            print("\nSignal dates:")
            signal_dates = data.index[long_signals | short_signals]
            for date in signal_dates[:5]:  # Show first 5 signals
                print(f"Signal at: {date}")
        
        # Test parameter generation
        print("\n3. Testing parameter combinations...")
        params = self._generate_param_combinations()
        print(f"Generated {len(params)} valid parameter combinations")
        if params:
            print("\nSample parameter set:")
            print(params[0])
            self.logger.debug_logger.info(f"Parameter combinations generated: {len(params)}")
        
        # Test single evaluation
        print("\n4. Testing parameter evaluation...")
        if params:
            test_results = self._evaluate_params(params[0], data)
            print("Evaluation results:")
            print(test_results)
            self.logger.debug_logger.info(f"Test evaluation results: {test_results}")

        return data, long_signals, short_signals, is_primary

    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch and preprocess market data with enhanced error handling"""
        try:
            if not mt5.initialize():
                self.logger.main_logger.error("Failed to initialize MT5")
                return None

            timeframe = getattr(mt5, f"TIMEFRAME_{self.timeframe}")
            rates = mt5.copy_rates_range(self.symbol, timeframe,
                                       self.start_date, self.end_date)
            
            if rates is None or len(rates) == 0:
                self.logger.main_logger.error(f"No data available for {self.symbol} {self.timeframe}")
                return None

            # Convert to DataFrame and handle timezone
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Basic data validation
            required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.main_logger.error("Missing required columns in data")
                return None
            
            return self._preprocess_data(df)
            
        except Exception as e:
            self.logger.main_logger.error(f"Error fetching data: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data with improved cleaning"""
        try:
            # Remove any zero or negative prices
            for col in ['open', 'high', 'low', 'close']:
                data = data[data[col] > 0]

            # Handle missing values
            data = data.fillna(method='ffill')
            
            # Remove outliers using rolling median
            window = 5
            for col in ['high', 'low', 'close', 'open']:
                median = data[col].rolling(window=window, center=True).median()
                std = data[col].rolling(window=window, center=True).std()
                
                # More conservative outlier threshold for XAUUSD
                threshold = 2.5
                data[col] = data[col].where(
                    abs(data[col] - median) <= threshold * std,
                    median
                )
            
            # Validate high/low relationships
            data['high'] = data[['high', 'open', 'close']].max(axis=1)
            data['low'] = data[['low', 'open', 'close']].min(axis=1)
            
            self.logger.debug_logger.info(
                f"Preprocessed data shape: {data.shape}, "
                f"Date range: {data.index[0]} to {data.index[-1]}"
            )
            
            return data

        except Exception as e:
            self.logger.main_logger.error(f"Error preprocessing data: {str(e)}")
            return data

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate entry signals using fixed parameters with improved validation"""
        try:
            # Calculate EMAs using entry parameters
            wavy_h = data['high'].ewm(span=self.entry_params['wavy_period'], adjust=False).mean()
            wavy_c = data['close'].ewm(span=self.entry_params['wavy_period'], adjust=False).mean()
            wavy_l = data['low'].ewm(span=self.entry_params['wavy_period'], adjust=False).mean()
            tunnel1 = data['close'].ewm(span=self.entry_params['tunnel_period1'], adjust=False).mean()
            tunnel2 = data['close'].ewm(span=self.entry_params['tunnel_period2'], adjust=False).mean()

            wavy_max = pd.concat([wavy_h, wavy_c, wavy_l], axis=1).max(axis=1)
            wavy_min = pd.concat([wavy_h, wavy_c, wavy_l], axis=1).min(axis=1)
            tunnel_max = pd.concat([tunnel1, tunnel2], axis=1).max(axis=1)
            tunnel_min = pd.concat([tunnel1, tunnel2], axis=1).min(axis=1)

            # Store for later use in exit conditions
            self.wavy_max = wavy_max
            self.wavy_min = wavy_min
            self.tunnel_max = tunnel_max
            self.tunnel_min = tunnel_min

            # Primary signals with price validation
            primary_longs = (
                (data['open'] > wavy_max) & 
                (wavy_min > tunnel_max) & 
                (data['open'] > 0)
            )
            
            primary_shorts = (
                (data['open'] < wavy_min) & 
                (wavy_max < tunnel_min) & 
                (data['open'] > 0)
            )

            # Secondary signals with gap validation
            min_gap = self.entry_params['min_gap_second']
            max_zone = self.entry_params['max_zone_percentage']

            secondary_longs = (
                (data['close'].shift(1) <= wavy_max.shift(1)) & 
                (data['close'] > wavy_max) &
                (data['close'] < tunnel_min) &
                ((tunnel_min - data['close']) > min_gap) &
                ((data['close'] - wavy_max) / (tunnel_min - wavy_max) <= max_zone) &
                (data['close'] > 0)
            )
            
            secondary_shorts = (
                (data['close'].shift(1) >= wavy_min.shift(1)) & 
                (data['close'] < wavy_min) &
                (data['close'] > tunnel_max) &
                ((data['close'] - tunnel_max) > min_gap) &
                ((wavy_min - data['close']) / (wavy_min - tunnel_max) <= max_zone) &
                (data['close'] > 0)
            )

            # Log signal counts
            self.logger.debug_logger.info(
                f"Generated signals:\n"
                f"Primary Longs: {primary_longs.sum()}\n"
                f"Primary Shorts: {primary_shorts.sum()}\n"
                f"Secondary Longs: {secondary_longs.sum()}\n"
                f"Secondary Shorts: {secondary_shorts.sum()}"
            )

            return (
                primary_longs | secondary_longs,
                primary_shorts | secondary_shorts,
                pd.Series(primary_longs | primary_shorts, index=data.index)
            )

        except Exception as e:
            self.logger.main_logger.error(f"Error generating signals: {str(e)}")
            return (
                pd.Series(False, index=data.index),
                pd.Series(False, index=data.index),
                pd.Series(False, index=data.index)
            )

    def _evaluate_params(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Evaluate a single parameter combination with enhanced trade management"""
        try:
            long_signals, short_signals, is_primary = self.generate_signals(data)
            
            trades = []
            current_position = None
            entry_price = 0
            entry_time = None
            remaining_position = 0
            
            for i in range(len(data)-1):
                # Skip if data is invalid
                if pd.isna(data['open'].iloc[i]) or data['open'].iloc[i] <= 0:
                    continue
                
                # Entry logic
                if current_position is None:
                    if long_signals.iloc[i]:
                        current_position = 'long'
                        entry_price = data['open'].iloc[i]
                        entry_time = data.index[i]
                        remaining_position = 100
                        self.logger.debug_logger.info(
                            f"Entry LONG at {entry_price} - "
                            f"{'Primary' if is_primary.iloc[i] else 'Secondary'}"
                        )
                        
                    elif short_signals.iloc[i]:
                        current_position = 'short'
                        entry_price = data['open'].iloc[i]
                        entry_time = data.index[i]
                        remaining_position = 100
                        self.logger.debug_logger.info(
                            f"Entry SHORT at {entry_price} - "
                            f"{'Primary' if is_primary.iloc[i] else 'Secondary'}"
                        )
                
                # Exit logic for open positions
                elif remaining_position > 0:
                    if current_position == 'long':
                        # Check take profits
                        for j, (lot_percent, weight) in enumerate(zip(
                            [params['tp1_lot_percent'], params['tp2_lot_percent'],
                             params['tp3_lot_percent'], params['tp4_lot_percent']],
                            [params['tp1_weight'], params['tp2_weight'],
                             params['tp3_weight'], params['tp4_weight']]
                        )):
                            if remaining_position >= lot_percent:
                                tp_level = entry_price * (1 + weight)
                                if data['high'].iloc[i] >= tp_level:
                                    trades.append({
                                        'entry_price': entry_price,
                                        'exit_price': tp_level,
                                        'position_size': lot_percent,
                                        'profit': (tp_level - entry_price) / entry_price,
                                        'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                        'exit_reason': f'tp{j+1}',
                                        'is_primary': is_primary.iloc[i],
                                        'position': current_position
                                    })
                                    remaining_position -= lot_percent
                                    self.logger.debug_logger.info(
                                        f"LONG TP{j+1} hit - Size: {lot_percent}% at {tp_level}"
                                    )
                        
                        # Check stop loss for remaining position
                        if remaining_position > 0:
                            sl_level = entry_price * (1 - params['wave_cross_buffer'])
                            if data['low'].iloc[i] <= sl_level:
                                trades.append({
                                    'entry_price': entry_price,
                                    'exit_price': data['close'].iloc[i],  # Use close for SL
                                    'position_size': remaining_position,
                                    'profit': (data['close'].iloc[i] - entry_price) / entry_price,
                                    'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                    'exit_reason': 'sl',
                                    'is_primary': is_primary.iloc[i],
                                    'position': current_position
                                })
                                remaining_position = 0
                                self.logger.debug_logger.info(
                                    f"LONG SL hit at {data['close'].iloc[i]}"
                                )
                    
                    elif current_position == 'short':
                        # Check take profits
                        for j, (lot_percent, weight) in enumerate(zip(
                            [params['tp1_lot_percent'], params['tp2_lot_percent'],
                             params['tp3_lot_percent'], params['tp4_lot_percent']],
                            [params['tp1_weight'], params['tp2_weight'],
                             params['tp3_weight'], params['tp4_weight']]
                        )):
                            if remaining_position >= lot_percent:
                                tp_level = entry_price * (1 - weight)
                                if data['low'].iloc[i] <= tp_level:
                                    trades.append({
                                        'entry_price': entry_price,
                                        'exit_price': tp_level,
                                        'position_size': lot_percent,
                                        'profit': (entry_price - tp_level) / entry_price,
                                        'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                        'exit_reason': f'tp{j+1}',
                                        'is_primary': is_primary.iloc[i],
                                        'position': current_position
                                    })
                                    remaining_position -= lot_percent
                                    self.logger.debug_logger.info(
                                        f"SHORT TP{j+1} hit - Size: {lot_percent}% at {tp_level}"
                                    )
                        
                        # Check stop loss for remaining position
                        if remaining_position > 0:
                            sl_level = entry_price * (1 + params['wave_cross_buffer'])
                            if data['high'].iloc[i] >= sl_level:
                                trades.append({
                                    'entry_price': entry_price,
                                    'exit_price': data['close'].iloc[i],
                                    'position_size': remaining_position,
                                    'profit': (entry_price - data['close'].iloc[i]) / entry_price,
                                    'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                    'exit_reason': 'sl',
                                    'is_primary': is_primary.iloc[i],
                                    'position': current_position
                                })
                                remaining_position = 0
                                self.logger.debug_logger.info(
                                    f"SHORT SL hit at {data['close'].iloc[i]}"
                                )
                    
                    # Reset position if fully closed
                    if remaining_position == 0:
                        current_position = None
                        entry_price = 0
                        entry_time = None

            # Calculate performance metrics
            if trades:
                df_trades = pd.DataFrame(trades)
        
                # Calculate cumulative and per-trade metrics
                total_profit = df_trades['profit'].sum()
                avg_profit_per_trade = df_trades['profit'].mean()
        
                results = {
                    'total_trades': len(df_trades),
                    'profitable_trades': len(df_trades[df_trades['profit'] > 0]),
                    'total_profit_pct': total_profit * 100,  # Total cumulative profit
                    'avg_profit_per_trade_pct': avg_profit_per_trade * 100,  # Average per trade
                    'max_drawdown': self._calculate_max_drawdown(df_trades['profit']),
                    'win_rate': len(df_trades[df_trades['profit'] > 0]) / len(df_trades),
                    'profit_factor': self._calculate_profit_factor(df_trades['profit']),
                    'sharpe_ratio': self._calculate_sharpe_ratio(df_trades['profit']),
                    'avg_hold_time': df_trades['hold_time'].mean(),
                    'primary_profit_per_trade': df_trades[df_trades['is_primary']]['profit'].mean() * 100,
                    'secondary_profit_per_trade': df_trades[~df_trades['is_primary']]['profit'].mean() * 100
            }
            else:
                results = self._get_default_results()

            return results

        except Exception as e:
            self.logger.main_logger.error(f"Error evaluating parameters: {str(e)}")
            return self._get_default_results()
        
    def _get_default_results(self) -> Dict:
        """Return default results dictionary for error cases"""
        return {
            'total_trades': 0,
            'profitable_trades': 0,
            'avg_profit': 0.0,
            'max_drawdown': 1.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'avg_hold_time': 0.0,
            'primary_profit': 0.0,
            'secondary_profit': 0.0
        }

    def _calculate_max_drawdown(self, profits: pd.Series) -> float:
        """Calculate maximum drawdown from a series of profits"""
        try:
            cumulative = (1 + profits).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = cumulative / running_max - 1
            return abs(drawdowns.min())
        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating max drawdown: {str(e)}")
            return 1.0

    def _calculate_profit_factor(self, profits: pd.Series) -> float:
        """Calculate profit factor with safety checks"""
        try:
            wins = profits[profits > 0].sum()
            losses = abs(profits[profits < 0].sum())
            return wins / losses if losses != 0 else float('inf')
        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating profit factor: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(self, profits: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio with safety checks"""
        try:
            if len(profits) < 2:
                return 0.0
            excess_returns = profits - risk_free_rate
            return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0.0
        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate parameter combinations with validation"""
        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        
        combinations = []
        total_attempted = 0
        valid_count = 0
        
        for values in product(*param_values):
            total_attempted += 1
            params = dict(zip(param_keys, values))
            
            if self._validate_params(params):
                combinations.append(params)
                valid_count += 1
        
        self.logger.debug_logger.info(
            f"Parameter generation: {valid_count} valid combinations "
            f"from {total_attempted} attempted"
        )
        
        return combinations

def optimize_parallel(self) -> Tuple[Dict, pd.DataFrame]:
    """Run parallel optimization process with aligned metrics"""
    try:
        data = self._get_market_data()
        if data is None:
            return None, None

        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        
        self.logger.main_logger.info(f"Starting optimization with {total_combinations} parameter combinations")
        
        num_cores = mp.cpu_count()
        chunk_size = max(1, min(1000, total_combinations // (num_cores * 4)))
        chunks = [param_combinations[i:i + chunk_size] 
                 for i in range(0, len(param_combinations), chunk_size)]
        
        start_time = time.time()
        results = []
        best_result = None
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {
                executor.submit(self._process_chunk, chunk, data.copy()): i 
                for i, chunk in enumerate(chunks)
            }
            
            completed = 0
            try:
                for future in concurrent.futures.as_completed(futures):
                    # Check for emergency stop
                    if os.path.exists(STOP_FILE):
                        self.logger.main_logger.warning("Stop file detected, stopping optimization...")
                        executor.shutdown(wait=False)
                        return None, None
                        
                    chunk_results = future.result(timeout=300)  # 5-minute timeout
                    if chunk_results:  # Only process if we got results
                        results.extend(chunk_results)
                        
                        completed += len(chunk_results)  # Update based on actual results
                        elapsed_time = time.time() - start_time
                        progress = min(100, (completed / total_combinations) * 100)
                        
                        # Update best result based on total profit
                        chunk_df = pd.DataFrame(chunk_results)
                        if not chunk_df.empty:
                            chunk_best = chunk_df.nlargest(1, 'total_profit_pct').iloc[0].to_dict()
                            if best_result is None or chunk_best['total_profit_pct'] > best_result['total_profit_pct']:
                                best_result = chunk_best
                        
                        # Estimate remaining time
                        remaining_time = (elapsed_time / progress) * (100 - progress) if progress > 0 else 0
                        
                        self.logger.log_progress(completed, total_combinations, 
                                               elapsed_time, remaining_time,
                                               best_result)
                        
            except KeyboardInterrupt:
                self.logger.main_logger.warning("Optimization interrupted by user")
                executor.shutdown(wait=False)
                return None, None
                
            except concurrent.futures.TimeoutError:
                self.logger.main_logger.error("Processing timeout")
                executor.shutdown(wait=False)
                return None, None

        # Process final results
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            # Use total_profit_pct for final ranking
            best_params = results_df.nlargest(1, 'total_profit_pct').iloc[0].to_dict()
            self.logger.log_results(best_params)
            
            # Save all results
            results_df.to_csv(self.base_path / f"{self.symbol}_{self.timeframe}_all_results.csv")
            return best_params, results_df
        else:
            self.logger.main_logger.warning("No valid results found")
            return None, None

    except Exception as e:
        self.logger.main_logger.error(f"Error during optimization: {str(e)}")
        return None, None
    def _process_chunk(self, chunk: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Process a chunk of parameter combinations with validation"""
        chunk_results = []
        for params in chunk:
            try:
                if self._validate_params(params):
                    evaluation = self._evaluate_params(params, data)
                    if evaluation['total_trades'] > 0:
                        chunk_results.append({**params, **evaluation})
                
            except Exception as e:
                self.logger.main_logger.error(f"Error processing parameters {params}: {str(e)}")
        return chunk_results
    
    def create_optimization_report(self, results_df: pd.DataFrame):
        """Create comprehensive optimization report with visualizations"""
        if results_df is None or len(results_df) == 0:
            self.logger.main_logger.warning("No results to create report")
            return

        report_path = self.base_path / f"{self.symbol}_{self.timeframe}_exit_report"
        report_path.mkdir(exist_ok=True)

        try:
            # Create visualizations
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Profit Distribution',
                    'Sharpe Ratio vs Win Rate',
                    'Take Profit Level Distribution',
                    'Hold Time vs Profit',
                    'Drawdown vs Profit Factor',
                    'Trade Count Distribution'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            # Profit distribution
            fig.add_trace(
                go.Histogram(x=results_df['avg_profit'].multiply(100),
                            name='Average Profit %',
                            nbinsx=50),
                row=1, col=1
            )

            # Sharpe vs Win Rate
            fig.add_trace(
                go.Scatter(x=results_df['sharpe_ratio'],
                          y=results_df['win_rate'].multiply(100),
                          mode='markers',
                          name='Risk-Return',
                          marker=dict(
                              color=results_df['profit_factor'],
                              colorscale='Viridis',
                              showscale=True,
                              colorbar=dict(title='Profit Factor')
                          )),
                row=1, col=2
            )

            # Take profit level distribution
            tp_levels = pd.DataFrame({
                'TP1': results_df['tp1_weight'].multiply(100),
                'TP2': results_df['tp2_weight'].multiply(100),
                'TP3': results_df['tp3_weight'].multiply(100),
                'TP4': results_df['tp4_weight'].multiply(100)
            })
            
            fig.add_trace(
                go.Box(x=tp_levels.values.flatten(),
                      name='TP Levels (%)'),
                row=2, col=1
            )

            # Hold time vs Profit
            fig.add_trace(
                go.Scatter(x=results_df['avg_hold_time'],
                          y=results_df['avg_profit'].multiply(100),
                          mode='markers',
                          name='Hold Time Impact',
                          marker=dict(
                              color=results_df['total_trades'],
                              colorscale='Viridis',
                              showscale=True,
                              colorbar=dict(title='Total Trades')
                          )),
                row=2, col=2
            )

            # Drawdown vs Profit Factor
            fig.add_trace(
                go.Scatter(x=results_df['max_drawdown'].multiply(100),
                          y=results_df['profit_factor'],
                          mode='markers',
                          name='Risk-Reward',
                          marker=dict(
                              color=results_df['sharpe_ratio'],
                              colorscale='Viridis',
                              showscale=True,
                              colorbar=dict(title='Sharpe Ratio')
                          )),
                row=3, col=1
            )

            # Trade Count Distribution
            fig.add_trace(
                go.Histogram(x=results_df['total_trades'],
                            name='Trade Count',
                            nbinsx=50),
                row=3, col=2
            )

            # Update layout
            fig.update_layout(
                height=1500,
                width=1200,
                showlegend=True,
                title_text=f"Exit Strategy Optimization Results - {self.symbol} {self.timeframe}"
            )

            # Update axes labels
            fig.update_xaxes(title_text="Profit (%)", row=1, col=1)
            fig.update_xaxes(title_text="Sharpe Ratio", row=1, col=2)
            fig.update_xaxes(title_text="Take Profit Level (%)", row=2, col=1)
            fig.update_xaxes(title_text="Hold Time (hours)", row=2, col=2)
            fig.update_xaxes(title_text="Max Drawdown (%)", row=3, col=1)
            fig.update_xaxes(title_text="Number of Trades", row=3, col=2)

            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_yaxes(title_text="Average Profit (%)", row=2, col=2)
            fig.update_yaxes(title_text="Profit Factor", row=3, col=1)
            fig.update_yaxes(title_text="Frequency", row=3, col=2)

            # Save plots
            fig.write_html(report_path / "optimization_results.html")
            
            # Save top results to CSV with proper formatting
            top_results = results_df.nlargest(20, 'composite_score').copy()
            for col in ['avg_profit', 'win_rate', 'max_drawdown']:
                top_results[col] = top_results[col].multiply(100)
            top_results.to_csv(report_path / "top_results.csv")

            # Create detailed summary report
            best_params = results_df.nlargest(1, 'composite_score').iloc[0]
            
            with open(report_path / "summary_report.txt", "w") as f:
                f.write(f"Exit Strategy Optimization Summary - {self.symbol} {self.timeframe}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Input Parameters:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Symbol: {self.symbol}\n")
                f.write(f"Timeframe: {self.timeframe}\n")
                f.write(f"Date Range: {self.start_date} to {self.end_date}\n")
                f.write("\nEntry Parameters Used:\n")
                for param, value in self.entry_params.items():
                    f.write(f"  {param}: {value}\n")
                
                f.write("\nBest Parameters Found:\n")
                f.write("-" * 20 + "\n")
                f.write("Take Profit Levels:\n")
                for i in range(1, 5):
                    f.write(f"TP{i}: {best_params[f'tp{i}_lot_percent']}% at {best_params[f'tp{i}_weight']*100:.2f}%\n")
                
                f.write(f"\nStop Loss Parameters:\n")
                f.write(f"Wave Cross Buffer: {best_params['wave_cross_buffer']*100:.3f}%\n")
                f.write(f"Tunnel Touch Buffer: {best_params['tunnel_touch_buffer']*100:.3f}%\n")
                
                f.write(f"\nPerformance Metrics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Trades: {best_params['total_trades']}\n")
                f.write(f"Win Rate: {best_params['win_rate']*100:.1f}%\n")
                f.write(f"Average Profit: {best_params['avg_profit']*100:.2f}%\n")
                f.write(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}\n")
                f.write(f"Profit Factor: {best_params['profit_factor']:.2f}\n")
                f.write(f"Max Drawdown: {best_params['max_drawdown']*100:.2f}%\n")
                f.write(f"Average Hold Time: {best_params['avg_hold_time']:.1f} hours\n")
                
                f.write(f"\nStrategy Comparison:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Primary Strategy Profit: {best_params['primary_profit']*100:.2f}%\n")
                f.write(f"Secondary Strategy Profit: {best_params['secondary_profit']*100:.2f}%\n")

        except Exception as e:
            self.logger.main_logger.error(f"Error creating optimization report: {str(e)}")


def create_combined_summary_report(all_results: Dict, base_dir: Path, symbol: str):
    """Create a comprehensive summary report combining results from all timeframes"""
    summary_path = base_dir / "combined_summary.txt"
    
    with open(summary_path, "w") as f:
        f.write(f"Combined Summary Report for {symbol}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for timeframe, results in all_results.items():
            f.write(f"\nTimeframe: {timeframe}\n")
            f.write("-" * 40 + "\n")
            
            # Entry Parameters
            f.write("Entry Parameters:\n")
            for param, value in results['entry_params'].items():
                f.write(f"  {param}: {value}\n")
            
            # Exit Parameters
            f.write("\nOptimized Exit Parameters:\n")
            exit_params = results['exit_params']
            
            f.write("Take Profit Levels:\n")
            for i in range(1, 5):
                f.write(f"  TP{i}: {exit_params[f'tp{i}_lot_percent']}% "
                       f"at {exit_params[f'tp{i}_weight']*100:.2f}%\n")
            
            f.write("\nStop Loss Parameters:\n")
            f.write(f"  Wave Cross Buffer: {exit_params['wave_cross_buffer']*100:.3f}%\n")
            f.write(f"  Tunnel Touch Buffer: {exit_params['tunnel_touch_buffer']*100:.3f}%\n")
            
            f.write("\nPerformance Metrics:\n")
            f.write(f"  Total Trades: {exit_params['total_trades']}\n")
            f.write(f"  Win Rate: {exit_params['win_rate']*100:.1f}%\n")
            f.write(f"  Average Profit: {exit_params['avg_profit']*100:.2f}%\n")
            f.write(f"  Sharpe Ratio: {exit_params['sharpe_ratio']:.2f}\n")
            f.write(f"  Profit Factor: {exit_params['profit_factor']:.2f}\n")
            f.write(f"  Max Drawdown: {exit_params['max_drawdown']*100:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
def main():
    """Main execution function"""
    # Configuration
    symbol = SYMBOL
    start_date = datetime.now() - timedelta(days=350)
    end_date = datetime.now()
    
    print(f"\nRunning optimization for {symbol} from {start_date} to {end_date}")

    # Create results directory with timestamp
    base_results_dir = Path(f"exit_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # Summary of all timeframe results
    all_results = {}

    for timeframe in OPTIMIZATION_TIMEFRAMES:
        try:
            # Reinitialize MT5 for each timeframe
            if mt5.initialize():
                print(f"\nOptimizing exit strategy for {symbol} on {timeframe}")
                entry_params = TIMEFRAME_ENTRY_PARAMS[timeframe]
                print(f"Using entry parameters: {entry_params}")
                
                optimizer = WavyTunnelExitOptimizer(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    entry_params=entry_params,
                    base_path=str(base_results_dir / timeframe)
                )

                # Run diagnostic test
                data, long_signals, short_signals, is_primary = optimizer.test_data_and_signals(
                    start_date, end_date
                )

                if data is not None and (long_signals.sum() > 0 or short_signals.sum() > 0):
                    print("\nDiagnostic test successful - proceeding with optimization...")
                    best_params, results_df = optimizer.optimize_parallel()
                    
                    if best_params is not None:
                        print("\nBest Exit Parameters Found:")
                        print("-" * 30)
                        print("Take Profit Levels:")
                        for i in range(1, 5):
                            print(f"TP{i}: {best_params[f'tp{i}_lot_percent']}% at {best_params[f'tp{i}_weight']*100:.2f}%")
                        
                        print("\nPerformance Metrics:")
                        print(f"Total Trades: {best_params['total_trades']}")
                        print(f"Win Rate: {best_params['win_rate']*100:.1f}%")
                        print(f"Average Profit: {best_params['avg_profit']*100:.2f}%")
                        print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
                        print(f"Max Drawdown: {best_params['max_drawdown']*100:.2f}%")
                        
                        all_results[timeframe] = {
                            'entry_params': entry_params,
                            'exit_params': best_params
                        }
                        
                        optimizer.create_optimization_report(results_df)
                    else:
                        print(f"No valid results found for {symbol} {timeframe}")
                else:
                    print("Diagnostic test failed - no valid signals generated")

            else:
                print(f"Failed to initialize MT5 for {timeframe}")
                continue

        except Exception as e:
            print(f"Error optimizing {symbol} {timeframe}: {str(e)}")
            continue
        finally:
            mt5.shutdown()

    # Create combined summary report
    create_combined_summary_report(all_results, base_results_dir, symbol)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()
        print("\nExit strategy optimization completed")
