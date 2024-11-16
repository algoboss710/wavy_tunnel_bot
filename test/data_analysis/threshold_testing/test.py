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
import json
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Added connection monitoring flags
MT5_INITIALIZED = False
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 5  # seconds

# Emergency stop file path
STOP_FILE = "stop_optimization.txt"

# Global configurations
SYMBOL = "XAUUSD"
OPTIMIZATION_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]

# Global configurations
SYMBOL = "XAUUSD"
OPTIMIZATION_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]

# Add this configuration:
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

# Risk management configurations
RISK_CONFIG = {
    'max_position_size': 0.02,
    'max_daily_risk': 0.05,
    'max_correlation_risk': 0.7,
    'position_scaling': {
        'excellent_condition': 1.0,
        'good_condition': 0.75,
        'moderate_condition': 0.5,
        'poor_condition': 0.25
    },
    'market_conditions': {
        'volatility_threshold': 1.5,
        'trend_strength_threshold': 0.5,
        'volume_threshold': 1.2,
        'correlation_threshold': 0.7
    }
}

# Added function to manage MT5 connection
def ensure_mt5_connection() -> bool:
    """Ensure MT5 connection is active and reconnect if necessary"""
    global MT5_INITIALIZED

    try:
        if not MT5_INITIALIZED:
            mt5.shutdown()
            time.sleep(1)
            if mt5.initialize():
                MT5_INITIALIZED = True
                return True

            for attempt in range(MAX_RECONNECT_ATTEMPTS):
                logging.warning(f"MT5 connection attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS}")
                mt5.shutdown()
                time.sleep(RECONNECT_DELAY)

                if mt5.initialize():
                    MT5_INITIALIZED = True
                    return True

            return False

        # Check if connection is active
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            MT5_INITIALIZED = False
            return ensure_mt5_connection()

        return True

    except Exception as e:
        logging.error(f"Error in MT5 connection: {str(e)}")
        MT5_INITIALIZED = False
        return False

def test_mt5_connection():
    """Test MT5 connection and basic functionality with improved error handling"""
    print("\nTesting MT5 Connection...")
    mt5.shutdown()

    if not ensure_mt5_connection():
        print("Failed to initialize MT5")
        return False

    print("MT5 Package Version:", mt5.__version__)
    print("Terminal Info:", mt5.terminal_info())

    # Test symbol info with retry
    for attempt in range(MAX_RECONNECT_ATTEMPTS):
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is not None:
            print(f"Symbol: {SYMBOL}")
            print(f"Points: {symbol_info.point}")
            print(f"Digits: {symbol_info.digits}")
            print(f"Trade Mode: {symbol_info.trade_mode}")
            break
        else:
            if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                print(f"Retrying symbol info retrieval... ({attempt + 1}/{MAX_RECONNECT_ATTEMPTS})")
                time.sleep(RECONNECT_DELAY)
                if not ensure_mt5_connection():
                    continue
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

def test_mt5_connection():
    """Test MT5 connection and basic functionality with improved error handling"""
    print("\nTesting MT5 Connection...")
    mt5.shutdown()

    if not ensure_mt5_connection():
        print("Failed to initialize MT5")
        return False

    print("MT5 Package Version:", mt5.__version__)
    print("Terminal Info:", mt5.terminal_info())

    # Test symbol info with retry
    for attempt in range(MAX_RECONNECT_ATTEMPTS):
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is not None:
            print(f"Symbol: {SYMBOL}")
            print(f"Points: {symbol_info.point}")
            print(f"Digits: {symbol_info.digits}")
            print(f"Trade Mode: {symbol_info.trade_mode}")
            break
        else:
            if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                print(f"Retrying symbol info retrieval... ({attempt + 1}/{MAX_RECONNECT_ATTEMPTS})")
                time.sleep(RECONNECT_DELAY)
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

# Added data validation function
def validate_market_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """Validate market data quality"""
    if data is None or data.empty:
        return False, "Empty dataset"

    # Check for minimum required columns
    required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
    if not all(col in data.columns for col in required_cols):
        return False, "Missing required columns"

    # Check for initial NaN values and remove them
    data = data.dropna()
    if len(data) < 100:  # Minimum required bars
        return False, "Insufficient valid data points"

    # Basic price validation
    if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
        return False, "Invalid price values detected"

    return True, "Data validation successful"

class OptimizationLogger:
    """Enhanced logging system for optimization process"""
    def __init__(self, base_path: Path, symbol: str, timeframe: str):
        self.base_path = base_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.log_path = base_path / f"{symbol}_{timeframe}"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.setup_loggers()
        self._logged_messages = set()

    def setup_loggers(self):
        # Main optimization logger
        self.main_logger = self._setup_logger('main', 'optimization.log')
        self.progress_logger = self._setup_logger('progress', 'progress.log')
        self.results_logger = self._setup_logger('results', 'results.log')
        self.debug_logger = self._setup_logger('debug', 'debug.log')
        self.trades_logger = self._setup_logger('trades', 'trades.log')
        self.error_logger = self._setup_logger('error', 'error.log')

    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(f"{self.symbol}_{self.timeframe}_{name}")
        logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(self.log_path / filename)
        fh.setLevel(logging.DEBUG)

        # Console handler with reduced output
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.handlers = []
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def error(self, message: str, exc_info: bool = False):
        """Log error messages"""
        self.main_logger.error(message, exc_info=exc_info)
        self.error_logger.error(message, exc_info=exc_info)

    def warning(self, message: str):
        """Log warning messages"""
        self.main_logger.warning(message)
        self.debug_logger.warning(message)

    def info(self, message: str):
        """Log info messages"""
        self.main_logger.info(message)
        self.debug_logger.info(message)

    def debug(self, message: str):
        """Log debug messages"""
        self.debug_logger.debug(message)

    def log_progress(self, current: int, total: int, elapsed_time: float, remaining_time: float,
                    best_result: Optional[Dict] = None):
        msg = f"Progress: {current}/{total} ({(current/total)*100:.2f}%) - Elapsed: {elapsed_time:.2f}s - Remaining: {remaining_time:.2f}s"
        if msg not in self._logged_messages:
            if best_result:
                msg += (f"\nBest so far - "
                       f"Trades: {best_result.get('total_trades', 0)}, "
                       f"Win Rate: {best_result.get('win_rate', 0)*100:.2f}%, "
                       f"Profit: {best_result.get('avg_profit', 0)*100:.2f}%")
            self.progress_logger.info(msg)
            self._logged_messages.add(msg)

    def log_results(self, results: Dict):
        """Log optimization results"""
        self.results_logger.info(f"Optimization Results:\n{results}")

    def log_trade(self, trade_info: Dict):
        """Log trade information"""
        self.trades_logger.info(
            f"Trade: {trade_info.get('position', 'unknown')} - "
            f"Entry: {trade_info.get('entry_price', 0):.2f} - "
            f"Exit: {trade_info.get('exit_price', 0):.2f} - "
            f"Profit: {trade_info.get('profit', 0)*100:.2f}% - "
            f"Type: {trade_info.get('exit_reason', 'unknown')}"
        )

class RiskManager:
    """Advanced risk management system with enhanced validation"""
    def __init__(self, risk_config: dict):
        self.risk_config = risk_config
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0,
            'largest_loss': 0.0,
            'peak_balance': 0.0,
            'current_drawdown': 0.0
        }
        self.position_sizer = None
        self.market_conditions = {}
        self.trade_history = []
        self.risk_alerts = []
        self.stop_trading = False
        self.min_trade_interval = 300  # 5 minutes minimum between trades
        self.max_trades_per_hour = 5
        self.last_trade_time = None
        # Added for position sizing
        self.account_info = None
        self.daily_risk_used = 0
        self.open_positions = []

    def initialize_position_sizer(self, account_info: dict):
        """Initialize position size calculation parameters"""
        self.account_info = account_info
        self.daily_risk_used = 0
        self.open_positions = []
        self.peak_balance = account_info['balance']
        # Initialize risk metrics
        self.current_exposure = 0.0
        self.max_position_value = account_info['balance'] * self.risk_config['max_position_size']
        self.daily_risk_limit = account_info['balance'] * self.risk_config['max_daily_risk']

    def validate_trade_timing(self, current_time: datetime) -> bool:
        """Validate trade timing to prevent overtrading"""
        if self.last_trade_time is not None:
            time_since_last = (current_time - self.last_trade_time).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False

            # Check trades per hour
            recent_trades = sum(1 for t in self.trade_history[-self.max_trades_per_hour:]
                              if (current_time - t['time']).total_seconds() <= 3600)
            if recent_trades >= self.max_trades_per_hour:
                return False

        return True

    def validate_trade(self,
                      signal_type: str,
                      entry_price: float,
                      stop_loss: float,
                      market_metrics: dict) -> Tuple[bool, float]:
        """Validate trade with enhanced checks"""
        try:
            if not ensure_mt5_connection():
                return False, 0.0

            # Basic validation
            if self.stop_trading or not entry_price or not stop_loss:
                return False, 0.0

            # Timing validation
            current_time = datetime.now()
            if not self.validate_trade_timing(current_time):
                return False, 0.0

            # Enhanced market condition check
            condition = self._determine_market_condition(
                market_metrics['rel_volatility'],
                market_metrics['trend_strength'],
                market_metrics['volume_ratio'],
                market_metrics['momentum']
            )

            # Risk checks
            position_size = self._calculate_safe_position_size(
                entry_price, stop_loss, condition, market_metrics
            )

            if position_size <= 0:
                return False, 0.0

            # Update last trade time if validation passes
            self.last_trade_time = current_time
            return True, position_size

        except Exception as e:
            logging.error(f"Error validating trade: {str(e)}")
            return False, 0.0

    def _calculate_safe_position_size(self,
                                    entry_price: float,
                                    stop_loss: float,
                                    condition: str,
                                    market_metrics: dict) -> float:
        """Calculate position size with enhanced safety checks"""
        try:
            risk_amount = self.risk_config['max_position_size'] * self.position_sizer.account_info['balance']
            risk_per_pip = abs(entry_price - stop_loss)

            if risk_per_pip <= 0:
                return 0.0

            base_size = risk_amount / risk_per_pip

            # Apply condition-based scaling
            condition_multiplier = self.risk_config['position_scaling'].get(condition, 0.25)

            # Apply volatility adjustment
            vol_multiplier = min(1.0, self.risk_config['market_conditions']['volatility_threshold'] /
                               market_metrics.get('volatility', float('inf')))

            final_size = base_size * condition_multiplier * vol_multiplier

            # Additional safety caps
            max_size = self.position_sizer.account_info['balance'] * 0.05  # 5% max position
            return min(final_size, max_size)

        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0.0

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

        # Initialize components
        self.logger = OptimizationLogger(self.base_path, symbol, timeframe)
        self.market_analyzer = MarketAnalyzer()

        # Enhanced symbol info retrieval with retries
        for _ in range(MAX_RECONNECT_ATTEMPTS):
            if not ensure_mt5_connection():
                continue

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                self.point = symbol_info.point
                self.digits = symbol_info.digits
                self.trade_mode = symbol_info.trade_mode
                break
        else:
            raise ValueError(f"Could not get symbol info for {symbol}")

        # Account info management
        if not hasattr(WavyTunnelExitOptimizer, '_account_info'):
            if ensure_mt5_connection():
                account_info = mt5.account_info()
                if account_info is not None:
                    WavyTunnelExitOptimizer._account_info = {
                        'balance': account_info.balance,
                        'equity': account_info.equity,
                        'margin_free': account_info.margin_free
                    }
                else:
                    WavyTunnelExitOptimizer._account_info = {
                        'balance': 10000.0,
                        'equity': 10000.0,
                        'margin_free': 10000.0
                    }
                    self.logger.warning("Using default account values")

        self.account_info = WavyTunnelExitOptimizer._account_info
        self.risk_manager = RiskManager(RISK_CONFIG)
        self.risk_manager.initialize_position_sizer(self.account_info)

        self.param_ranges = self._setup_param_ranges()
        self.min_signals_required = 50
        self.max_signal_ratio = 3.0
        self.min_secondary_winrate = 0.5
        self.regime_period = 50

        self.performance_metrics = {
            'signal_balance': 0.0,
            'avg_profit_primary': 0.0,
            'avg_profit_secondary': 0.0,
            'risk_reward_ratio': 0.0
        }

        self.optimization_results = []
        self.best_params = None
        self.best_performance = None

    def _setup_param_ranges(self) -> Dict:
        """Define parameter ranges for optimization"""
        return {
            # Take profit distribution parameters
            'tp1_lot_percent': [40, 50, 60],
            'tp2_lot_percent': [20, 25, 30],
            'tp3_lot_percent': [10, 15],
            'tp4_lot_percent': [10, 15],

            # Take profit distances calibrated for XAUUSD
            'tp1_weight': np.array([0.001, 0.002, 0.003]),
            'tp2_weight': np.array([0.004, 0.005, 0.006]),
            'tp3_weight': np.array([0.007, 0.008, 0.009]),
            'tp4_weight': np.array([0.010, 0.011, 0.012]),

            # Stop loss parameters
            'wave_cross_buffer': np.array([0.001, 0.002]),
            'tunnel_touch_buffer': np.array([0.001, 0.002])
        }


    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Enhanced market data retrieval with validation"""
        try:
            if not ensure_mt5_connection():
                return None

            timeframe = getattr(mt5, f"TIMEFRAME_{self.timeframe}")
            rates = mt5.copy_rates_range(self.symbol, timeframe,
                                       self.start_date, self.end_date)

            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data available for {self.symbol} {self.timeframe}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Validate data quality
            is_valid, message = validate_market_data(df)
            if not is_valid:
                self.logger.warning(f"Data validation failed: {message}")  # Updated
                return None

            return self._clean_data(df)

        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")  # Updated
            return None

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data"""
        try:
            # Remove rows with any NaN values
            data = data.dropna()

            # Remove zero or negative prices
            for col in ['open', 'high', 'low', 'close']:
                data = data[data[col] > 0]

            # Ensure price relationships
            data = data[
                (data['high'] >= data['low']) &
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])
            ]

            # Ensure minimum data points
            if len(data) < self.min_signals_required:
                self.logger.warning("Insufficient data points after cleaning")
                return pd.DataFrame()

            return data

        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            return pd.DataFrame()

    def optimize_parallel(self) -> Tuple[Dict, pd.DataFrame]:
        """Run parallel optimization with enhanced error handling"""
        try:
            data = self._get_market_data()
            if data is None or data.empty:
                return None, None

            param_combinations = self._generate_param_combinations()
            total_combinations = len(param_combinations)

            if total_combinations == 0:
                self.logger.warning("No valid parameter combinations generated")
                return None, None

            # Setup parallel processing
            num_cores = mp.cpu_count()
            chunk_size = max(1, min(1000, total_combinations // (num_cores * 4)))
            chunks = [param_combinations[i:i + chunk_size]
                     for i in range(0, len(param_combinations), chunk_size)]

            results = []
            completed = 0
            start_time = time.time()

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = {
                    executor.submit(self._process_chunk, chunk, data.copy()): i
                    for i, chunk in enumerate(chunks)
                }

                for future in concurrent.futures.as_completed(futures):
                    try:
                        if os.path.exists(STOP_FILE):
                            executor.shutdown(wait=False)
                            return None, None

                        chunk_results = future.result(timeout=300)
                        if chunk_results:
                            results.extend(chunk_results)
                            completed += len(chunk_results)

                            elapsed_time = time.time() - start_time
                            self._update_optimization_progress(
                                completed, total_combinations, elapsed_time, chunk_results
                            )

                    except Exception as e:
                        self.logger.error(f"Error processing chunk: {str(e)}")
                        continue

            return self._finalize_results(results)

        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            return None, None

    def _finalize_results(self, results: List[Dict]) -> Tuple[Dict, pd.DataFrame]:
        """Finalize and validate optimization results"""
        try:
            if not results:
                return None, None

            results_df = pd.DataFrame(results)
            if len(results_df) == 0:
                return None, None

            best_params = self._get_best_params(results_df)
            if best_params is None:
                return None, None

            self._save_optimization_results(results_df, best_params)
            return best_params, results_df

        except Exception as e:
            self.logger.error(f"Error finalizing results: {str(e)}")
            return None, None

class MarketAnalyzer:
    """Market analysis and condition detection"""
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.market_states = []
        self.current_regime = None

    def analyze_market_condition(self, data: pd.DataFrame) -> Dict:
        """Analyze current market conditions with enhanced validation"""
        try:
            if data is None or data.empty:
                return None

            # Calculate volatility metrics
            returns = data['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std()
            avg_volatility = volatility.rolling(50).mean()

            # Calculate trend metrics
            sma_short = data['close'].rolling(20).mean()
            sma_long = data['close'].rolling(50).mean()
            trend_strength = ((sma_short - sma_long) / sma_long * 100)

            # Calculate volume metrics
            volume_sma = data['tick_volume'].rolling(20).mean()
            relative_volume = data['tick_volume'] / volume_sma.replace(0, np.nan).fillna(volume_sma.mean())

            # Determine market regime
            regime = self._determine_regime(
                volatility.iloc[-1],
                avg_volatility.iloc[-1],
                trend_strength.iloc[-1],
                relative_volume.iloc[-1]
            )

            metrics = {
                'regime': regime,
                'volatility': volatility.iloc[-1],
                'avg_volatility': avg_volatility.iloc[-1],
                'trend_strength': trend_strength.iloc[-1],
                'relative_volume': relative_volume.iloc[-1],
                'market_condition': regime  # Added for report generation
            }

            return metrics

        except Exception as e:
            logging.error(f"Error in market analysis: {str(e)}")
            return None

def create_combined_summary_report(all_results: Dict, base_dir: Path, symbol: str):
    """Create comprehensive summary report with error handling"""
    try:
        if not all_results:
            logging.warning("No results to create summary report")
            return

        summary_path = base_dir / "combined_summary.txt"

        with open(summary_path, "w") as f:
            f.write(f"Combined Summary Report for {symbol}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for timeframe, results in all_results.items():
                if not results or 'entry_params' not in results:
                    continue

                f.write(f"\nTimeframe: {timeframe}\n")
                f.write("-" * 40 + "\n")

                try:
                    # Entry Parameters
                    f.write("Entry Parameters:\n")
                    for param, value in results['entry_params'].items():
                        f.write(f"  {param}: {value}\n")

                    # Exit Parameters
                    if 'exit_params' in results:
                        exit_params = results['exit_params']
                        f.write("\nOptimized Exit Parameters:\n")

                        if 'tp1_lot_percent' in exit_params:
                            f.write("Take Profit Levels:\n")
                            for i in range(1, 5):
                                f.write(f"  TP{i}: {exit_params[f'tp{i}_lot_percent']}% "
                                       f"at {exit_params[f'tp{i}_weight']*100:.2f}%\n")

                        if all(key in exit_params for key in ['wave_cross_buffer', 'tunnel_touch_buffer']):
                            f.write("\nStop Loss Parameters:\n")
                            f.write(f"  Wave Cross Buffer: {exit_params['wave_cross_buffer']*100:.3f}%\n")
                            f.write(f"  Tunnel Touch Buffer: {exit_params['tunnel_touch_buffer']*100:.3f}%\n")

                        # Performance Metrics
                        f.write("\nPerformance Metrics:\n")
                        metrics = ['total_trades', 'win_rate', 'avg_profit_per_trade_pct',
                                 'sharpe_ratio', 'profit_factor', 'max_drawdown']
                        for metric in metrics:
                            if metric in exit_params:
                                value = exit_params[metric]
                                if isinstance(value, float):
                                    value = f"{value:.2f}"
                                f.write(f"  {metric}: {value}\n")

                except Exception as e:
                    logging.error(f"Error writing results for {timeframe}: {str(e)}")
                    continue

                f.write("\n" + "=" * 80 + "\n")

    except Exception as e:
        logging.error(f"Error creating combined summary report: {str(e)}")

def main():
    """Main execution function with enhanced error handling and connection management"""
    try:
        # Configuration
        symbol = SYMBOL
        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now()

        print(f"\nRunning optimization for {symbol} from {start_date} to {end_date}")

        # Create results directory
        base_results_dir = Path(f"exit_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        base_results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(base_results_dir / 'optimization.log'),
                logging.StreamHandler()
            ]
        )

        # Summary of results
        all_results = {}

        # Initial MT5 connection test
        if not test_mt5_connection():
            raise ConnectionError("Failed to establish MT5 connection")

        for timeframe in OPTIMIZATION_TIMEFRAMES:
            try:
                logging.info(f"\nOptimizing exit strategy for {timeframe}")

                # Force MT5 reinitialization for each timeframe
                mt5.shutdown()
                time.sleep(1)
                if not ensure_mt5_connection():
                    logging.error(f"Failed to connect MT5 for {timeframe}")
                    continue

                # Verify symbol availability
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logging.error(f"Symbol {symbol} not available for {timeframe}")
                    continue

                # Get entry parameters
                entry_params = TIMEFRAME_ENTRY_PARAMS.get(timeframe)
                if not entry_params:
                    logging.error(f"No entry parameters found for {timeframe}")
                    continue

                # Create optimizer instance
                optimizer = WavyTunnelExitOptimizer(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    entry_params=entry_params,
                    base_path=str(base_results_dir / timeframe)
                )

                # Run optimization
                best_params, results_df = optimizer.optimize_parallel()

                if best_params is not None:
                    all_results[timeframe] = {
                        'entry_params': entry_params,
                        'exit_params': best_params
                    }

                    # Create detailed reports
                    optimizer.create_optimization_report(results_df)
                else:
                    logging.warning(f"No valid results found for {timeframe}")

            except Exception as e:
                logging.error(f"Error optimizing {timeframe}: {str(e)}")
                continue

            finally:
                # Ensure clean disconnect between timeframes
                mt5.shutdown()
                time.sleep(1)

        # Create combined analysis if we have results
        if all_results:
            try:
                create_combined_summary_report(all_results, base_results_dir, symbol)
                logging.info("Created combined analysis reports")
            except Exception as e:
                logging.error(f"Error creating combined reports: {str(e)}")
        else:
            logging.warning("No valid results found for any timeframe")

    except Exception as e:
        logging.error(f"Critical error during execution: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()
        logging.info("\nExit strategy optimization completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nOptimization interrupted by user")
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()