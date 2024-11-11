import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from itertools import product
from scipy.ndimage import gaussian_filter1d
import psutil
import time
from typing import Dict, List, Tuple, Optional

class OptimizationLogger:
    """Logger class for optimization process"""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.log_path = base_path / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Set up different loggers
        self.main_logger = self._setup_logger('main', 'optimization.log')
        self.signal_logger = self._setup_logger('signal', 'signals.log')
        self.performance_logger = self._setup_logger('performance', 'performance.log')

    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path / filename)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def log_optimization_step(self, symbol: str, timeframe: str, params: dict, results: dict):
        self.main_logger.info(
            f"\nOptimization Step"
            f"\nSymbol: {symbol}"
            f"\nTimeframe: {timeframe}"
            f"\nParameters: {params}"
            f"\nResults: {results}"
            f"\n{'='*50}"
        )

    def log_signal(self, signal_type: str, time: datetime, price: float, confidence: float):
        self.signal_logger.info(
            f"\nSignal Detected"
            f"\nType: {signal_type}"
            f"\nTime: {time}"
            f"\nPrice: {price}"
            f"\nConfidence: {confidence:.2f}"
            f"\n{'-'*50}"
        )

    def log_performance(self, metrics: dict):
        self.performance_logger.info(
            f"\nPerformance Metrics"
            f"\nExecution Time: {metrics.get('execution_time', 0):.2f}s"
            f"\nMemory Usage: {metrics.get('memory_usage', 0):.2f}MB"
            f"\nSuccess Rate: {metrics.get('success_rate', 0):.2%}"
            f"\nTotal Signals: {metrics.get('total_signals', 0)}"
            f"\n{'-'*50}"
        )

class MultiSymbolOptimizer:
    def __init__(self, symbols: List[str], timeframes: List[str], wavy_params: Dict[str, Dict[str, Dict]], base_path: str = "optimization_results"):
        self.symbols = symbols
        self.timeframes = {tf: getattr(mt5, f"TIMEFRAME_{tf}") for tf in timeframes if hasattr(mt5, f"TIMEFRAME_{tf}")}
        self.base_path = Path(base_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = self.base_path / self.run_id
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = OptimizationLogger(self.results_path)

        # Initialize MT5 connection
        if not mt5.initialize():
            raise ConnectionError("Failed to initialize MetaTrader 5.")
        account_info = mt5.account_info()
        if account_info is None:
            raise ConnectionError("Failed to connect to account.")
        self.logger.main_logger.info(f"Connected to account {account_info.login}")

        # Wavy Tunnel parameters setup
        self.wavy_params = wavy_params

        # Initialize tracking variables
        self.best_configs = []
        self.total_trades_analyzed = 0
        self.total_trades_with_peaks_dips = 0

        # Set up optimization parameters
        self.setup_optimization_parameters()
    def setup_optimization_parameters(self):
        """Setup optimization parameters and ranges"""
        self.lookback_range = range(10, 101, 10)
        self.threshold_range = np.arange(0.001, 0.02, 0.001)
        self.smoothing_window_range = range(3, 16, 2)
        self.min_signal_distance = 10
        self.validation_window = 20

    def get_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from MT5 and perform initial processing"""
        try:
            rates = mt5.copy_rates_range(symbol, self.timeframes[timeframe], start_date, end_date)
            if rates is None or len(rates) == 0:
                self.logger.main_logger.warning(f"No data available for {symbol} {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Add technical indicators
            df['atr'] = self.calculate_atr(df)
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()

            self.logger.main_logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            self.logger.main_logger.error(f"Error fetching data: {str(e)}")
            return None

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(window=period).mean()

    def apply_wavy_tunnel(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Apply Wavy Tunnel EMA configurations per timeframe and symbol."""
        params = self.wavy_params.get(symbol, {}).get(timeframe, {})
        wavy_period = params.get('wavy_period', 21)
        high_ema_period = params.get('high_ema_period', 34)
        low_ema_period = params.get('low_ema_period', 34)
        close_ema_period = params.get('close_ema_period', 21)

        data['wavy_high'] = data['high'].ewm(span=high_ema_period, adjust=False).mean()
        data['wavy_low'] = data['low'].ewm(span=low_ema_period, adjust=False).mean()
        data['wavy_close'] = data['close'].ewm(span=close_ema_period, adjust=False).mean()

        self.logger.main_logger.info(f"Wavy Tunnel configured for {symbol} {timeframe} with periods {wavy_period}, {high_ema_period}, {low_ema_period}")

    def identify_peaks_dips(self, data: pd.DataFrame, lookback: int, threshold: float) -> Tuple[pd.Series, pd.Series]:
        """Enhanced peak and dip identification with multiple confirmation factors"""
        peaks = pd.Series(0, index=data.index)
        dips = pd.Series(0, index=data.index)

        # Convert data to numpy arrays for faster processing
        high_prices = data['high'].values
        low_prices = data['low'].values

        # Smooth data using numpy arrays
        smoothed_high = gaussian_filter1d(high_prices, sigma=3)
        smoothed_low = gaussian_filter1d(low_prices, sigma=3)

        for i in range(lookback, len(data) - lookback):
            try:
                # Create window slice with Timedelta adjustment
                start_idx = max(0, i - lookback)
                end_idx = min(len(data), i + lookback + 1)
                window_data = data.iloc[start_idx:end_idx].copy()

                # Peak detection with smoothed data
                if self.validate_peak(window_data, threshold, smoothed_high[i]):
                    peaks.iloc[i] = smoothed_high[i]
                    self.log_signal_detection("Peak", data.index[i], smoothed_high[i])

                # Dip detection with smoothed data
                if self.validate_dip(window_data, threshold, smoothed_low[i]):
                    dips.iloc[i] = smoothed_low[i]
                    self.log_signal_detection("Dip", data.index[i], smoothed_low[i])

            except Exception as e:
                self.logger.main_logger.error(f"Error in peak/dip detection at index {i}: {str(e)}")
                continue

        return peaks, dips

    def optimize_parameters(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[Dict, List[Dict]]:
        """Optimize parameters for peak and dip detection"""
        best_score = float('-inf')
        best_params = None
        all_results = []

        parameter_combinations = list(product(self.lookback_range, self.threshold_range))

        for lookback, threshold in parameter_combinations:
            peaks, dips = self.identify_peaks_dips(data, lookback, threshold)
            metrics = self.calculate_signal_metrics(data, peaks, dips)
            score = self.calculate_parameter_score(metrics)

            if score > best_score:
                best_score = score
                best_params = {'lookback': lookback, 'threshold': threshold}

            all_results.append({'lookback': lookback, 'threshold': threshold, 'score': score})

        return best_params, all_results

def main():
    symbols = {"EURUSD": {}, "XAUUSD": {}}
    timeframes = ["M15", "H1", "H4", "D1"]
    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()

    wavy_params = {
        "EURUSD": {
            "M15": {"wavy_period": 20, "atr_multiplier": 2.0, "high_ema_period": 34, "low_ema_period": 34, "close_ema_period": 21},
            "H1": {"wavy_period": 30, "atr_multiplier": 1.5, "high_ema_period": 50, "low_ema_period": 50, "close_ema_period": 30}
        },
        "XAUUSD": {
            "H4": {"wavy_period": 25, "atr_multiplier": 2.5, "high_ema_period": 55, "low_ema_period": 55, "close_ema_period": 25},
            "D1": {"wavy_period": 34, "atr_multiplier": 1.8, "high_ema_period": 100, "low_ema_period": 100, "close_ema_period": 34}
        }
    }

    optimizer = MultiSymbolOptimizer(symbols, timeframes, wavy_params)

    for symbol in symbols:
        for timeframe in timeframes:
            data = optimizer.get_data(symbol, timeframe, start_date, end_date)
            if data is not None:
                data = optimizer.apply_wavy_tunnel(data, symbol, timeframe)
                best_params, all_results = optimizer.optimize_parameters(data, symbol, timeframe)
                print(f"Best parameters for {symbol} {timeframe}: {best_params}")

    mt5.shutdown()
