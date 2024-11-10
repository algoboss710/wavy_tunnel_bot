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
    def __init__(self, symbols: List[str], timeframes: List[str], wavy_params: Dict[str, Dict[str, dict]], base_path: str = "optimization_results"):
        self.symbols = symbols
        self.timeframes = {tf: getattr(mt5, f"TIMEFRAME_{tf}") for tf in timeframes if hasattr(mt5, f"TIMEFRAME_{tf}")}
        self.wavy_params = wavy_params
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

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with advanced cleaning and normalization"""
        try:
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')

            # Remove outliers
            for col in ['high', 'low', 'close', 'open']:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data[col] = data[col].mask(z_scores > 3, data[col].rolling(5, center=True).mean())

            # Normalize volume
            data['normalized_volume'] = (data['tick_volume'] - data['tick_volume'].rolling(20).min()) / \
                                      (data['tick_volume'].rolling(20).max() - data['tick_volume'].rolling(20).min())

            return data

        except Exception as e:
            self.logger.main_logger.error(f"Error in preprocessing: {str(e)}")
            return data
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

    def validate_peak(self, window_data: pd.DataFrame, threshold: float, smoothed_current_high: float) -> bool:
        """Validate peak with multiple confirmation factors using smoothed high data"""
        try:
            center_idx = len(window_data) // 2
            lookback_high = window_data['high'].iloc[:center_idx].max()
            forward_high = window_data['high'].iloc[center_idx+1:].max()

            price_threshold = smoothed_current_high * threshold

            if not (smoothed_current_high > lookback_high + price_threshold and
                    smoothed_current_high > forward_high + price_threshold):
                return False

            # Additional validations
            volume_confirmed = self.validate_volume(window_data)
            trend_aligned = self.validate_trend_alignment(window_data, 'peak')

            return volume_confirmed and trend_aligned

        except Exception as e:
            self.logger.main_logger.error(f"Error in peak validation: {str(e)}")
            return False

    def validate_dip(self, window_data: pd.DataFrame, threshold: float, smoothed_current_low: float) -> bool:
        """Validate dip with multiple confirmation factors using smoothed low data"""
        try:
            center_idx = len(window_data) // 2
            lookback_low = window_data['low'].iloc[:center_idx].min()
            forward_low = window_data['low'].iloc[center_idx+1:].min()

            price_threshold = smoothed_current_low * threshold

            if not (smoothed_current_low < lookback_low - price_threshold and
                    smoothed_current_low < forward_low - price_threshold):
                return False

            # Additional validations
            volume_confirmed = self.validate_volume(window_data)
            trend_aligned = self.validate_trend_alignment(window_data, 'dip')

            return volume_confirmed and trend_aligned

        except Exception as e:
            self.logger.main_logger.error(f"Error in dip validation: {str(e)}")
            return False

    def validate_volume(self, window_data: pd.DataFrame) -> bool:
        """Validate volume confirmation"""
        try:
            center_idx = len(window_data) // 2
            current_volume = window_data['tick_volume'].iloc[center_idx]
            avg_volume = window_data['tick_volume'].iloc[:center_idx].mean()
            return current_volume > avg_volume * 1.2
        except Exception as e:
            self.logger.main_logger.error(f"Error in volume validation: {str(e)}")
            return False

    def validate_trend_alignment(self, window_data: pd.DataFrame, signal_type: str) -> bool:
        """Validate trend alignment"""
        try:
            # Calculate short-term trend
            ma_short = window_data['close'].rolling(window=5).mean()
            ma_long = window_data['close'].rolling(window=20).mean()

            center_idx = len(window_data) // 2

            if signal_type == 'peak':
                return ma_short.iloc[center_idx] > ma_long.iloc[center_idx]
            else:
                return ma_short.iloc[center_idx] < ma_long.iloc[center_idx]
        except Exception as e:
            self.logger.main_logger.error(f"Error in trend alignment validation: {str(e)}")
            return False

    def log_signal_detection(self, signal_type: str, timestamp: datetime, price: float):
        """Log detected signals with details"""
        confidence = self.calculate_signal_confidence()
        self.logger.log_signal(signal_type, timestamp, price, confidence)

    def calculate_signal_confidence(self) -> float:
        """Calculate confidence score for signals"""
        return 0.85
    def calculate_signal_metrics(self, data: pd.DataFrame, peaks: pd.Series, dips: pd.Series) -> Dict:
        """Calculate comprehensive signal metrics"""
        metrics = {
            'total_peaks': len(peaks[peaks > 0]),
            'total_dips': len(dips[dips > 0]),
            'success_rate': 0.0,
            'average_return': 0.0,
            'max_drawdown': 0.0
        }

        # Calculate success rate and returns
        successful_signals = 0
        total_return = 0
        max_drawdown = 0

        for idx in peaks[peaks > 0].index:
            success, ret = self.analyze_peak_performance(data, idx)
            if success:
                successful_signals += 1
                total_return += ret
            max_drawdown = min(max_drawdown, ret)

        for idx in dips[dips > 0].index:
            success, ret = self.analyze_dip_performance(data, idx)
            if success:
                successful_signals += 1
                total_return += ret
            max_drawdown = min(max_drawdown, ret)

        total_signals = metrics['total_peaks'] + metrics['total_dips']
        if total_signals > 0:
            metrics['success_rate'] = successful_signals / total_signals
            metrics['average_return'] = total_return / total_signals
            metrics['max_drawdown'] = max_drawdown

        return metrics

    def analyze_peak_performance(self, data: pd.DataFrame, idx: int, forward_window: int = 20) -> Tuple[bool, float]:
        """Analyze performance of peak signals"""
        try:
            if idx + forward_window >= len(data):
                return False, 0

            entry_price = data['high'].iloc[idx]
            forward_prices = data['low'].iloc[idx+1:idx+forward_window+1]
            lowest_price = forward_prices.min()

            return_pct = (lowest_price - entry_price) / entry_price
            success = return_pct < -0.001  # Consider successful if price drops at least 0.1%

            return success, return_pct
        except Exception as e:
            self.logger.main_logger.error(f"Error in peak performance analysis: {str(e)}")
            return False, 0

    def analyze_dip_performance(self, data: pd.DataFrame, idx: int, forward_window: int = 20) -> Tuple[bool, float]:
        """Analyze performance of dip signals"""
        try:
            if idx + forward_window >= len(data):
                return False, 0

            entry_price = data['low'].iloc[idx]
            forward_prices = data['high'].iloc[idx+1:idx+forward_window+1]
            highest_price = forward_prices.max()

            return_pct = (highest_price - entry_price) / entry_price
            success = return_pct > 0.001  # Consider successful if price rises at least 0.1%

            return success, return_pct
        except Exception as e:
            self.logger.main_logger.error(f"Error in dip performance analysis: {str(e)}")
            return False, 0

def main():
    symbols = {
        "EURUSD": {"M15": {"wavy": (30, 100, 200), "atr": (14, 2)}, "H1": {"wavy": (50, 120, 240), "atr": (14, 2)}},
        "XAUUSD": {"H1": {"wavy": (21, 89, 144), "atr": (10, 1.5)}, "D1": {"wavy": (34, 144, 233), "atr": (14, 2)}}
    }
    timeframes = ["M15", "H1", "H4", "D1"]
    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()

    optimizer = MultiSymbolOptimizer(list(symbols.keys()), timeframes)

    try:
        for symbol, tf_configs in symbols.items():
            for timeframe, config in tf_configs.items():
                optimizer.logger.main_logger.info(f"Starting optimization for {symbol} {timeframe}")
                data = optimizer.get_data(symbol, timeframe, start_date, end_date)

                if data is not None:
                    data = optimizer.preprocess_data(data)

                    # Configure Wavy Tunnel and ATR settings
                    data['wavy_h'] = optimizer.calculate_ema(data['high'], config['wavy'][0])
                    data['wavy_c'] = optimizer.calculate_ema(data['close'], config['wavy'][0])
                    data['wavy_l'] = optimizer.calculate_ema(data['low'], config['wavy'][0])
                    data['tunnel1'] = optimizer.calculate_ema(data['close'], config['wavy'][1])
                    data['tunnel2'] = optimizer.calculate_ema(data['close'], config['wavy'][2])
                    data['atr'] = optimizer.calculate_atr(data['high'], data['low'], data['close'], config['atr'][0])
                    data['threshold'] = data['atr'] * config['atr'][1]

                    # Integrate Peak and Dip Optimization
                    best_params, all_results = optimizer.optimize_parameters(data, symbol, timeframe)

                    if best_params:
                        print(f"\nBest parameters for {symbol} {timeframe}:")
                        print(f"Lookback: {best_params['lookback']}")
                        print(f"Threshold: {best_params['threshold']:.6f}")
                        print(f"Score: {best_params['score']:.4f}")
                        print("Metrics:")
                        for key, value in best_params['metrics'].items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"\nNo optimal parameters found for {symbol} {timeframe}")

    except Exception as e:
        print(f"Error during optimization: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
