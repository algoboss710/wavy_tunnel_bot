import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from itertools import product
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional, Union

class TradingLogger:
    """Simplified logger for trading operations"""
    def __init__(self, base_path: Path):
        self.logger = self._setup_logger(base_path / "trading.log")

    def _setup_logger(self, log_file: Path) -> logging.Logger:
        logger = logging.getLogger('trading')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def log(self, message: str, level: str = 'info'):
        getattr(self.logger, level)(message)

class MarketAnalyzer:
    """Handles data processing and signal detection"""
    def __init__(self, symbols: List[str], timeframes: List[str], base_path: str = "trading_results"):
        self.symbols = symbols
        self.timeframes = {tf: getattr(mt5, f"TIMEFRAME_{tf}")
                         for tf in timeframes if hasattr(mt5, f"TIMEFRAME_{tf}")}
        self.base_path = Path(base_path)
        self.logger = TradingLogger(self.base_path)

        # Initialize MT5
        if not self._initialize_mt5():
            raise ConnectionError("Failed to initialize MetaTrader 5")

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.log("Failed to initialize MT5", 'error')
            return False

        account_info = mt5.account_info()
        if account_info is None:
            self.logger.log("Failed to connect to account", 'error')
            return False

        self.logger.log(f"Connected to account {account_info.login}")
        return True

    def get_data(self, symbol: str, timeframe: str,
                 start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch and preprocess market data"""
        try:
            rates = mt5.copy_rates_range(symbol, self.timeframes[timeframe],
                                       start_date, end_date)
            if rates is None or len(rates) == 0:
                self.logger.log(f"No data available for {symbol} {timeframe}", 'warning')
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Add technical indicators and clean data in one pass
            df = self._process_data(df)

            self.logger.log(f"Processed {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            self.logger.log(f"Error fetching data: {str(e)}", 'error')
            return None

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data with technical indicators and cleaning in one pass"""
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Calculate technical indicators
        high, low, close = data['high'], data['low'], data['close']

        # ATR calculation
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        # Volume indicators
        data['volume_sma'] = data['tick_volume'].rolling(20).mean()
        data['normalized_volume'] = ((data['tick_volume'] - data['tick_volume'].rolling(20).min()) /
                                   (data['tick_volume'].rolling(20).max() -
                                    data['tick_volume'].rolling(20).min()))

        # Clean outliers using z-score
        for col in ['high', 'low', 'close', 'open']:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            data[col] = data[col].mask(z_scores > 3, data[col].rolling(5, center=True).mean())

        return data

    def detect_market_signals(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
        """Detect market signals using combined peak/dip detection"""
        peaks = pd.Series(0, index=data.index)
        dips = pd.Series(0, index=data.index)

        # Smooth price data
        smoothed_high = gaussian_filter1d(data['high'].values, sigma=3)
        smoothed_low = gaussian_filter1d(data['low'].values, sigma=3)

        lookback = params['lookback']
        threshold = params['threshold']

        for i in range(lookback, len(data) - lookback):
            window_data = data.iloc[i-lookback:i+lookback+1].copy()

            # Check peaks
            if self._validate_signal(window_data, smoothed_high[i], threshold, 'peak'):
                peaks.iloc[i] = smoothed_high[i]

            # Check dips
            if self._validate_signal(window_data, smoothed_low[i], threshold, 'dip'):
                dips.iloc[i] = smoothed_low[i]

        return peaks, dips

    def _validate_signal(self, window_data: pd.DataFrame, current_price: float,
                        threshold: float, signal_type: str) -> bool:
        """Unified signal validation for both peaks and dips"""
        try:
            center_idx = len(window_data) // 2

            # Price validation
            if signal_type == 'peak':
                lookback_price = window_data['high'].iloc[:center_idx].max()
                forward_price = window_data['high'].iloc[center_idx+1:].max()
                valid_price = (current_price > lookback_price + (current_price * threshold) and
                             current_price > forward_price + (current_price * threshold))
            else:  # dip
                lookback_price = window_data['low'].iloc[:center_idx].min()
                forward_price = window_data['low'].iloc[center_idx+1:].min()
                valid_price = (current_price < lookback_price - (current_price * threshold) and
                             current_price < forward_price - (current_price * threshold))

            if not valid_price:
                return False

            # Volume validation
            current_volume = window_data['tick_volume'].iloc[center_idx]
            avg_volume = window_data['tick_volume'].iloc[:center_idx].mean()
            if current_volume <= avg_volume * 1.2:
                return False

            # Trend validation
            ma_short = window_data['close'].rolling(window=5).mean()
            ma_long = window_data['close'].rolling(window=20).mean()
            trend_aligned = ((ma_short.iloc[center_idx] > ma_long.iloc[center_idx])
                           if signal_type == 'peak'
                           else (ma_short.iloc[center_idx] < ma_long.iloc[center_idx]))

            return trend_aligned

        except Exception as e:
            self.logger.log(f"Error in signal validation: {str(e)}", 'error')
            return False
class StrategyOptimizer:
    """Handles strategy optimization"""
    def __init__(self, analyzer: MarketAnalyzer):
        self.analyzer = analyzer
        self.lookback_range = range(10, 101, 10)
        self.threshold_range = np.arange(0.001, 0.02, 0.001)

    def optimize_parameters(self, data: pd.DataFrame, symbol: str,
                          timeframe: str) -> Tuple[Dict, List[Dict]]:
        """Optimize strategy parameters"""
        best_score = float('-inf')
        best_params = None
        all_results = []

        parameter_combinations = list(product(self.lookback_range, self.threshold_range))
        total_combinations = len(parameter_combinations)

        self.analyzer.logger.log(
            f"Starting optimization for {symbol} {timeframe} "
            f"with {total_combinations} combinations"
        )

        for i, (lookback, threshold) in enumerate(parameter_combinations, 1):
            try:
                params = {'lookback': lookback, 'threshold': threshold}
                peaks, dips = self.analyzer.detect_market_signals(data, params)
                metrics = self._calculate_metrics(data, peaks, dips)
                score = self._calculate_score(metrics)

                result = {
                    'lookback': lookback,
                    'threshold': threshold,
                    'metrics': metrics,
                    'score': score
                }
                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = result

                if i % 10 == 0:
                    progress = (i / total_combinations) * 100
                    self.analyzer.logger.log(
                        f"Progress: {progress:.1f}% for {symbol} {timeframe}"
                    )

            except Exception as e:
                self.analyzer.logger.log(
                    f"Error in optimization iteration: {str(e)}", 'error'
                )

        return best_params, all_results

    def _calculate_metrics(self, data: pd.DataFrame, peaks: pd.Series,
                         dips: pd.Series) -> Dict:
        """Calculate performance metrics"""
        metrics = {
            'total_signals': len(peaks[peaks > 0]) + len(dips[dips > 0]),
            'success_rate': 0.0,
            'average_return': 0.0,
            'max_drawdown': 0.0
        }

        if metrics['total_signals'] == 0:
            return metrics

        returns = []
        for idx in peaks[peaks > 0].index:
            ret = self._analyze_signal_performance(data, idx, 'peak')
            if ret is not None:
                returns.append(ret)

        for idx in dips[dips > 0].index:
            ret = self._analyze_signal_performance(data, idx, 'dip')
            if ret is not None:
                returns.append(ret)

        if returns:
            metrics['success_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            metrics['average_return'] = np.mean(returns)
            metrics['max_drawdown'] = min(returns)

        return metrics

    def _analyze_signal_performance(self, data: pd.DataFrame, idx: int,
                                  signal_type: str, forward_window: int = 20) -> Optional[float]:
        """Analyze performance of a signal"""
        try:
            if idx + forward_window >= len(data):
                return None

            if signal_type == 'peak':
                entry_price = data['high'].iloc[idx]
                exit_price = data['low'].iloc[idx+1:idx+forward_window+1].min()
            else:  # dip
                entry_price = data['low'].iloc[idx]
                exit_price = data['high'].iloc[idx+1:idx+forward_window+1].max()

            return (exit_price - entry_price) / entry_price

        except Exception as e:
            self.analyzer.logger.log(
                f"Error in signal performance analysis: {str(e)}", 'error'
            )
            return None

    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate overall strategy score"""
        try:
            weights = {
                'success_rate': 0.4,
                'average_return': 0.3,
                'max_drawdown': 0.3
            }

            return (metrics['success_rate'] * weights['success_rate'] +
                   metrics['average_return'] * weights['average_return'] +
                   (1 + metrics['max_drawdown']) * weights['max_drawdown'])

        except Exception as e:
            self.analyzer.logger.log(f"Error calculating score: {str(e)}", 'error')
            return float('-inf')

def main():
    # Configuration
    symbols = ["EURUSD", "XAUUSD"]
    timeframes = ["M15", "H1", "H4", "D1"]
    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()

    try:
        # Initialize analyzer and optimizer
        analyzer = MarketAnalyzer(symbols, timeframes)
        optimizer = StrategyOptimizer(analyzer)

        # Run optimization for each symbol and timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                analyzer.logger.log(f"Processing {symbol} {timeframe}")

                # Get and process data
                data = analyzer.get_data(symbol, timeframe, start_date, end_date)
                if data is None:
                    continue

                # Optimize parameters
                best_params, _ = optimizer.optimize_parameters(data, symbol, timeframe)

                if best_params:
                    print(f"\nBest parameters for {symbol} {timeframe}:")
                    print(f"Lookback: {best_params['lookback']}")
                    print(f"Threshold: {best_params['threshold']:.6f}")
                    print(f"Score: {best_params['score']:.4f}")
                    print("Metrics:")
                    for key, value in best_params['metrics'].items():
                        print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error during optimization: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()