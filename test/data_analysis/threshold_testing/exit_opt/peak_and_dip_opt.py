import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from itertools import product
from scipy.ndimage import gaussian_filter1d
import psutil  # For monitoring system resources
import plotly.graph_objects as go
import plotly.express as px

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
        """Set up individual logger with specific configuration"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Create file handler
        handler = logging.FileHandler(self.log_path / filename)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))

        # Add handler to logger
        logger.addHandler(handler)

        return logger

    def log_optimization_step(self, symbol: str, timeframe: str, params: dict, results: dict):
        """Log optimization step details"""
        self.main_logger.info(
            f"\nOptimization Step"
            f"\nSymbol: {symbol}"
            f"\nTimeframe: {timeframe}"
            f"\nParameters: {params}"
            f"\nResults: {results}"
            f"\n{'='*50}"
        )

    def log_signal(self, signal_type: str, time: datetime, price: float, confidence: float):
        """Log signal detection details"""
        self.signal_logger.info(
            f"\nSignal Detected"
            f"\nType: {signal_type}"
            f"\nTime: {time}"
            f"\nPrice: {price}"
            f"\nConfidence: {confidence:.2f}"
            f"\n{'-'*50}"
        )

    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        self.performance_logger.info(
            f"\nPerformance Metrics"
            f"\nExecution Time: {metrics.get('execution_time', 0):.2f}s"
            f"\nMemory Usage: {metrics.get('memory_usage', 0):.2f}MB"
            f"\nSuccess Rate: {metrics.get('success_rate', 0):.2%}"
            f"\nTotal Signals: {metrics.get('total_signals', 0)}"
            f"\n{'-'*50}"
        )

@dataclass
class SignalData:
    time: datetime
    type: str
    price: float
    confidence: float
    market_conditions: Dict
    confirmations: Dict

class MultiSymbolOptimizer:
    def __init__(self, symbols: List[str], timeframes: List[str], base_path: str = "optimization_results"):
        self.symbols = symbols
        self.timeframes = {tf: getattr(mt5, f"TIMEFRAME_{tf}") for tf in timeframes if hasattr(mt5, f"TIMEFRAME_{tf}")}
        self.base_path = Path(base_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = self.base_path / self.run_id
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = OptimizationLogger(self.results_path)

        # Performance tracking
        self.performance_metrics = {
            'execution_time': [],
            'memory_usage': [],
            'signals_per_timeframe': {},
            'optimization_iterations': 0
        }

        # Market condition tracking
        self.market_conditions = {
            'volatility_regimes': [],
            'trend_phases': [],
            'trading_sessions': []
        }

        # Initialize MT5 connection
        if not mt5.initialize():
            raise ConnectionError("Failed to initialize MetaTrader 5.")
        self.logger.main_logger.info("MetaTrader 5 initialized successfully")

        self.setup_optimization_parameters()

    def setup_optimization_parameters(self):
        """Setup optimization parameters and ranges"""
        self.lookback_range = range(5, 51, 5)
        self.threshold_range = np.arange(0.0005, 0.01, 0.0005)
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

            # Add market session information
            data['session'] = data.index.map(self.identify_trading_session)

            return data

        except Exception as e:
            self.logger.main_logger.error(f"Error in preprocessing: {str(e)}")
            return data

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

    def identify_trading_session(self, timestamp: datetime) -> str:
        """Identify trading session based on time"""
        hour = timestamp.hour
        if 8 <= hour < 16:
            return 'london'
        elif 13 <= hour < 21:
            return 'new_york'
        elif 0 <= hour < 8:
            return 'asia'
        return 'overlap'

    def calculate_trend_strength(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate trend strength using multiple indicators"""
        # ADX calculation
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff()
        tr = self.calculate_atr(data, 1)

        plus_di = 100 * (plus_dm.rolling(window).mean() / tr.rolling(window).mean())
        minus_di = 100 * (minus_dm.rolling(window).mean() / tr.rolling(window).mean())
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window).mean()

        return adx

    def identify_peaks_dips(self, data: pd.DataFrame, lookback: int, threshold: float) -> Tuple[pd.Series, pd.Series]:
        """Enhanced peak and dip identification with multiple confirmation factors"""
        peaks = pd.Series(0, index=data.index)
        dips = pd.Series(0, index=data.index)

        # Smooth data
        smoothed_high = gaussian_filter1d(data['high'].values, sigma=3)
        smoothed_low = gaussian_filter1d(data['low'].values, sigma=3)

        for i in range(lookback, len(data) - lookback):
            # Peak detection
            if self.validate_peak(data, i, lookback, threshold):
                peaks.iloc[i] = data['high'].iloc[i]
                self.log_signal_detection("Peak", data.index[i], data['high'].iloc[i])

            # Dip detection
            if self.validate_dip(data, i, lookback, threshold):
                dips.iloc[i] = data['low'].iloc[i]
                self.log_signal_detection("Dip", data.index[i], data['low'].iloc[i])

        return peaks, dips

    def log_signal_detection(self, signal_type: str, timestamp: datetime, price: float):
        """Log detected signals with details"""
        self.logger.signal_logger.info(
            f"Detected {signal_type} at {timestamp}"
            f"\nPrice: {price:.5f}"
            f"\nConfidence: {self.calculate_signal_confidence():.2f}"
        )

    def calculate_signal_confidence(self) -> float:
        """Calculate confidence score for signals"""
        # Implement confidence calculation logic
        return 0.85  # Placeholder

    def validate_peak(self, data: pd.DataFrame, idx: int, lookback: int, threshold: float) -> bool:
        """Validate peak with multiple confirmation factors"""
        if not self.check_basic_peak_conditions(data, idx, lookback, threshold):
            return False

        # Additional validations
        volume_confirmed = self.validate_volume(data, idx)
        pattern_confirmed = self.validate_pattern(data, idx)
        trend_aligned = self.validate_trend_alignment(data, idx, 'peak')

        return volume_confirmed and pattern_confirmed and trend_aligned

    def validate_dip(self, data: pd.DataFrame, idx: int, lookback: int, threshold: float) -> bool:
        """Validate dip with multiple confirmation factors"""
        if not self.check_basic_dip_conditions(data, idx, lookback, threshold):
            return False

        # Additional validations
        volume_confirmed = self.validate_volume(data, idx)
        pattern_confirmed = self.validate_pattern(data, idx)
        trend_aligned = self.validate_trend_alignment(data, idx, 'dip')

        return volume_confirmed and pattern_confirmed and trend_aligned

    def check_basic_peak_conditions(self, data: pd.DataFrame, idx: int, lookback: int, threshold: float) -> bool:
        """Check basic conditions for peak validation"""
        current_high = data['high'].iloc[idx]
        lookback_window = slice(idx - lookback, idx)
        lookforward_window = slice(idx + 1, idx + lookback + 1)

        price_threshold = current_high * (1 - threshold)

        higher_than_previous = current_high > data['high'].iloc[lookback_window].max()
        higher_than_next = current_high > data['high'].iloc[lookforward_window].max()

        return higher_than_previous and higher_than_next

def check_basic_dip_conditions(self, data: pd.DataFrame, idx: int, lookback: int, threshold: float) -> bool:
        """Check basic conditions for dip validation"""
        current_low = data['low'].iloc[idx]
        lookback_window = slice(idx - lookback, idx)
        lookforward_window = slice(idx + 1, idx + lookback + 1)

        price_threshold = current_low * (1 + threshold)

        lower_than_previous = current_low < data['low'].iloc[lookback_window].min()
        lower_than_next = current_low < data['low'].iloc[lookforward_window].min()

        return lower_than_previous and lower_than_next

def validate_volume(self, data: pd.DataFrame, idx: int) -> bool:
        """Validate volume confirmation"""
        current_volume = data['tick_volume'].iloc[idx]
        avg_volume = data['tick_volume'].iloc[idx-20:idx].mean()
        return current_volume > avg_volume * 1.2  # 20% above average

def validate_pattern(self, data: pd.DataFrame, idx: int) -> bool:
        """Validate price pattern confirmation"""
        # Implement pattern recognition logic
        return True  # Placeholder

def validate_trend_alignment(self, data: pd.DataFrame, idx: int, signal_type: str) -> bool:
        """Validate trend alignment"""
        trend = self.calculate_trend_strength(data.iloc[:idx])
        if signal_type == 'peak':
            return trend.iloc[-1] > 25  # Strong uptrend
        else:
            return trend.iloc[-1] < -25  # Strong downtrend

def calculate_signal_metrics(self, data: pd.DataFrame, peaks: pd.Series, dips: pd.Series) -> Dict:
        """Calculate comprehensive signal metrics"""
        metrics = {
            'peak_success_rate': 0.0,
            'dip_success_rate': 0.0,
            'avg_profit_potential': 0.0,
            'false_signals': 0,
            'signal_distribution': {},
            'market_condition_performance': {}
        }

        # Analyze peaks
        for idx in peaks[peaks > 0].index:
            success = self.analyze_signal_success(data, idx, 'peak')
            metrics['peak_success_rate'] += int(success)

        # Analyze dips
        for idx in dips[dips > 0].index:
            success = self.analyze_signal_success(data, idx, 'dip')
            metrics['dip_success_rate'] += int(success)

        # Calculate final metrics
        total_peaks = len(peaks[peaks > 0])
        total_dips = len(dips[dips > 0])

        if total_peaks > 0:
            metrics['peak_success_rate'] /= total_peaks
        if total_dips > 0:
            metrics['dip_success_rate'] /= total_dips

        return metrics

def analyze_signal_success(self, data: pd.DataFrame, idx: int, signal_type: str) -> bool:
        """Analyze if a signal was successful"""
        forward_window = 20  # Bars to look ahead
        if idx + forward_window >= len(data):
            return False

        if signal_type == 'peak':
            price_movement = (data['high'].iloc[idx] - data['low'].iloc[idx:idx+forward_window].min()) / data['high'].iloc[idx]
            return price_movement > 0.01  # 1% movement
        else:
            price_movement = (data['high'].iloc[idx:idx+forward_window].max() - data['low'].iloc[idx]) / data['low'].iloc[idx]
            return price_movement > 0.01

def run_optimization(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Dict:
        """Run full optimization process"""
        start_time = time.time()
        self.logger.main_logger.info(f"Starting optimization for {symbol} {timeframe}")

        try:
            # Get and preprocess data
            data = self.get_data(symbol, timeframe, start_date, end_date)
            if data is None:
                raise ValueError(f"No data available for {symbol} {timeframe}")

            data = self.preprocess_data(data)

            # Split data for validation
            split_idx = int(len(data) * 0.7)
            train_data = data[:split_idx]
            test_data = data[split_idx:]

            best_params = self.find_optimal_parameters(train_data)
            validation_results = self.validate_parameters(test_data, best_params)

            # Log performance metrics
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            self.log_optimization_results(symbol, timeframe, best_params, validation_results, execution_time, memory_usage)

            return {
                'best_parameters': best_params,
                'validation_results': validation_results,
                'execution_time': execution_time,
                'memory_usage': memory_usage
            }

        except Exception as e:
            self.logger.main_logger.error(f"Optimization failed: {str(e)}")
            return None

def find_optimal_parameters(self, data: pd.DataFrame) -> Dict:
        """Find optimal parameters using grid search"""
        best_score = float('-inf')
        best_params = None

        parameter_combinations = list(product(
            self.lookback_range,
            self.threshold_range,
            self.smoothing_window_range
        ))

        for lookback, threshold, smoothing in parameter_combinations:
            self.performance_metrics['optimization_iterations'] += 1

            # Apply parameters
            peaks, dips = self.identify_peaks_dips(data, lookback, threshold)
            metrics = self.calculate_signal_metrics(data, peaks, dips)

            # Calculate score
            score = self.calculate_parameter_score(metrics)

            if score > best_score:
                best_score = score
                best_params = {
                    'lookback': lookback,
                    'threshold': threshold,
                    'smoothing': smoothing
                }

            self.log_optimization_step(lookback, threshold, smoothing, score)

        return best_params

def validate_parameters(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Validate parameters on test data"""
        peaks, dips = self.identify_peaks_dips(
            data,
            params['lookback'],
            params['threshold']
        )

        return self.calculate_signal_metrics(data, peaks, dips)

def log_optimization_step(self, lookback: int, threshold: float, smoothing: int, score: float):
        """Log each optimization step"""
        self.logger.main_logger.info(
            f"Parameters: lookback={lookback}, threshold={threshold:.4f}, smoothing={smoothing}"
            f"\nScore: {score:.4f}"
        )

def log_optimization_results(self, symbol: str, timeframe: str, params: Dict, results: Dict,
                               execution_time: float, memory_usage: float):
        """Log optimization results"""
        self.logger.main_logger.info(
            f"\nOptimization Results for {symbol} {timeframe}"
            f"\nBest Parameters: {params}"
            f"\nValidation Results: {results}"
            f"\nExecution Time: {execution_time:.2f}s"
            f"\nMemory Usage: {memory_usage:.2f}MB"
        )

def calculate_parameter_score(self, metrics: Dict) -> float:
        """Calculate overall score for parameter combination"""
        weights = {
            'peak_success_rate': 0.3,
            'dip_success_rate': 0.3,
            'false_signals': -0.2,
            'avg_profit_potential': 0.2
        }

        score = (
            metrics['peak_success_rate'] * weights['peak_success_rate'] +
            metrics['dip_success_rate'] * weights['dip_success_rate'] +
            metrics['avg_profit_potential'] * weights['avg_profit_potential']
        )

        # Penalize for false signals
        false_signal_rate = metrics['false_signals'] / max(
            (len(metrics['signal_distribution'])), 1)
        score += false_signal_rate * weights['false_signals']

        return score

def generate_optimization_report(self, symbol: str, timeframe: str, results: Dict):
        """Generate detailed HTML report with interactive charts"""
        report = Report(self.results_path)
        report.add_overview(symbol, timeframe, results)
        report.add_performance_metrics(results)
        report.add_signal_analysis(results)
        report.save()

def cleanup(self):
        """Cleanup resources"""
        mt5.shutdown()
        self.logger.main_logger.info("Optimization completed. Resources cleaned up.")

class Report:
    """Class for generating HTML reports with interactive charts"""
    def __init__(self, path: Path):
        self.path = path
        self.content = []

    def add_overview(self, symbol: str, timeframe: str, results: Dict):
        """Add overview section to report"""
        pass  # Implement report generation logic

    def add_performance_metrics(self, results: Dict):
        """Add performance metrics section to report"""
        pass  # Implement performance metrics visualization

    def add_signal_analysis(self, results: Dict):
        """Add signal analysis section to report"""
        pass  # Implement signal analysis visualization

    def save(self):
        """Save report to file"""
        pass  # Implement save logic

if __name__ == "__main__":
    # Example usage
    symbols = ["EURUSD", "XAUUSD"]
    timeframes = ["M15", "H1", "H4", "D1"]
    optimizer = MultiSymbolOptimizer(symbols, timeframes)

    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()

    try:
        for symbol in symbols:
            for timeframe in timeframes:
                results = optimizer.run_optimization(symbol, timeframe, start_date, end_date)
                if results:
                    optimizer.generate_optimization_report(symbol, timeframe, results)
    finally:
        optimizer.cleanup()