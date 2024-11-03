import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from itertools import product
from scipy.ndimage import gaussian_filter1d

class MultiSymbolOptimizer:
    def __init__(self, symbols, timeframes, base_path="optimization_results"):
        self.symbols = symbols
        self.timeframes = {tf: getattr(mt5, f"TIMEFRAME_{tf}") for tf in timeframes if hasattr(mt5, f"TIMEFRAME_{tf}")}
        self.base_path = Path(base_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = self.base_path / self.run_id
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        handler = logging.FileHandler(self.results_path / "optimization.log")
        self.logger.addHandler(handler)

        # Initialize MT5 connection
        if not mt5.initialize():
            raise ConnectionError("Failed to initialize MetaTrader 5.")
        account_info = mt5.account_info()
        if account_info is None:
            raise ConnectionError("Failed to connect to account.")
        self.logger.info(f"Connected to account {account_info.login}")
        self.best_configs = []
        self.total_trades_analyzed = 0
        self.total_trades_with_peaks_dips = 0

    def get_data(self, symbol, timeframe, start_date, end_date):
        rates = mt5.copy_rates_range(symbol, self.timeframes[timeframe], start_date, end_date)
        if rates is None or len(rates) == 0:
            self.logger.warning(f"No data available for {symbol} {timeframe} in range {start_date} to {end_date}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        self.logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        return df

    def preprocess_data(self, prices, smoothing_window=5):
        """Smooth data to reduce noise before peak/dip detection."""
        return pd.Series(gaussian_filter1d(prices, sigma=smoothing_window), index=prices.index)

    def identify_peaks_dips(self, prices, lookback, threshold, symmetry=True, multi_level=True):
        peaks = pd.Series(0, index=prices.index)
        dips = pd.Series(0, index=prices.index)

        for i in range(lookback, len(prices) - lookback):
            current_price = prices.iloc[i]
            prev_prices = prices.iloc[i - lookback:i]
            next_prices = prices.iloc[i + 1:i + lookback + 1]

            # Symmetry Check
            is_symmetrical_peak = (current_price > prev_prices.mean()) and (current_price > next_prices.mean())
            is_symmetrical_dip = (current_price < prev_prices.mean()) and (current_price < next_prices.mean())

            # Relative and Multi-Level Peaks and Dips
            if (current_price > prev_prices.max() * (1 + threshold) and current_price > next_prices.max() * (1 + threshold) and is_symmetrical_peak):
                peaks.iloc[i] = current_price if multi_level else 1

            if (current_price < prev_prices.min() * (1 - threshold) and current_price < next_prices.min() * (1 - threshold) and is_symmetrical_dip):
                dips.iloc[i] = current_price if multi_level else 1

        return peaks, dips

    def calculate_wavy_tunnel(self, data):
        """Calculate Wavy Tunnel EMAs and identify long/short entry conditions."""
        wavy_h = data['high'].ewm(span=34, adjust=False).mean()
        wavy_c = data['close'].ewm(span=34, adjust=False).mean()
        wavy_l = data['low'].ewm(span=34, adjust=False).mean()
        tunnel1 = data['close'].ewm(span=144, adjust=False).mean()
        tunnel2 = data['close'].ewm(span=169, adjust=False).mean()

        # Entry conditions for Wavy Tunnel strategy
        data['long_condition'] = (data['open'] > wavy_h) & (wavy_l > tunnel2)
        data['short_condition'] = (data['open'] < wavy_l) & (wavy_h < tunnel1)

        return data, wavy_h, wavy_c, wavy_l, tunnel1, tunnel2

    def optimize_parameters(self, data, symbol, timeframe):
        best_params = None
        best_score = -np.inf
        results = []

        lookback_range = [5, 10, 15, 20, 25, 30]
        threshold_range = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]

        for lookback, threshold in product(lookback_range, threshold_range):
            smoothed_prices = self.preprocess_data(data['close'])
            peaks, dips = self.identify_peaks_dips(smoothed_prices, lookback, threshold)

            data, _, _, _, _, _ = self.calculate_wavy_tunnel(data)
            long_entries = data['long_condition'] & (peaks > 0)
            short_entries = data['short_condition'] & (dips > 0)

            # Track each analyzed trade with additional details
            for time, is_long, is_short in zip(data.index, long_entries, short_entries):
                self.total_trades_analyzed += 1
                if is_long or is_short:
                    self.total_trades_with_peaks_dips += 1
                    entry_type = "Long" if is_long else "Short"
                    self.logger.info(f"{symbol} {timeframe} | Trade analyzed at {time} | Entry Type: {entry_type}")

            score = self.calculate_signal_quality(long_entries, short_entries, data)
            results.append((lookback, threshold, score))

            if score > best_score:
                best_score = score
                best_params = (lookback, threshold)

            self.logger.info(f"Lookback: {lookback}, Threshold: {threshold}, Score: {score:.4f}")

        if best_params is not None:
            self.logger.info(f"Best parameters for {symbol} {timeframe}: Lookback={best_params[0]}, Threshold={best_params[1]}, Score={best_score:.4f}")
            self.best_configs.append((symbol, timeframe, best_params[0], best_params[1], best_score))
        else:
            self.logger.warning(f"No optimal parameters found for {symbol} {timeframe}")

        return best_params, results

    def calculate_signal_quality(self, long_entries, short_entries, data):
        total_signals = long_entries.sum() + short_entries.sum()
        if total_signals == 0:
            self.logger.info("No valid entries detected for this configuration.")
            return -np.inf
        avg_signal_distance = len(data) / total_signals if total_signals > 0 else np.inf
        profitability_score = total_signals * avg_signal_distance
        return profitability_score

    def close_connection(self):
        mt5.shutdown()

    def log_best_configs_table(self):
        if not self.best_configs:
            self.logger.info("No best configurations to display.")
            return

        header = f"{'Symbol':<10} {'Timeframe':<10} {'Lookback':<10} {'Threshold':<10} {'Score':<10}"
        self.logger.info("\n" + "=" * len(header))
        self.logger.info(header)
        self.logger.info("-" * len(header))
        for symbol, tf, lookback, threshold, score in self.best_configs:
            self.logger.info(f"{symbol:<10} {tf:<10} {lookback:<10} {threshold:<10.4f} {score:<10.4f}")
        self.logger.info("=" * len(header))
        self.logger.info(f"Total trades analyzed: {self.total_trades_analyzed}")
        self.logger.info(f"Total trades with peak/dip configuration applied: {self.total_trades_with_peaks_dips}")

def main():
    symbols = ["XAUUSD", "EURUSD"]
    timeframes = ["M15", "H1", "H4", "D1"]
    optimizer = MultiSymbolOptimizer(symbols, timeframes)

    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()

    for symbol in symbols:
        for tf in timeframes:
            data = optimizer.get_data(symbol, tf, start_date, end_date)
            if data is not None:
                best_params, all_results = optimizer.optimize_parameters(data, symbol, tf)
                if best_params:
                    print(f"Best parameters for {symbol} {tf}: Lookback={best_params[0]}, Threshold={best_params[1]}")
                else:
                    print(f"No optimal parameters found for {symbol} {tf}")

    optimizer.log_best_configs_table()
    optimizer.close_connection()

if __name__ == "__main__":
    main()
