import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from itertools import product

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

        # Store best parameters for final output
        self.best_configs = []

    def get_data(self, symbol, timeframe, start_date, end_date):
        """Retrieve historical data for a given symbol and timeframe."""
        rates = mt5.copy_rates_range(symbol, self.timeframes[timeframe], start_date, end_date)
        if rates is None or len(rates) == 0:
            self.logger.warning(f"No data available for {symbol} {timeframe} in range {start_date} to {end_date}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        self.logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        return df

    def identify_peaks_dips(self, prices, lookback, threshold):
        """Identify peaks and dips in the price data."""
        peaks = pd.Series(0, index=prices.index)
        dips = pd.Series(0, index=prices.index)
        for i in range(lookback, len(prices) - lookback):
            current_price = prices.iloc[i]
            prev_prices = prices.iloc[i-lookback:i]
            next_prices = prices.iloc[i+1:i+lookback+1]

            # Peak check
            if (current_price > prev_prices.max() * (1 + threshold) and
                current_price > next_prices.max() * (1 + threshold)):
                peaks.iloc[i] = 1

            # Dip check
            if (current_price < prev_prices.min() * (1 - threshold) and
                current_price < next_prices.min() * (1 - threshold)):
                dips.iloc[i] = 1

        self.logger.info(f"Identified {peaks.sum()} peaks and {dips.sum()} dips")
        return peaks, dips

    def optimize_parameters(self, data, symbol, timeframe):
        """Test different combinations of lookback and threshold to optimize peak and dip detection."""
        best_params = None
        best_score = -np.inf
        results = []

        lookback_range = [5, 10, 15, 20]
        threshold_range = [0.001, 0.002, 0.003, 0.004]

        for lookback, threshold in product(lookback_range, threshold_range):
            peaks, dips = self.identify_peaks_dips(data['close'], lookback, threshold)
            score = self.calculate_signal_quality(peaks, dips, data)
            results.append((lookback, threshold, score))

            if score > best_score:
                best_score = score
                best_params = (lookback, threshold)

            self.logger.info(f"Lookback: {lookback}, Threshold: {threshold}, Score: {score:.4f}")

        if best_params is not None:
            self.logger.info(f"Best parameters for {symbol} {timeframe}: Lookback={best_params[0]}, Threshold={best_params[1]}, Score={best_score:.4f}")
            # Add to the best configurations list
            self.best_configs.append((symbol, timeframe, best_params[0], best_params[1], best_score))
        else:
            self.logger.warning(f"No optimal parameters found for {symbol} {timeframe}")

        return best_params, results

    def calculate_signal_quality(self, peaks, dips, data):
        """A simple quality metric based on peak/dip density."""
        total_signals = peaks.sum() + dips.sum()
        avg_signal_distance = len(data) / total_signals if total_signals > 0 else np.inf
        return -avg_signal_distance  # For example, we want more frequent signals

    def close_connection(self):
        mt5.shutdown()

    def log_best_configs_table(self):
        """Log the best configurations in a table format."""
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

    # Output the final summary in table format
    optimizer.log_best_configs_table()
    optimizer.close_connection()

if __name__ == "__main__":
    main()
