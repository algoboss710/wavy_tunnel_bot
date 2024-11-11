import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import time
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class OptimizationLogger:
    """Handles logging for the optimization process"""
    def __init__(self, base_path: Path, symbol: str, timeframe: str):
        self.base_path = base_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.log_path = base_path / f"{symbol}_{timeframe}"
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Setup different loggers
        self.setup_loggers()

    def setup_loggers(self):
        # Main optimization logger
        self.main_logger = self._setup_logger('main', 'optimization.log')
        # Progress logger
        self.progress_logger = self._setup_logger('progress', 'progress.log')
        # Results logger
        self.results_logger = self._setup_logger('results', 'results.log')

    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(f"{self.symbol}_{self.timeframe}_{name}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path / filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log_params(self, params: Dict):
        self.main_logger.info(f"Starting optimization with parameters: {params}")

    def log_progress(self, current: int, total: int, elapsed_time: float):
        progress = (current / total) * 100
        self.progress_logger.info(
            f"Progress: {current}/{total} ({progress:.2f}%) - "
            f"Elapsed Time: {elapsed_time:.2f}s"
        )

    def log_results(self, results: Dict):
        self.results_logger.info(f"Optimization Results:\n{results}")

class WavyTunnelOptimizer:
    """Main optimizer class for Wavy Tunnel strategy"""
    def __init__(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime,
                 base_path: str = "optimization_results"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.base_path = Path(base_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = OptimizationLogger(self.base_path, symbol, timeframe)

        # Parameter ranges for optimization
        self.param_ranges = self._setup_param_ranges()

    def _setup_param_ranges(self) -> Dict:
        """Define parameter ranges for optimization"""
        return {
            'wavy_period': range(20, 51, 5),
            'tunnel_period1': range(100, 301, 20),
            'tunnel_period2': range(120, 321, 20),
            'min_gap_second': range(5, 31, 5),
            'max_zone_percentage': np.arange(0.1, 0.5, 0.05),
        }

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.main_logger.error("Failed to initialize MT5")
            return False
        return True

    def get_data(self) -> Optional[pd.DataFrame]:
        """Fetch and preprocess market data"""
        try:
            if not self._initialize_mt5():
                return None

            timeframe = getattr(mt5, f"TIMEFRAME_{self.timeframe}")
            rates = mt5.copy_rates_range(self.symbol, timeframe,
                                       self.start_date, self.end_date)

            if rates is None or len(rates) == 0:
                self.logger.main_logger.error(f"No data available for {self.symbol} {self.timeframe}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            return self._preprocess_data(df)

        except Exception as e:
            self.logger.main_logger.error(f"Error fetching data: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the market data"""
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Remove outliers
        for col in ['high', 'low', 'close', 'open']:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            data[col] = data[col].mask(z_scores > 3, data[col].rolling(5, center=True).mean())

        # Add basic volume metrics
        data['volume_sma'] = data['tick_volume'].rolling(20).mean()

        return data
    def generate_signals(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.Series]:
        """Generate entry signals for both primary and secondary strategies"""
        # Calculate EMAs
        wavy_h = data['high'].ewm(span=params['wavy_period'], adjust=False).mean()
        wavy_c = data['close'].ewm(span=params['wavy_period'], adjust=False).mean()
        wavy_l = data['low'].ewm(span=params['wavy_period'], adjust=False).mean()
        tunnel1 = data['close'].ewm(span=params['tunnel_period1'], adjust=False).mean()
        tunnel2 = data['close'].ewm(span=params['tunnel_period2'], adjust=False).mean()

        # Calculate max/min values for waves and tunnels
        wavy_max = pd.concat([wavy_h, wavy_c, wavy_l], axis=1).max(axis=1)
        wavy_min = pd.concat([wavy_h, wavy_c, wavy_l], axis=1).min(axis=1)
        tunnel_max = pd.concat([tunnel1, tunnel2], axis=1).max(axis=1)
        tunnel_min = pd.concat([tunnel1, tunnel2], axis=1).min(axis=1)

        # Primary Strategy Signals
        primary_longs = (data['open'] > wavy_max) & (wavy_min > tunnel_max)
        primary_shorts = (data['open'] < wavy_min) & (wavy_max < tunnel_min)

        # Secondary Strategy Signals
        # Detect crossovers
        crossover_up = (data['close'].shift(1) <= wavy_max.shift(1)) & (data['close'] > wavy_max)
        crossover_down = (data['close'].shift(1) >= wavy_min.shift(1)) & (data['close'] < wavy_min)

        # Calculate distances for secondary strategy
        long_distance = tunnel_min - data['close']
        short_distance = data['close'] - tunnel_max

        # Calculate zone percentages
        zone_percentage_long = (data['close'] - wavy_max) / (tunnel_min - wavy_max)
        zone_percentage_short = (wavy_min - data['close']) / (wavy_min - tunnel_max)

        # Secondary strategy conditions
        secondary_longs = (crossover_up &
                         (data['close'] < tunnel_min) &
                         (long_distance > params['min_gap_second'] * data['tick_volume'].mean()) &
                         (zone_percentage_long <= params['max_zone_percentage']))

        secondary_shorts = (crossover_down &
                          (data['close'] > tunnel_max) &
                          (short_distance > params['min_gap_second'] * data['tick_volume'].mean()) &
                          (zone_percentage_short <= params['max_zone_percentage']))

        # Combine signals
        long_signals = primary_longs | secondary_longs
        short_signals = primary_shorts | secondary_shorts

        return long_signals, short_signals

    def evaluate_signals(self, data: pd.DataFrame, long_signals: pd.Series,
                        short_signals: pd.Series, forward_bars: int = 5) -> Dict:
        """Evaluate the quality of entry signals"""
        results = {
            'total_signals': 0,
            'winning_signals': 0,
            'avg_profit_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'primary_signals': 0,
            'secondary_signals': 0
        }

        # Calculate forward returns
        forward_returns = pd.Series(index=data.index, dtype=float)

        # Calculate returns for long signals
        for idx in data.index[long_signals]:
            if idx + forward_bars <= data.index[-1]:
                forward_return = (data['close'].loc[idx:idx + forward_bars].max() -
                                data['open'].loc[idx]) / data['open'].loc[idx]
                forward_returns[idx] = forward_return

        # Calculate returns for short signals
        for idx in data.index[short_signals]:
            if idx + forward_bars <= data.index[-1]:
                forward_return = (data['open'].loc[idx] -
                                data['close'].loc[idx:idx + forward_bars].min()) / data['open'].loc[idx]
                forward_returns[idx] = forward_return

        # Calculate metrics
        valid_returns = forward_returns.dropna()
        if len(valid_returns) > 0:
            results['total_signals'] = len(valid_returns)
            results['winning_signals'] = (valid_returns > 0).sum()
            results['avg_profit_loss'] = valid_returns.mean()
            results['max_drawdown'] = valid_returns.min()
            results['sharpe_ratio'] = (valid_returns.mean() / valid_returns.std()
                                     if valid_returns.std() != 0 else 0)
            results['win_rate'] = results['winning_signals'] / results['total_signals']

        return results
    def optimize_parallel(self) -> Tuple[Dict, pd.DataFrame]:
        """Run parallel optimization process"""
        data = self.get_data()
        if data is None:
            return None, None

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)

        self.logger.log_params({'total_combinations': total_combinations})

        # Split combinations for parallel processing
        num_cores = mp.cpu_count()
        chunk_size = max(1, total_combinations // (num_cores * 4))
        chunks = [param_combinations[i:i + chunk_size]
                 for i in range(0, len(param_combinations), chunk_size)]

        start_time = time.time()
        results = []

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit chunks for parallel processing
            futures = [
                executor.submit(self._process_chunk, chunk, data.copy())
                for chunk in chunks
            ]

            # Process results as they complete
            for i, future in enumerate(futures):
                chunk_results = future.result()
                results.extend(chunk_results)

                elapsed_time = time.time() - start_time
                self.logger.log_progress(
                    (i + 1) * chunk_size,
                    total_combinations,
                    elapsed_time
                )

        # Convert results to DataFrame and find best parameters
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            best_params = results_df.nlargest(1, 'sharpe_ratio').iloc[0].to_dict()
            self.logger.log_results(best_params)
        else:
            best_params = None
            self.logger.main_logger.warning("No valid results found")

        return best_params, results_df

    def _process_chunk(self, chunk: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Process a chunk of parameter combinations"""
        chunk_results = []
        for params in chunk:
            try:
                # Generate signals
                long_signals, short_signals = self.generate_signals(data, params)

                # Evaluate signals
                evaluation = self.evaluate_signals(data, long_signals, short_signals)

                # Store results
                result = {**params, **evaluation}
                chunk_results.append(result)

            except Exception as e:
                self.logger.main_logger.error(f"Error processing parameters {params}: {str(e)}")

        return chunk_results

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations for testing"""
        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())

        combinations = []
        for values in product(*param_values):
            combinations.append(dict(zip(param_keys, values)))

        return combinations

    def create_optimization_report(self, results_df: pd.DataFrame):
        """Create and save optimization report with visualizations"""
        if results_df is None or len(results_df) == 0:
            self.logger.main_logger.warning("No results to create report")
            return

        # Create report directory
        report_path = self.base_path / f"{self.symbol}_{self.timeframe}_report"
        report_path.mkdir(exist_ok=True)

        # Create performance visualization
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Sharpe Ratio Distribution',
                                        'Win Rate vs Sharpe Ratio',
                                        'Parameter Impact on Sharpe Ratio',
                                        'Trade Count Distribution'))

        # Sharpe ratio distribution
        fig.add_trace(go.Histogram(x=results_df['sharpe_ratio'],
                                 name='Sharpe Ratio'),
                     row=1, col=1)

        # Win rate vs Sharpe ratio
        fig.add_trace(go.Scatter(x=results_df['win_rate'],
                               y=results_df['sharpe_ratio'],
                               mode='markers',
                               name='Win Rate vs Sharpe'),
                     row=1, col=2)

        # Parameter impact
def create_optimization_report(self, results_df: pd.DataFrame):
        """Create and save optimization report with visualizations"""
        if results_df is None or len(results_df) == 0:
            self.logger.main_logger.warning("No results to create report")
            return

        # Create report directory
        report_path = self.base_path / f"{self.symbol}_{self.timeframe}_report"
        report_path.mkdir(exist_ok=True)

        # Create performance visualization
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Sharpe Ratio Distribution',
                                        'Win Rate vs Sharpe Ratio',
                                        'Parameter Impact on Sharpe Ratio',
                                        'Trade Count Distribution'))

        # Sharpe ratio distribution
        fig.add_trace(go.Histogram(x=results_df['sharpe_ratio'],
                                 name='Sharpe Ratio'),
                     row=1, col=1)

        # Win rate vs Sharpe ratio
        fig.add_trace(go.Scatter(x=results_df['win_rate'],
                               y=results_df['sharpe_ratio'],
                               mode='markers',
                               name='Win Rate vs Sharpe'),
                     row=1, col=2)

        # Parameter impact plots
        for param in ['wavy_period', 'tunnel_period1', 'tunnel_period2']:
            fig.add_trace(go.Box(x=results_df[param],
                               y=results_df['sharpe_ratio'],
                               name=param),
                         row=2, col=1)

        # Trade count distribution
        fig.add_trace(go.Histogram(x=results_df['total_signals'],
                                 name='Trade Count'),
                     row=2, col=2)

        # Update layout
        fig.update_layout(height=800, width=1200,
                         title_text=f"Optimization Results for {self.symbol} {self.timeframe}")

        # Save plot
        fig.write_html(report_path / "optimization_results.html")

        # Save top results to CSV
        top_results = results_df.nlargest(20, 'sharpe_ratio')
        top_results.to_csv(report_path / "top_results.csv")

        # Create summary text report
        with open(report_path / "summary_report.txt", "w") as f:
            f.write(f"Optimization Summary for {self.symbol} {self.timeframe}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Best Parameters:\n")
            best_params = results_df.nlargest(1, 'sharpe_ratio').iloc[0]
            for param in self.param_ranges.keys():
                f.write(f"{param}: {best_params[param]}\n")

            f.write("\nPerformance Metrics:\n")
            f.write(f"Sharpe Ratio: {best_params['sharpe_ratio']:.4f}\n")
            f.write(f"Win Rate: {best_params['win_rate']:.2%}\n")
            f.write(f"Total Signals: {best_params['total_signals']}\n")
            f.write(f"Average Profit/Loss: {best_params['avg_profit_loss']:.4f}\n")

            f.write("\nOptimization Statistics:\n")
            f.write(f"Total Combinations Tested: {len(results_df)}\n")
            f.write(f"Sharpe Ratio Range: {results_df['sharpe_ratio'].min():.4f} to {results_df['sharpe_ratio'].max():.4f}\n")
            f.write(f"Win Rate Range: {results_df['win_rate'].min():.2%} to {results_df['win_rate'].max():.2%}\n")

def main():
    """Main execution function"""
    # Configuration
    symbol = "XAUUSD"  # Can be changed to any symbol
    timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    # Process each timeframe
    for timeframe in timeframes:
        try:
            print(f"\nOptimizing {symbol} on {timeframe}")

            # Initialize optimizer
            optimizer = WavyTunnelOptimizer(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # Run optimization
            best_params, results_df = optimizer.optimize_parallel()

            if best_params is not None:
                print(f"\nBest parameters found for {symbol} {timeframe}:")
                print(f"Wavy Period: {best_params['wavy_period']}")
                print(f"Tunnel Period 1: {best_params['tunnel_period1']}")
                print(f"Tunnel Period 2: {best_params['tunnel_period2']}")
                print(f"Min Gap Second: {best_params['min_gap_second']}")
                print(f"Max Zone Percentage: {best_params['max_zone_percentage']:.2f}")
                print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.4f}")
                print(f"Win Rate: {best_params['win_rate']:.2%}")
                print(f"Total Signals: {best_params['total_signals']}")

                # Create report
                optimizer.create_optimization_report(results_df)

            else:
                print(f"No valid results found for {symbol} {timeframe}")

        except Exception as e:
            print(f"Error optimizing {symbol} {timeframe}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()
        print("\nOptimization process completed")