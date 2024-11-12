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

    def log_progress(self, current: int, total: int, elapsed_time: float, remaining_time: float):
        progress = (current / total) * 100
        self.progress_logger.info(
            f"Progress: {current}/{total} ({progress:.2f}%) - "
            f"Elapsed Time: {elapsed_time:.2f}s - "
            f"Estimated Remaining: {remaining_time:.2f}s"
        )

    def log_results(self, results: Dict):
        self.results_logger.info(f"Optimization Results:\n{results}")

class WavyTunnelExitOptimizer:
    """Optimizer class for Wavy Tunnel exit strategies"""
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

        # Initialize logger
        self.logger = OptimizationLogger(self.base_path, symbol, timeframe)

        # Parameter ranges for optimization
        self.param_ranges = self._setup_param_ranges()

    def _setup_param_ranges(self) -> Dict:
        """Define parameter ranges with reduced combinations"""
        return {
            # Take profit distribution parameters - reduced steps
            'tp1_lot_percent': range(40, 81, 10),  # 5 values
            'tp2_lot_percent': range(10, 31, 10),  # 3 values
            'tp3_lot_percent': range(5, 21, 5),    # 4 values
            'tp4_lot_percent': range(5, 21, 5),    # 4 values
            
            # Take profit weight multipliers - reduced steps
            'tp1_weight': np.arange(0.1, 0.4, 0.1),  # 3 values
            'tp2_weight': np.arange(0.3, 0.6, 0.1),  # 3 values
            'tp3_weight': np.arange(0.5, 0.8, 0.1),  # 3 values
            'tp4_weight': np.arange(0.7, 1.0, 0.1),  # 3 values
            
            # Stop loss parameters - reduced steps
            'wave_cross_buffer': np.arange(0, 0.002, 0.001),  # 2 values
            'tunnel_touch_buffer': np.arange(0, 0.002, 0.001)  # 2 values
        }

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection with error handling"""
        try:
            if not mt5.initialize():
                self.logger.main_logger.error("Failed to initialize MT5")
                return False
            return True
        except Exception as e:
            self.logger.main_logger.error(f"Error initializing MT5: {str(e)}")
            return False

    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch and preprocess market data with improved error handling"""
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
        """Preprocess market data with improved outlier handling"""
        try:
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove outliers using rolling median
            for col in ['high', 'low', 'close', 'open']:
                median = data[col].rolling(window=5, center=True).median()
                std = data[col].rolling(window=5, center=True).std()
                data[col] = data[col].where(
                    abs(data[col] - median) <= 3 * std,
                    median
                )
            
            return data
        except Exception as e:
            self.logger.main_logger.error(f"Error preprocessing data: {str(e)}")
            return data

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate entry signals using fixed parameters"""
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

            # Generate signals
            primary_longs = (data['open'] > wavy_max) & (wavy_min > tunnel_max)
            primary_shorts = (data['open'] < wavy_min) & (wavy_max < tunnel_min)
            
            secondary_longs = (
                (data['close'].shift(1) <= wavy_max.shift(1)) & 
                (data['close'] > wavy_max) &
                (data['close'] < tunnel_min) &
                ((tunnel_min - data['close']) > self.entry_params['min_gap_second']) &
                ((data['close'] - wavy_max) / (tunnel_min - wavy_max) <= self.entry_params['max_zone_percentage'])
            )
            
            secondary_shorts = (
                (data['close'].shift(1) >= wavy_min.shift(1)) & 
                (data['close'] < wavy_min) &
                (data['close'] > tunnel_max) &
                ((data['close'] - tunnel_max) > self.entry_params['min_gap_second']) &
                ((wavy_min - data['close']) / (wavy_min - tunnel_max) <= self.entry_params['max_zone_percentage'])
            )

            return (primary_longs | secondary_longs, 
                    primary_shorts | secondary_shorts,
                    pd.Series(primary_longs | primary_shorts, index=data.index))
        except Exception as e:
            self.logger.main_logger.error(f"Error generating signals: {str(e)}")
            return pd.Series(False, index=data.index), pd.Series(False, index=data.index), pd.Series(False, index=data.index)

    def _evaluate_params(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Evaluate a single parameter combination"""
        try:
            long_signals, short_signals, is_primary = self.generate_signals(data)
            
            results = {
                'total_trades': 0,
                'profitable_trades': 0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'avg_hold_time': 0.0,
                'primary_profit': 0.0,
                'secondary_profit': 0.0
            }
            
            trades = []
            current_position = None
            entry_price = 0
            entry_time = None
            
            for i in range(len(data)-1):
                if current_position is None:
                    # Check for new entry
                    if long_signals.iloc[i]:
                        current_position = 'long'
                        entry_price = data['open'].iloc[i]
                        entry_time = data.index[i]
                    elif short_signals.iloc[i]:
                        current_position = 'short'
                        entry_price = data['open'].iloc[i]
                        entry_time = data.index[i]
                else:
                    # Handle exit conditions
                    exit_price = None
                    exit_reason = None
                    
                    if current_position == 'long':
                        # Check take profit levels
                        for j, (lot_percent, weight) in enumerate(zip(
                            [params['tp1_lot_percent'], params['tp2_lot_percent'], 
                             params['tp3_lot_percent'], params['tp4_lot_percent']],
                            [params['tp1_weight'], params['tp2_weight'], 
                             params['tp3_weight'], params['tp4_weight']])):
                            
                            tp_level = entry_price * (1 + weight)
                            if data['high'].iloc[i] >= tp_level:
                                exit_price = tp_level
                                exit_reason = f'tp{j+1}'
                                break
                        
                        # Check stop loss (wave cross)
                        if exit_price is None and data['close'].iloc[i] < (entry_price * (1 - params['wave_cross_buffer'])):
                            exit_price = data['close'].iloc[i]
                            exit_reason = 'sl'
                    
                    elif current_position == 'short':
                        # Similar logic for short positions
                        for j, (lot_percent, weight) in enumerate(zip(
                            [params['tp1_lot_percent'], params['tp2_lot_percent'], 
                             params['tp3_lot_percent'], params['tp4_lot_percent']],
                            [params['tp1_weight'], params['tp2_weight'], 
                             params['tp3_weight'], params['tp4_weight']])):
                            
                            tp_level = entry_price * (1 - weight)
                            if data['low'].iloc[i] <= tp_level:
                                exit_price = tp_level
                                exit_reason = f'tp{j+1}'
                                break
                        
                        # Check stop loss (wave cross)
                        if exit_price is None and data['close'].iloc[i] > (entry_price * (1 + params['wave_cross_buffer'])):
                            exit_price = data['close'].iloc[i]
                            exit_reason = 'sl'
                    
                    # Record trade if exited
                    if exit_price is not None:
                        profit = (exit_price - entry_price) / entry_price if current_position == 'long' else (entry_price - exit_price) / entry_price
                        hold_time = (data.index[i] - entry_time).total_seconds() / 3600  # hours
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit,
                            'hold_time': hold_time,
                            'exit_reason': exit_reason,
                            'is_primary': is_primary.iloc[i]
                        })
                        
                        current_position = None
            
            # Calculate performance metrics
            if trades:
                df_trades = pd.DataFrame(trades)
                results['total_trades'] = len(df_trades)
                results['profitable_trades'] = len(df_trades[df_trades['profit'] > 0])
                results['avg_profit'] = df_trades['profit'].mean()
                results['max_drawdown'] = self._calculate_max_drawdown(df_trades['profit'])
                results['profit_factor'] = (
                    abs(df_trades[df_trades['profit'] > 0]['profit'].sum()) /
                    abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
                    if len(df_trades[df_trades['profit'] < 0]) > 0 else float('inf')
                )
                results['sharpe_ratio'] = (
                    df_trades['profit'].mean() / df_trades['profit'].std()
                    if len(df_trades) > 1 else 0
                )
                results['avg_hold_time'] = df_trades['hold_time'].mean()
                results['primary_profit'] = df_trades[df_trades['is_primary']]['profit'].mean()
                results['secondary_profit'] = df_trades[~df_trades['is_primary']]['profit'].mean()
            
            return results
            
        except Exception as e:
            self.logger.main_logger.error(f"Error evaluating parameters: {str(e)}")
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'avg_profit': 0.0,
                'max_drawdown': 1.0,
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
            self.logger.main_logger.error(f"Error calculating max drawdown: {str(e)}")
            return 1.0

    def create_optimization_report(self, results_df: pd.DataFrame):
        """Create and save optimization report with visualizations"""
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
                    'Sharpe Ratio vs Profit Factor',
                    'Take Profit Level Distribution',
                    'Hold Time vs Profit',
                    'Primary vs Secondary Strategy Performance',
                    'Max Drawdown Distribution'
                )
            )

            # Profit distribution
            fig.add_trace(
                go.Histogram(x=results_df['avg_profit'],
                            name='Average Profit'),
                row=1, col=1
            )

            # Sharpe vs Profit Factor
            fig.add_trace(
                go.Scatter(x=results_df['sharpe_ratio'],
                          y=results_df['profit_factor'],
                          mode='markers',
                          name='Risk-Return'),
                row=1, col=2
            )

            # Take profit level distribution
            tp_levels = pd.DataFrame({
                'TP1': results_df['tp1_weight'],
                'TP2': results_df['tp2_weight'],
                'TP3': results_df['tp3_weight'],
                'TP4': results_df['tp4_weight']
            })
            
            fig.add_trace(
                go.Box(x=tp_levels.values.flatten(),
                      name='TP Levels'),
                row=2, col=1
            )

            # Hold time vs Profit
            fig.add_trace(
                go.Scatter(x=results_df['avg_hold_time'],
                          y=results_df['avg_profit'],
                          mode='markers',
                          name='Hold Time Impact'),
                row=2, col=2
            )

            # Primary vs Secondary Performance
            fig.add_trace(
                go.Scatter(x=results_df['primary_profit'],
                          y=results_df['secondary_profit'],
                          mode='markers',
                          name='Strategy Comparison'),
                row=3, col=1
            )

            # Max Drawdown
            fig.add_trace(
                go.Histogram(x=results_df['max_drawdown'],
                            name='Max Drawdown'),
                row=3, col=2
            )

            # Update layout
            fig.update_layout(
                height=1200,
                width=1200,
                showlegend=True,
                title_text=f"Exit Strategy Optimization Results - {self.symbol} {self.timeframe}"
            )

            # Save visualizations
            fig.write_html(report_path / "exit_optimization_results.html")

            # Save top results to CSV
            results_df.sort_values('sharpe_ratio', ascending=False).head(20).to_csv(
                report_path / "top_results.csv"
            )

            # Create summary report
            with open(report_path / "summary_report.txt", "w") as f:
                f.write(f"Exit Strategy Optimization Summary - {self.symbol} {self.timeframe}\n")
                f.write("=" * 50 + "\n\n")
                
                best_params = results_df.nlargest(1, 'sharpe_ratio').iloc[0]
                
                f.write("Best Parameters:\n")
                f.write("-" * 20 + "\n")
                f.write("Take Profit Levels:\n")
                for i in range(1, 5):
                    f.write(f"TP{i}: {best_params[f'tp{i}_lot_percent']}% at {best_params[f'tp{i}_weight']*100:.1f}%\n")
                
                f.write(f"\nStop Loss Parameters:\n")
                f.write(f"Wave Cross Buffer: {best_params['wave_cross_buffer']*100:.3f}%\n")
                f.write(f"Tunnel Touch Buffer: {best_params['tunnel_touch_buffer']*100:.3f}%\n")
                
                f.write(f"\nPerformance Metrics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Trades: {best_params['total_trades']}\n")
                f.write(f"Win Rate: {best_params['profitable_trades']/best_params['total_trades']*100:.1f}%\n")
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

def main():
    """Main execution function"""
    # Define optimized entry parameters for each timeframe
    timeframe_entry_params = {
        "M5": {
            'wavy_period': 25,
            'tunnel_period1': 120,
            'tunnel_period2': 140,
            'min_gap_second': 10,
            'max_zone_percentage': 0.2
        },
        "M15": {
            'wavy_period': 30,
            'tunnel_period1': 130,
            'tunnel_period2': 150,
            'min_gap_second': 12,
            'max_zone_percentage': 0.22
        },
        "M30": {
            'wavy_period': 34,
            'tunnel_period1': 144,
            'tunnel_period2': 169,
            'min_gap_second': 15,
            'max_zone_percentage': 0.25
        },
        "H1": {
            'wavy_period': 38,
            'tunnel_period1': 160,
            'tunnel_period2': 185,
            'min_gap_second': 18,
            'max_zone_percentage': 0.28
        },
        "H4": {
            'wavy_period': 42,
            'tunnel_period1': 180,
            'tunnel_period2': 200,
            'min_gap_second': 20,
            'max_zone_percentage': 0.3
        },
        "D1": {
            'wavy_period': 45,
            'tunnel_period1': 200,
            'tunnel_period2': 220,
            'min_gap_second': 25,
            'max_zone_percentage': 0.35
        }
    }

    # Configuration
    symbol = "EURUSD"
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    # Create results directory with timestamp
    base_results_dir = Path(f"exit_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # Summary of all timeframe results
    all_results = {}

    for timeframe, entry_params in timeframe_entry_params.items():
        try:
            print(f"\nOptimizing exit strategy for {symbol} on {timeframe}")
            print(f"Using entry parameters: {entry_params}")
            
            optimizer = WavyTunnelExitOptimizer(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                entry_params=entry_params,
                base_path=str(base_results_dir / timeframe)
            )

            best_params, results_df = optimizer.optimize_parallel()
            
            if best_params is not None:
                print("\nBest Exit Parameters Found:")
                print("-" * 30)
                print("Take Profit Levels:")
                for i in range(1, 5):
                    print(f"TP{i}: {best_params[f'tp{i}_lot_percent']}% at {best_params[f'tp{i}_weight']*100:.1f}%")
                
                print("\nPerformance Metrics:")
                print(f"Total Trades: {best_params['total_trades']}")
                print(f"Win Rate: {best_params['profitable_trades']/best_params['total_trades']*100:.1f}%")
                print(f"Average Profit: {best_params['avg_profit']*100:.2f}%")
                print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
                
                all_results[timeframe] = {
                    'entry_params': entry_params,
                    'exit_params': best_params
                }
                
                optimizer.create_optimization_report(results_df)
            else:
                print(f"No valid results found for {symbol} {timeframe}")

        except Exception as e:
            print(f"Error optimizing {symbol} {timeframe}: {str(e)}")
            continue

    # Create summary report for all timeframes
    create_combined_summary_report(all_results, base_results_dir, symbol)

def create_combined_summary_report(all_results: Dict, base_dir: Path, symbol: str):
    """Create a summary report combining results from all timeframes"""
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
                f.write(f"  TP{i}: {exit_params[f'tp{i}_lot_percent']}% at {exit_params[f'tp{i}_weight']*100:.1f}%\n")
            
            f.write("\nPerformance Metrics:\n")
            f.write(f"  Total Trades: {exit_params['total_trades']}\n")
            f.write(f"  Win Rate: {exit_params['profitable_trades']/exit_params['total_trades']*100:.1f}%\n")
            f.write(f"  Average Profit: {exit_params['avg_profit']*100:.2f}%\n")
            f.write(f"  Sharpe Ratio: {exit_params['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Drawdown: {exit_params['max_drawdown']*100:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")

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