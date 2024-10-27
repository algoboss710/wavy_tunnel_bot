import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict
import json
import os
import concurrent.futures
import time
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

class OptimizationResult:
    def __init__(self, config: Dict, summary: Dict):
        self.config = config
        self.summary = summary
        self.net_profit = summary.get('net_profit', 0)
        self.total_trades = (
            summary.get('trades_executed_long', 0) +
            summary.get('trades_executed_short', 0) +
            summary.get('trades_executed_second_long', 0) +
            summary.get('trades_executed_second_short', 0)
        )
        self.win_rate = summary.get('win_rate', 0)

class EnhancedStrategyOptimizer:
    def __init__(self,
                 start_date: datetime,
                 end_date: datetime,
                 symbol: str = "XAUUSD",
                 timeframe: str = "H4",
                 base_path: str = "optimization_results"):

        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.timeframe = timeframe
        self.base_path = Path(base_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = self.base_path / self.run_id
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Parameter ranges for optimization
        self.parameter_ranges = {
            'lookback_values': [50, 75, 100, 125, 150],
            'threshold_values': [0.0005, 0.001, 0.002, 0.003, 0.004],
            'tp_combinations': [
                {"levels": [0.2, 0.4, 0.6, 0.8], "quantities": [0.6, 0.15, 0.15, 0.1]},
                {"levels": [0.15, 0.3, 0.45, 0.6], "quantities": [0.4, 0.3, 0.2, 0.1]},
                {"levels": [0.25, 0.5, 0.75, 1.0], "quantities": [0.7, 0.1, 0.1, 0.1]}
            ],
            'atr_multiplier_values': [1.5, 2.0, 2.5, 3.0]
        }

        # Initialize MT5
        if not mt5.initialize():
            raise Exception("MetaTrader5 initialization failed")

        self.logger.info(f"Initialized optimizer for {symbol} {timeframe}")
        self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    def setup_logging(self):
        """Setup enhanced logging"""
        log_file = self.results_path / "optimization.log"
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger = logging.getLogger(f"optimizer_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def generate_configurations(self) -> List[Dict]:
        """Generate all parameter combinations for testing"""
        configs = []
        total_combinations = (
            len(self.parameter_ranges['lookback_values']) *
            len(self.parameter_ranges['threshold_values']) *
            len(self.parameter_ranges['tp_combinations']) *
            len(self.parameter_ranges['atr_multiplier_values'])
        )

        self.logger.info(f"Generating {total_combinations} parameter combinations")

        for lookback in self.parameter_ranges['lookback_values']:
            for threshold in self.parameter_ranges['threshold_values']:
                for tp_combo in self.parameter_ranges['tp_combinations']:
                    for atr_mult in self.parameter_ranges['atr_multiplier_values']:
                        config = {
                            "currency_pair": self.symbol,
                            "timeframe": self.timeframe,
                            "wavy_ema": 34,
                            "tunnel_ema1": 144,
                            "tunnel_ema2": 169,
                            "atr_period": 14,
                            "rsi_period": 14,
                            "rsi_upper": 70,
                            "rsi_lower": 30,
                            "apply_rsi_filter": True,
                            "lot_size": 0.01,
                            "enable_second_strategy": True,
                            "max_allow_into_zone": 0.25,
                            "peak_dip_lookback": lookback,
                            "peak_dip_threshold": threshold,
                            "tp_levels": tp_combo["levels"],
                            "tp_quantities": tp_combo["quantities"],
                            "tp_atr_multiplier": atr_mult
                        }
                        configs.append(config)

        return configs

    def run_parallel_optimization(self) -> Dict:
        """Run optimization using parallel processing"""
        configs = self.generate_configurations()
        cpu_count = os.cpu_count()
        max_workers = max(1, cpu_count - 1)

        start_time = time.time()
        results = []

        self.logger.info(f"Starting optimization with {len(configs)} combinations using {max_workers} workers")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {executor.submit(self.run_backtest, config): config
                              for config in configs}

            completed = 0
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    summary = future.result()
                    if summary:
                        results.append(OptimizationResult(config, summary))
                        self.log_interim_result(config, summary, completed, len(configs))

                    completed += 1
                    if completed % 10 == 0:
                        self.log_progress(completed, len(configs), start_time)

                except Exception as e:
                    self.logger.error(f"Test failed for config: {config}")
                    self.logger.error(f"Error: {str(e)}")

        total_time = time.time() - start_time
        self.logger.info(f"Optimization completed in {total_time/60:.1f} minutes")

        return self.analyze_results(results)

    def log_interim_result(self, config: Dict, summary: Dict, completed: int, total: int):
        """Log detailed interim results"""
        self.logger.info(f"\nTest #{completed + 1}/{total} completed:")
        self.logger.info("Configuration:")
        for key, value in config.items():
            if key in ['peak_dip_lookback', 'peak_dip_threshold', 'tp_atr_multiplier']:
                self.logger.info(f"- {key}: {value}")

        self.logger.info("Performance:")
        self.logger.info(f"- Bars analyzed: {summary.get('bars_analyzed', 0)}")
        self.logger.info(f"- Total trades: {summary.get('total_trades', 0)}")
        self.logger.info(f"- Primary strategy trades: {summary.get('trades_executed_long', 0) + summary.get('trades_executed_short', 0)}")
        self.logger.info(f"- Secondary strategy trades: {summary.get('trades_executed_second_long', 0) + summary.get('trades_executed_second_short', 0)}")
        self.logger.info(f"- Net profit: ${summary.get('net_profit', 0):,.2f}")

    def log_progress(self, completed: int, total: int, start_time: float):
        """Log progress with time estimation"""
        elapsed = time.time() - start_time
        avg_time = elapsed / completed
        remaining = (total - completed) * avg_time

        self.logger.info(
            f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
            f"Est. remaining time: {remaining/60:.1f} minutes"
        )

    def analyze_results(self, results: List[OptimizationResult]) -> Dict:
        """Analyze optimization results"""
        # Sort by net profit
        sorted_results = sorted(results, key=lambda x: x.net_profit, reverse=True)

        # Generate analysis
        analysis = {
            'best_results': self.format_top_results(sorted_results[:10]),
            'statistics': self.calculate_statistics(results),
            'parameter_analysis': self.analyze_parameters(results)
        }

        # Save results
        self.save_results(analysis)
        self.create_visualizations(results)

        return analysis

    def format_top_results(self, top_results: List[OptimizationResult]) -> List[Dict]:
        """Format top results for output"""
        formatted_results = []
        for i, result in enumerate(top_results, 1):
            formatted_results.append({
                'rank': i,
                'config': {
                    key: value for key, value in result.config.items()
                    if key in ['peak_dip_lookback', 'peak_dip_threshold',
                             'tp_levels', 'tp_quantities', 'tp_atr_multiplier']
                },
                'performance': {
                    'net_profit': result.net_profit,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'primary_trades': result.summary.get('trades_executed_long', 0) +
                                    result.summary.get('trades_executed_short', 0),
                    'secondary_trades': result.summary.get('trades_executed_second_long', 0) +
                                      result.summary.get('trades_executed_second_short', 0)
                }
            })
        return formatted_results

    def calculate_statistics(self, results: List[OptimizationResult]) -> Dict:
        """Calculate comprehensive statistics"""
        net_profits = [r.net_profit for r in results]
        total_trades = [r.total_trades for r in results]

        return {
            'profit_statistics': {
                'mean': np.mean(net_profits),
                'median': np.median(net_profits),
                'std': np.std(net_profits),
                'min': min(net_profits),
                'max': max(net_profits)
            },
            'trade_statistics': {
                'mean': np.mean(total_trades),
                'median': np.median(total_trades),
                'std': np.std(total_trades),
                'min': min(total_trades),
                'max': max(total_trades)
            },
            'total_configurations_tested': len(results)
        }

    def analyze_parameters(self, results: List[OptimizationResult]) -> Dict:
        """Analyze parameter effectiveness"""
        parameter_analysis = defaultdict(lambda: defaultdict(list))

        for result in results:
            parameter_analysis['peak_dip_lookback'][result.config['peak_dip_lookback']].append(result.net_profit)
            parameter_analysis['peak_dip_threshold'][result.config['peak_dip_threshold']].append(result.net_profit)
            parameter_analysis['tp_atr_multiplier'][result.config['tp_atr_multiplier']].append(result.net_profit)

        # Calculate average profit for each parameter value
        analysis = {}
        for param, values in parameter_analysis.items():
            analysis[param] = {
                str(value): np.mean(profits)
                for value, profits in values.items()
            }

        return analysis

    def create_visualizations(self, results: List[OptimizationResult]):
        """Create visualization plots"""
        # Profit distribution
        plt.figure(figsize=(10, 6))
        profits = [r.net_profit for r in results]
        sns.histplot(profits, kde=True)
        plt.title('Distribution of Net Profits')
        plt.xlabel('Net Profit ($)')
        plt.ylabel('Frequency')
        plt.savefig(self.results_path / 'profit_distribution.png')
        plt.close()

        # Parameter analysis plots
        self.create_parameter_analysis_plots(results)

    def create_parameter_analysis_plots(self, results: List[OptimizationResult]):
        """Create parameter analysis plots"""
        params_to_plot = ['peak_dip_lookback', 'peak_dip_threshold', 'tp_atr_multiplier']

        for param in params_to_plot:
            plt.figure(figsize=(10, 6))
            param_values = [r.config[param] for r in results]
            profits = [r.net_profit for r in results]

            sns.scatterplot(x=param_values, y=profits)
            plt.title(f'Net Profit vs {param}')
            plt.xlabel(param)
            plt.ylabel('Net Profit ($)')
            plt.savefig(self.results_path / f'{param}_analysis.png')
            plt.close()

    def save_results(self, analysis: Dict):
        """Save optimization results"""
        # Save full results
        results_file = self.results_path / 'optimization_results.json'
        with open(results_file, 'w') as f:
            json.dump(analysis, f, indent=4)

        # Save summary of best results
        summary_file = self.results_path / 'best_results_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=== OPTIMIZATION RESULTS ===\n\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(f"Date Range: {self.start_date.date()} to {self.end_date.date()}\n\n")

            f.write("Top 5 Configurations:\n\n")
            for result in analysis['best_results'][:5]:
                f.write(f"#{result['rank']} Configuration (Profit: ${result['performance']['net_profit']:,.2f})\n")
                f.write("Parameters:\n")
                for key, value in result['config'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("Performance:\n")
                for key, value in result['performance'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

        self.logger.info(f"Results saved to {self.results_path}")

def main():
    """Main execution function"""
    # Set date range for optimization
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    symbol = "XAUUSD"
    timeframe = "H4"

    try:
        # Initialize optimizer
        optimizer = EnhancedStrategyOptimizer(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            timeframe=timeframe
        )

        # Run optimization
        optimizer.logger.info("Starting optimization process")
        results = optimizer.run_parallel_optimization()

        # Print summary to console
        print("\nOptimization Complete!")
        print("\nTop 5 Configurations:")

        for result in results['best_results'][:5]:
            print(f"\n#{result['rank']} Configuration (Profit: ${result['performance']['net_profit']:,.2f})")
            print("Parameters:")
            for key, value in result['config'].items():
                print(f"- {key}: {value}")
            print("Performance:")
            for key, value in result['performance'].items():
                if key != 'net_profit':  # Already printed above
                    print(f"- {key}: {value}")

        print("\nOptimization Statistics:")
        print(f"Total configurations tested: {results['statistics']['total_configurations_tested']}")
        print(f"Mean profit: ${results['statistics']['profit_statistics']['mean']:,.2f}")
        print(f"Best profit: ${results['statistics']['profit_statistics']['max']:,.2f}")
        print(f"Average trades per configuration: {results['statistics']['trade_statistics']['mean']:.0f}")

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        logging.error(f"Optimization failed: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()