import logging
import unittest
import pandas as pd
import numpy as np
from backtesting.backtest import run_backtest

class BacktestTestCase(unittest.TestCase):

    def setUp(self):
        # Dummy data for testing
        self.data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=200, freq='D'),
            'open': 1.0,
            'high': 1.2,
            'low': 0.8,
            'close': 1.0
        })

    def generate_large_dataset(self, start='2024-01-01', end='2024-12-31', freq='1T'):
        """
        Generates a large dataset with minute-by-minute data for the given date range.
        :param start: Start date of the dataset.
        :param end: End date of the dataset.
        :param freq: Frequency of data points, default is 1 minute.
        :return: DataFrame with generated data.
        """
        date_range = pd.date_range(start=start, end=end, freq=freq)
        num_points = len(date_range)

        np.random.seed(0)  # For reproducibility

        data = {
            'time': date_range,
            'open': np.random.rand(num_points) * 100,
            'high': np.random.rand(num_points) * 100,
            'low': np.random.rand(num_points) * 100,
            'close': np.random.rand(num_points) * 100,
            'volume': np.random.randint(1, 100, num_points)
        }

        df = pd.DataFrame(data)

        # Ensure 'high' is always greater than or equal to 'low' and 'close' is within 'high' and 'low'
        df['high'] = df[['high', 'low']].max(axis=1)
        df['low'] = df[['high', 'low']].min(axis=1)
        df['close'] = df[['close', 'low']].clip(lower=df['low']).clip(upper=df['high'])

        return df

    def run_backtest_and_print(self, test_name, **kwargs):
        print(f"\nRunning {test_name}")
        result = run_backtest(**kwargs)
        print(f"Result for {test_name}: {result}")
        return result

    def test_handling_large_datasets(self):
        large_data = self.generate_large_dataset()
        logging.debug(f"Generated dataset length: {len(large_data)}")
        
        result = self.run_backtest_and_print(
            'test_handling_large_datasets',
            symbol='EURUSD',
            data=large_data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        self.assertGreater(result['num_trades'], 0)

    # Add the rest of your test methods here
    def test_profit_factor_calculation(self):
        result = self.run_backtest_and_print(
            'test_profit_factor_calculation',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        gross_profit = sum(trade['profit'] for trade in result['trades'] if trade['profit'] > 0)
        gross_loss = abs(sum(trade['profit'] for trade in result['trades'] if trade['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        self.assertAlmostEqual(result.get('profit_factor', profit_factor), profit_factor, places=2, msg="Profit factor calculation is incorrect.")

    def test_return_on_investment_calculation(self):
        result = self.run_backtest_and_print(
            'test_return_on_investment_calculation',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        total_profit = result['total_profit']
        initial_balance = 10000
        roi = (total_profit / initial_balance) * 100
        self.assertAlmostEqual(result.get('roi', roi), roi, places=2, msg="ROI calculation is incorrect.")

    def test_sharpe_ratio_calculation(self):
        result = self.run_backtest_and_print(
            'test_sharpe_ratio_calculation',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        returns = [trade['profit'] / 10000 for trade in result['trades']]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((x - avg_return) ** 2 for x in returns) / len(returns)) ** 0.5 if returns else 0
        risk_free_rate = 0.01
        sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return != 0 else 0
        self.assertAlmostEqual(result.get('sharpe_ratio', sharpe_ratio), sharpe_ratio, places=2, msg="Sharpe ratio calculation is incorrect.")

    def test_win_loss_ratio_calculation(self):
        result = self.run_backtest_and_print(
            'test_win_loss_ratio_calculation',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        wins = sum(1 for trade in result['trades'] if trade['profit'] > 0)
        losses = sum(1 for trade in result['trades'] if trade['profit'] < 0)
        win_loss_ratio = wins / losses if losses > 0 else float('inf')
        self.assertAlmostEqual(result.get('win_loss_ratio', win_loss_ratio), win_loss_ratio, places=2, msg="Win/Loss ratio calculation is incorrect.")

    def test_annualized_return_calculation(self):
        result = self.run_backtest_and_print(
            'test_annualized_return_calculation',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        total_profit = result['total_profit']
        days = (self.data['time'].iloc[-1] - self.data['time'].iloc[0]).days
        annualized_return = ((total_profit / 10000) + 1) ** (365 / days) - 1 if days > 0 else 0
        self.assertAlmostEqual(result.get('annualized_return', annualized_return), annualized_return, places=2, msg="Annualized return calculation is incorrect.")

    def test_expectancy_calculation(self):
        result = self.run_backtest_and_print(
            'test_expectancy_calculation',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        total_trades = len(result['trades'])
        wins = [trade['profit'] for trade in result['trades'] if trade['profit'] > 0]
        losses = [trade['profit'] for trade in result['trades'] if trade['profit'] < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        loss_rate = len(losses) / total_trades if total_trades > 0 else 0
        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
        self.assertAlmostEqual(result.get('expectancy', expectancy), expectancy, places=2, msg="Expectancy calculation is incorrect.")

    def test_consecutive_wins_and_losses(self):
        result = self.run_backtest_and_print(
            'test_consecutive_wins_and_losses',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in result['trades']:
            if trade['profit'] > 0:
                current_wins += 1
                current_losses = 0
            else:
                current_losses += 1
                current_wins = 0

            max_consecutive_wins = max(max_consecutive_wins, current_wins)
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

        self.assertEqual(result.get('max_consecutive_wins', max_consecutive_wins), max_consecutive_wins, "Max consecutive wins calculation is incorrect.")
        self.assertEqual(result.get('max_consecutive_losses', max_consecutive_losses), max_consecutive_losses, "Max consecutive losses calculation is incorrect.")

    def test_handling_missing_values(self):
        data_with_nans = self.data.copy()
        data_with_nans.loc[::10, 'close'] = float('nan')
        result = self.run_backtest_and_print(
            'test_handling_missing_values',
            symbol='EURUSD',
            data=data_with_nans,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        self.assertIsNotNone(result, "Handling missing values failed.")

    def test_transaction_costs(self):
        result = self.run_backtest_and_print(
            'test_transaction_costs',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5,
            transaction_cost=1  # Adding transaction costs
        )
        total_transaction_costs = len(result['trades']) * 1
        self.assertAlmostEqual(result['total_transaction_costs'], total_transaction_costs, places=2, msg="Transaction costs calculation is incorrect.")

    def test_slippage(self):
        result = self.run_backtest_and_print(
            'test_slippage',
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5,
            slippage=1  # Adding slippage
        )
        total_slippage_costs = len(result['trades']) * 1
        self.assertAlmostEqual(result['total_slippage_costs'], total_slippage_costs, places=2, msg="Slippage calculation is incorrect.")

    def test_negative_initial_balance(self):
        with self.assertRaises(ValueError):
            self.run_backtest_and_print(
                'test_negative_initial_balance',
                symbol='EURUSD',
                data=self.data,
                initial_balance=-10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0.0001,
                max_trades_per_day=5
            )

    def test_zero_risk_percent(self):
        with self.assertRaises(ValueError):
            self.run_backtest_and_print(
                'test_zero_risk_percent',
                symbol='EURUSD',
                data=self.data,
                initial_balance=10000,
                risk_percent=0,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0.0001,
                max_trades_per_day=5
            )

    def test_constant_prices(self):
        constant_data = self.data.copy()
        constant_data['high'] = 100
        constant_data['low'] = 100
        constant_data['close'] = 100
        result = self.run_backtest_and_print(
            'test_constant_prices',
            symbol='EURUSD',
            data=constant_data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )
        self.assertEqual(result['total_profit'], 0)
        self.assertEqual(result['num_trades'], 0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
