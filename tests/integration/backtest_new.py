import unittest
import pandas as pd
from backtesting.backtest import run_backtest
import logging

class TestRunBacktest(unittest.TestCase):

    def setUp(self):
        # Create sample data
        data = {
            'time': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'high': [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128],
            'low': [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108],
            'close': [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        }
        self.data = pd.DataFrame(data)

    def test_initial_balance(self):
        result = run_backtest(
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

        self.assertEqual(result['total_profit'], 0)
        self.assertEqual(result['num_trades'], 0)
        self.assertEqual(result['win_rate'], 0)
        self.assertEqual(result['max_drawdown'], 0)

    def test_max_loss_per_day(self):
        result = run_backtest(
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

        # Ensure result contains trades key
        self.assertIn('trades', result, "Result dictionary does not contain 'trades' key.")
        
        actual_loss = sum(trade['profit'] for trade in result['trades'] if 'profit' in trade and trade['profit'] < 0)
        expected_loss = -100  # Expecting that max loss per day is respected
        if result['trades']:
            self.assertLessEqual(actual_loss, expected_loss)
        else:
            self.assertEqual(actual_loss, 0)

    def test_pip_value_validation(self):
        with self.assertRaises(ZeroDivisionError):
            run_backtest(
                symbol='EURUSD',
                data=self.data,
                initial_balance=10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0,
                max_trades_per_day=5
            )

    def test_stop_loss_pips_validation(self):
        with self.assertRaises(ZeroDivisionError):
            run_backtest(
                symbol='EURUSD',
                data=self.data,
                initial_balance=10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=0,
                pip_value=0.0001,
                max_trades_per_day=5
            )

    def test_max_trades_per_day(self):
        result = run_backtest(
            symbol='EURUSD',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=1  # Limiting to 1 trade per day for this test
        )

        # Ensure result contains trades key
        self.assertIn('trades', result, "Result dictionary does not contain 'trades' key.")
        
        # Check that the number of trades does not exceed the max trades per day
        trades_per_day = {}
        for trade in result['trades']:
            day = trade['entry_time'].date()
            if day not in trades_per_day:
                trades_per_day[day] = 0
            trades_per_day[day] += 1
        
        for day, count in trades_per_day.items():
            self.assertLessEqual(count, 1, f"More than 1 trade executed on {day}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
