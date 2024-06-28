import unittest
from unittest.mock import patch
import pandas as pd
from backtesting.backtest import run_backtest

class TestRunBacktest(unittest.TestCase):

    @patch('backtesting.backtest.execute_trade')
    @patch('backtesting.backtest.calculate_ema')
    @patch('backtesting.backtest.detect_peaks_and_dips')
    @patch('backtesting.backtest.generate_trade_signal')
    @patch('backtesting.backtest.check_entry_conditions')
    def test_max_trades_per_day(self, mock_check_entry_conditions, mock_generate_trade_signal, mock_detect_peaks_and_dips, mock_calculate_ema, mock_execute_trade):
        # Prepare mock data
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'time': dates,
            'open': [1.1 + i*0.0001 for i in range(100)],
            'high': [1.105 + i*0.0001 for i in range(100)],
            'low': [1.095 + i*0.0001 for i in range(100)],
            'close': [1.1 + i*0.0001 for i in range(100)]
        })

        # Mock return values
        mock_calculate_ema.side_effect = lambda x, y: x
        mock_detect_peaks_and_dips.return_value = (pd.Series([0] * len(data)), pd.Series([0] * len(data)))
        mock_generate_trade_signal.return_value = ['buy'] * len(data)
        mock_check_entry_conditions.side_effect = [(True, False)] * len(data)

        initial_balance = 10000
        risk_percent = 1
        min_take_profit = 10
        max_loss_per_day = 100
        starting_equity = 10000
        stop_loss_pips = 10
        pip_value = 10
        max_trades_per_day = 5

        results = run_backtest(
            symbol='TEST',
            data=data,
            initial_balance=initial_balance,
            risk_percent=risk_percent,
            min_take_profit=min_take_profit,
            max_loss_per_day=max_loss_per_day,
            starting_equity=starting_equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            max_trades_per_day=max_trades_per_day
        )

        self.assertLessEqual(results['num_trades'], max_trades_per_day)

    @patch('backtesting.backtest.execute_trade')
    @patch('backtesting.backtest.calculate_ema')
    @patch('backtesting.backtest.detect_peaks_and_dips')
    @patch('backtesting.backtest.generate_trade_signal')
    @patch('backtesting.backtest.check_entry_conditions')
    def test_max_loss_per_day(self, mock_check_entry_conditions, mock_generate_trade_signal, mock_detect_peaks_and_dips, mock_calculate_ema, mock_execute_trade):
        # Prepare mock data
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'time': dates,
            'open': [1.1 + i*0.0001 for i in range(100)],
            'high': [1.105 + i*0.0001 for i in range(100)],
            'low': [1.095 + i*0.0001 for i in range(100)],
            'close': [1.1 + i*0.0001 for i in range(100)]
        })

        # Mock return values
        mock_calculate_ema.side_effect = lambda x, y: x
        mock_detect_peaks_and_dips.return_value = (pd.Series([0] * len(data)), pd.Series([0] * len(data)))
        mock_generate_trade_signal.return_value = ['buy'] * len(data)
        mock_check_entry_conditions.side_effect = [(True, False)] * len(data)

        initial_balance = 10000
        risk_percent = 1
        min_take_profit = 10
        max_loss_per_day = 100
        starting_equity = 10000
        stop_loss_pips = 10
        pip_value = 10
        max_trades_per_day = None  # Not limiting trades in this test

        results = run_backtest(
            symbol='TEST',
            data=data,
            initial_balance=initial_balance,
            risk_percent=risk_percent,
            min_take_profit=min_take_profit,
            max_loss_per_day=max_loss_per_day,
            starting_equity=starting_equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            max_trades_per_day=max_trades_per_day
        )

        self.assertLessEqual(results['total_profit'], -max_loss_per_day)  # Ensure max loss per day is respected

    @patch('backtesting.backtest.execute_trade')
    @patch('backtesting.backtest.calculate_ema')
    @patch('backtesting.backtest.detect_peaks_and_dips')
    @patch('backtesting.backtest.generate_trade_signal')
    @patch('backtesting.backtest.check_entry_conditions')
    def test_initial_balance(self, mock_check_entry_conditions, mock_generate_trade_signal, mock_detect_peaks_and_dips, mock_calculate_ema, mock_execute_trade):
        # Prepare mock data
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'time': dates,
            'open': [1.1 + i*0.0001 for i in range(100)],
            'high': [1.105 + i*0.0001 for i in range(100)],
            'low': [1.095 + i*0.0001 for i in range(100)],
            'close': [1.1 + i*0.0001 for i in range(100)]
        })

        # Mock return values
        mock_calculate_ema.side_effect = lambda x, y: x
        mock_detect_peaks_and_dips.return_value = (pd.Series([0] * len(data)), pd.Series([0] * len(data)))
        mock_generate_trade_signal.return_value = ['buy'] * len(data)
        mock_check_entry_conditions.side_effect = [(True, False)] * len(data)

        initial_balance = 10000
        risk_percent = 1
        min_take_profit = 10
        max_loss_per_day = 100
        starting_equity = 10000
        stop_loss_pips = 10
        pip_value = 10
        max_trades_per_day = None

        results = run_backtest(
            symbol='TEST',
            data=data,
            initial_balance=initial_balance,
            risk_percent=risk_percent,
            min_take_profit=min_take_profit,
            max_loss_per_day=max_loss_per_day,
            starting_equity=starting_equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            max_trades_per_day=max_trades_per_day
        )

        self.assertEqual(results['total_profit'], 0)  # With mocked data, profit should be zero initially
        self.assertEqual(results['num_trades'], 50)  # With 100 data points and 50 buys, each row triggers a trade

    @patch('backtesting.backtest.execute_trade')
    @patch('backtesting.backtest.calculate_ema')
    @patch('backtesting.backtest.detect_peaks_and_dips')
    @patch('backtesting.backtest.generate_trade_signal')
    @patch('backtesting.backtest.check_entry_conditions')
    def test_risk_management(self, mock_check_entry_conditions, mock_generate_trade_signal, mock_detect_peaks_and_dips, mock_calculate_ema, mock_execute_trade):
        # Prepare mock data
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'time': dates,
            'open': [1.1 + i*0.0001 for i in range(100)],
            'high': [1.105 + i*0.0001 for i in range(100)],
            'low': [1.095 + i*0.0001 for i in range(100)],
            'close': [1.1 + i*0.0001 for i in range(100)]
        })

        # Mock return values
        mock_calculate_ema.side_effect = lambda x, y: x
        mock_detect_peaks_and_dips.return_value = (pd.Series([0] * len(data)), pd.Series([0] * len(data)))
        mock_generate_trade_signal.return_value = ['buy'] * len(data)
        mock_check_entry_conditions.side_effect = [(True, False)] * len(data)

        initial_balance = 10000
        risk_percent = 1
        min_take_profit = 10
        max_loss_per_day = 100
        starting_equity = 10000
        stop_loss_pips = 0  # This should trigger the stop loss validation
        pip_value = 10
        max_trades_per_day = None

        with self.assertLogs(level='ERROR') as log:
            results = run_backtest(
                symbol='TEST',
                data=data,
                initial_balance=initial_balance,
                risk_percent=risk_percent,
                min_take_profit=min_take_profit,
                max_loss_per_day=max_loss_per_day,
                starting_equity=starting_equity,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value,
                max_trades_per_day=max_trades_per_day
            )
            self.assertIn('stop_loss_pips is zero. This value must not be zero.', log.output[0])

    @patch('backtesting.backtest.execute_trade')
    @patch('backtesting.backtest.calculate_ema')
    @patch('backtesting.backtest.detect_peaks_and_dips')
    @patch('backtesting.backtest.generate_trade_signal')
    @patch('backtesting.backtest.check_entry_conditions')
    def test_pip_value_validation(self, mock_check_entry_conditions, mock_generate_trade_signal, mock_detect_peaks_and_dips, mock_calculate_ema, mock_execute_trade):
        # Prepare mock data
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'time': dates,
            'open': [1.1 + i*0.0001 for i in range(100)],
            'high': [1.105 + i*0.0001 for i in range(100)],
            'low': [1.095 + i*0.0001 for i in range(100)],
            'close': [1.1 + i*0.0001 for i in range(100)]
        })

        # Mock return values
        mock_calculate_ema.side_effect = lambda x, y: x
        mock_detect_peaks_and_dips.return_value = (pd.Series([0] * len(data)), pd.Series([0] * len(data)))
        mock_generate_trade_signal.return_value = ['buy'] * len(data)
        mock_check_entry_conditions.side.effect = [(True, False)] * len(data)

        initial_balance = 10000
        risk_percent = 1
        min_take_profit = 10
        max_loss_per_day = 100
        starting_equity = 10000
        stop_loss_pips = 10
        pip_value = 0  # This should trigger the pip value validation
        max_trades_per_day = None

        with self.assertLogs(level='ERROR') as log:
            results = run_backtest(
                symbol='TEST',
                data=data,
                initial_balance=initial_balance,
                risk_percent=risk_percent,
                min_take_profit=min_take_profit,
                max_loss_per_day=max_loss_per_day,
                starting_equity=starting_equity,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value,
                max_trades_per_day=max_trades_per_day
            )
            self.assertIn('pip_value is zero. This value must not be zero.', log.output[0])

if __name__ == '__main__':
    unittest.main()
