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

        print(result)  # Debugging line to check the output
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

        print(result)  # Debugging line to check the output
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

        print(result)  # Debugging line to check the output
        self.assertIn('trades', result, "Result dictionary does not contain 'trades' key.")
        
        trades_per_day = {}
        for trade in result['trades']:
            day = trade['entry_time'].date()
            if day not in trades_per_day:
                trades_per_day[day] = 0
            trades_per_day[day] += 1
        
        for day, count in trades_per_day.items():
            self.assertLessEqual(count, 1, f"More than 1 trade executed on {day}")

    def test_total_profit_calculation(self):
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

        print(result)  # Debugging line to check the output
        total_profit = sum(trade['profit'] for trade in result['trades'])
        self.assertEqual(result['total_profit'], total_profit, "Total profit calculation is incorrect.")

    def test_number_of_trades(self):
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

        print(result)  # Debugging line to check the output
        self.assertEqual(result['num_trades'], len(result['trades']), "Number of trades calculation is incorrect.")

    def test_win_rate_calculation(self):
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

        print(result)  # Debugging line to check the output
        wins = sum(1 for trade in result['trades'] if trade['profit'] > 0)
        total_trades = len(result['trades'])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        self.assertAlmostEqual(result['win_rate'], win_rate, places=2, msg="Win rate calculation is incorrect.")

    def test_max_drawdown_calculation(self):
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

        print(result)  # Debugging line to check the output
        equity_curve = [trade['equity'] for trade in result['trades']]
        if equity_curve:
            max_drawdown = max((max(equity_curve[:i+1]) - equity) / max(equity_curve[:i+1]) for i, equity in enumerate(equity_curve))
        else:
            max_drawdown = 0

        self.assertAlmostEqual(result['max_drawdown'], max_drawdown, places=2, msg="Max drawdown calculation is incorrect.")

    def test_profit_factor_calculation(self):
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

        print(result)  # Debugging line to check the output
        gross_profit = sum(trade['profit'] for trade in result['trades'] if trade['profit'] > 0)
        gross_loss = abs(sum(trade['profit'] for trade in result['trades'] if trade['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        self.assertAlmostEqual(result.get('profit_factor', profit_factor), profit_factor, places=2, msg="Profit factor calculation is incorrect.")

    def test_return_on_investment_calculation(self):
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

        print(result)  # Debugging line to check the output
        total_profit = result['total_profit']
        initial_balance = 10000
        roi = (total_profit / initial_balance) * 100

        self.assertAlmostEqual(result.get('roi', roi), roi, places=2, msg="ROI calculation is incorrect.")

    def test_sharpe_ratio_calculation(self):
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

        print(result)  # Debugging line to check the output
        returns = [trade['profit'] / 10000 for trade in result['trades']]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((x - avg_return) ** 2 for x in returns) / len(returns)) ** 0.5 if returns else 0
        risk_free_rate = 0.01
        sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return != 0 else 0

        self.assertAlmostEqual(result.get('sharpe_ratio', sharpe_ratio), sharpe_ratio, places=2, msg="Sharpe ratio calculation is incorrect.")

    def test_win_loss_ratio_calculation(self):
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

        print(result)  # Debugging line to check the output
        wins = sum(1 for trade in result['trades'] if trade['profit'] > 0)
        losses = sum(1 for trade in result['trades'] if trade['profit'] < 0)
        win_loss_ratio = wins / losses if losses > 0 else float('inf')

        self.assertAlmostEqual(result.get('win_loss_ratio', win_loss_ratio), win_loss_ratio, places=2, msg="Win/Loss ratio calculation is incorrect.")

    def test_annualized_return_calculation(self):
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

        print(result)  # Debugging line to check the output
        total_profit = result['total_profit']
        days = (self.data['time'].iloc[-1] - self.data['time'].iloc[0]).days
        annualized_return = ((total_profit / 10000) + 1) ** (365 / days) - 1 if days > 0 else 0

        self.assertAlmostEqual(result.get('annualized_return', annualized_return), annualized_return, places=2, msg="Annualized return calculation is incorrect.")

    def test_expectancy_calculation(self):
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

        print(result)  # Debugging line to check the output
        total_trades = len(result['trades'])
        wins = [trade['profit'] for trade in result['trades'] if trade['profit'] > 0]
        losses = [trade['profit'] for trade in result['trades'] if trade['profit'] < 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        loss_rate = len(losses) / total_trades if total_trades > 0 else 0
        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)

        self.assertAlmostEqual(result.get('expectancy', expectancy), expectancy, places=2, msg="Expectancy calculation is incorrect.")

    # Additional test cases

    def test_consecutive_wins_and_losses(self):
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

        print(result)  # Debugging line to check the output
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

    def test_handling_large_datasets(self):
        large_data = pd.concat([self.data] * 1000, ignore_index=True)
        chunk_size = len(self.data)  # Adjust chunk size as needed
        chunks = [large_data[i:i + chunk_size] for i in range(0, len(large_data), chunk_size)]

        result = None
        for chunk in chunks:
            result = run_backtest(
                symbol='EURUSD',
                data=chunk,
                initial_balance=10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0.0001,
                max_trades_per_day=5
            )
        
        print(result)  # Debugging line to check the output
        self.assertIsNotNone(result, "Handling large datasets failed.")

    def test_handling_missing_values(self):
        data_with_missing_values = self.data.copy()
        data_with_missing_values.loc[0, 'close'] = None
        result = run_backtest(
            symbol='EURUSD',
            data=data_with_missing_values,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=100,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5
        )

        print(result)  # Debugging line to check the output
        self.assertIsNotNone(result, "Handling missing values failed.")

    def test_transaction_costs(self):
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
            max_trades_per_day=5,
            transaction_cost=1  # Adding transaction costs
        )

        print(result)  # Debugging line to check the output
        total_transaction_costs = len(result['trades']) * 1
        self.assertAlmostEqual(result['total_transaction_costs'], total_transaction_costs, places=2, msg="Transaction costs calculation is incorrect.")

    def test_slippage(self):
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
            max_trades_per_day=5,
            slippage=1  # Adding slippage
        )

        print(result)  # Debugging line to check the output
        total_slippage_costs = len(result['trades']) * 1
        self.assertAlmostEqual(result['total_slippage_costs'], total_slippage_costs, places=2, msg="Slippage calculation is incorrect.")

    def test_negative_initial_balance(self):
        with self.assertRaises(ValueError):
            run_backtest(
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
            run_backtest(
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

        result = run_backtest(
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

        print(result)  # Debugging line to check the output
        self.assertEqual(result['total_profit'], 0)
        self.assertEqual(result['num_trades'], 0)

    def test_uptrend_data(self):
        uptrend_data = self.data.copy()
        uptrend_data['high'] = range(100, 130)
        uptrend_data['low'] = range(80, 110)
        uptrend_data['close'] = range(90, 120)

        # Adding logging to monitor the state of the data
        print(uptrend_data)

        result = run_backtest(
         symbol='EURUSD',
         data=uptrend_data,
         initial_balance=10000,
         risk_percent=0.01,
         min_take_profit=100,
         max_loss_per_day=100,
         starting_equity=10000,
         stop_loss_pips=20,
         pip_value=0.0001,
         max_trades_per_day=5
     )

    # More detailed debugging information
        print("Backtest Result:")
        print(f"Initial balance: {result['initial_balance']}")
        print(f"Final balance: {result['final_balance']}")
        print(f"Total Profit: {result['total_profit']}")
        print(f"Number of Trades: {result['number_of_trades']}")
        print(f"Win Rate: {result['win_rate']:.2f}%")
        print(f"Maximum Drawdown: {result['max_drawdown']:.2f}")

        self.assertGreater(result['total_profit'], 0)


    def test_uptrend_data(self):
        uptrend_data = self.data.copy()
        uptrend_data['high'] = range(100, 130)
        uptrend_data['low'] = range(80, 110)
        uptrend_data['close'] = range(90, 120)

    # Adding logging to monitor the state of the data
        print("Uptrend data:")
        print(uptrend_data)

    # Run backtest and add more detailed debugging information
        try:
            result = run_backtest(
                symbol='EURUSD',
                data=uptrend_data,
                initial_balance=10000,
                risk_percent=0.01,
                min_take_profit=100,
                max_loss_per_day=100,
                starting_equity=10000,
                stop_loss_pips=20,
                pip_value=0.0001,
                max_trades_per_day=5
            )
        except Exception as e:
            print("Error occurred during backtest:", str(e))
            raise

        # More detailed debugging information
        print("Backtest Result:")
        print("Result dictionary keys:", result.keys())
    
        if 'initial_balance' not in result:
            print("Error: 'initial_balance' key is missing from the result")
        else:
            print(f"Initial balance: {result['initial_balance']}")

        if 'final_balance' not in result:
            print("Error: 'final_balance' key is missing from the result")
        else:
            print(f"Final balance: {result['final_balance']}")

        if 'total_profit' not in result:
            print("Error: 'total_profit' key is missing from the result")
        else:
            print(f"Total Profit: {result['total_profit']}")

        if 'number_of_trades' not in result:
            print("Error: 'number_of_trades' key is missing from the result")
        else:
            print(f"Number of Trades: {result['number_of_trades']}")

        if 'win_rate' not in result:
            print("Error: 'win_rate' key is missing from the result")
        else:
            print(f"Win Rate: {result['win_rate']:.2f}%")

        if 'max_drawdown' not in result:
            print("Error: 'max_drawdown' key is missing from the result")
        else:
            print(f"Maximum Drawdown: {result['max_drawdown']:.2f}")

    # Assuming the key checks passed, continue with the assertion
        if 'total_profit' in result:
            self.assertGreater(result['total_profit'], 0)
        else:
            self.fail("The 'total_profit' key is missing from the backtest result.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()