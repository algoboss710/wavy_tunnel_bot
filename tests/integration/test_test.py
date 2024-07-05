import unittest
import pandas as pd
from datetime import datetime, timedelta
from backtesting import backtest  # Ensure this matches your import path

class TestRunBacktestOutputs(unittest.TestCase):

    def setUp(self):
        # Create a simple DataFrame to use as input
        start_date = datetime(2024, 1, 1)
        self.data = pd.DataFrame({
            'time': [start_date + timedelta(days=i) for i in range(50)],
            'open': [1.1 + 0.01*i for i in range(50)],
            'high': [1.2 + 0.01*i for i in range(50)],
            'low': [1.0 + 0.01*i for i in range(50)],
            'close': [1.15 + 0.01*i for i in range(50)]
        })

    def test_run_backtest_outputs(self):
        result = backtest.run_backtest(
            symbol='TEST',
            data=self.data,
            initial_balance=10000,
            risk_percent=0.01,
            min_take_profit=50,
            max_loss_per_day=100,
            starting_equity=10000,
            stop_loss_pips=20,
            pip_value=0.0001,
            max_trades_per_day=5,
            slippage=0,
            transaction_cost=0
        )

        self.assertIn('final_balance', result)
        self.assertIn('total_profit', result)
        self.assertIn('num_trades', result)
        self.assertIn('win_rate', result)
        self.assertIn('max_drawdown', result)
        self.assertIn('buy_condition', result)
        self.assertIn('sell_condition', result)
        self.assertIn('trades', result)
        self.assertIn('total_slippage_costs', result)
        self.assertIn('total_transaction_costs', result)

        print(f"Result: {result}")

if __name__ == '__main__':
    unittest.main()
