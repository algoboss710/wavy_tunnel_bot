import unittest
from unittest.mock import patch
from utils.plotting import plot_backtest_results

class TestPlotting(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_plot_backtest_results(self, mock_pyplot_show):
        data = {
            'time': [1, 2, 3],
            'close': [100, 200, 300]
        }
        trades = [
            {'entry_time': 1, 'entry_price': 100, 'exit_time': 2, 'exit_price': 200},
            {'entry_time': 2, 'entry_price': 200, 'exit_time': 3, 'exit_price': 300}
        ]
        plot_backtest_results(data, trades)
        mock_pyplot_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()