import unittest
from unittest.mock import patch, Mock
from strategy.trade_logic import calculate_position_size, entry_long, entry_short, exit_trade
from utils.error_handling import handle_error

class TestStrategyFunctions(unittest.TestCase):

    def test_calculate_position_size_valid_inputs(self):
        print("Running test_calculate_position_size_valid_inputs")
        balance = 10000
        risk_percent = 0.01
        stop_loss_pips = 50
        pip_value = 10
        expected_position_size = 0.2
        result = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
        self.assertEqual(result, expected_position_size)

    def test_calculate_position_size_zero_stop_loss(self):
        print("Running test_calculate_position_size_zero_stop_loss")
        balance = 10000
        risk_percent = 0.01
        stop_loss_pips = 0
        pip_value = 10
        expected_position_size = 0
        result = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
        self.assertEqual(result, expected_position_size)

    def test_calculate_position_size_zero_pip_value(self):
        print("Running test_calculate_position_size_zero_pip_value")
        balance = 10000
        risk_percent = 0.01
        stop_loss_pips = 50
        pip_value = 0
        expected_position_size = 0
        result = calculate_position_size(balance, risk_percent, stop_loss_pips, pip_value)
        self.assertEqual(result, expected_position_size)

    @patch('strategy.trade_logic.execute_trade')
    def test_entry_long_success(self, mock_execute_trade):
        print("Running test_entry_long_success")
        mock_execute_trade.return_value = 'Order placed successfully'
        symbol = 'EURUSD'
        lot_size = 1.0
        price = 1.2345
        sl = 1.2300
        tp = 1.2400
        deviation = 10
        magic = 12345
        comment = 'Test Long Entry'
        result = entry_long(symbol, lot_size, price, sl, tp, deviation, magic, comment)
        self.assertEqual(result, 'Order placed successfully')
        mock_execute_trade.assert_called_once_with({
            'action': 'BUY',
            'symbol': symbol,
            'volume': lot_size,
            'price': price,
            'sl': sl,
            'tp': tp,
            'deviation': deviation,
            'magic': magic,
            'comment': comment,
            'type': 'ORDER_TYPE_BUY',
            'type_filling': 'ORDER_FILLING_FOK',
            'type_time': 'ORDER_TIME_GTC'
        })

    @patch('strategy.trade_logic.execute_trade')
    def test_entry_short_success(self, mock_execute_trade):
        print("Running test_entry_short_success")
        mock_execute_trade.return_value = 'Order placed successfully'
        symbol = 'EURUSD'
