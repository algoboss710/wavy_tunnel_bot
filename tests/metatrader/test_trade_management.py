import unittest
from unittest.mock import patch
from metatrader.trade_management import place_order, close_position, modify_order

class TestTradeManagement(unittest.TestCase):
    @patch('MetaTrader5.order_send', return_value={'retcode': 10009})  # 10009: SUCCESS
    def test_place_order_success(self, mock_order_send):
        symbol = "EURUSD"
        order_type = "buy"
        volume = 0.1
        price = 1.2345
        sl = 1.2300
        tp = 1.2400
        result = place_order(symbol, order_type, volume, price, sl, tp)
        self.assertEqual(result, "python script order")

    @patch('MetaTrader5.order_send', return_value={'retcode': 10004})  # 10004: ERROR
    def test_place_order_failure(self, mock_order_send):
        symbol = "EURUSD"
        order_type = "buy"
        volume = 0.1
        price = 1.2345
        sl = 1.2300
        tp = 1.2400
        result = place_order(symbol, order_type, volume, price, sl, tp)
        self.assertEqual(result, "Order failed")

    @patch('MetaTrader5.positions_get', return_value=[{"ticket": 12345}])
    @patch('MetaTrader5.Close', return_value={"comment": "Close successful"})
    def test_close_position_success(self, mock_close, mock_positions_get):
        ticket = 12345
        result = close_position(ticket)
        self.assertEqual(result, "Close successful")

    @patch('MetaTrader5.positions_get', return_value=[])
    def test_close_position_failure(self, mock_positions_get):
        ticket = 12345
        result = close_position(ticket)
        self.assertEqual(result, "Position not found")

    @patch('MetaTrader5.order_check', return_value={"type": 0})  # 0: BUY
    @patch('MetaTrader5.order_send', return_value={"comment": "Modify successful"})
    def test_modify_order_success(self, mock_order_send, mock_order_check):
        ticket = 12345
        sl = 1.2300
        tp = 1.2400
        result = modify_order(ticket, sl, tp)
        self.assertEqual(result, "Modify successful")

    @patch('MetaTrader5.order_check', return_value=None)
    def test_modify_order_failure(self, mock_order_check):
        ticket = 12345
        sl = 1.2300
        tp = 1.2400
        result = modify_order(ticket, sl, tp)
        self.assertEqual(result, "Order not found")

if __name__ == '__main__':
    unittest.main()