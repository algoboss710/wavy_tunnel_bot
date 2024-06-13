import unittest
from unittest.mock import patch, MagicMock
from utils.data_validation import validate_data, sanitize_data, TradeRequestSchema, validate_trade_request
from pydantic import ValidationError

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.valid_trade_request = {
            "action": "BUY",
            "symbol": "EURUSD",
            "volume": 0.1,
            "price": 1.2345,
            "sl": 1.2300,
            "tp": 1.2400,
            "deviation": 10,
            "magic": 12345,
            "comment": "Test order",
            "type": "ORDER_TYPE_BUY",
            "type_filling": "ORDER_FILLING_FOK",
            "type_time": "ORDER_TIME_GTC"
        }

    @patch('utils.data_validation.handle_error')
    def test_validate_data(self, mock_handle_error):
        self.assertTrue(validate_data(self.valid_trade_request, TradeRequestSchema))
        
    @patch('utils.data_validation.handle_error')
    def test_validate_data_invalid(self, mock_handle_error):
        invalid_data = self.valid_trade_request.copy()
        invalid_data.pop("action")
        self.assertFalse(validate_data(invalid_data, TradeRequestSchema))

    def test_sanitize_data(self):
        data = {
            "action": "  BUY  ",
            "symbol": "EURUSD",
            "volume": 0.1,
            "price": 1.2345,
            "sl": 1.2300,
            "tp": 1.2400
        }
        sanitized_data = sanitize_data(data)
        self.assertEqual(sanitized_data["action"], "BUY")

if __name__ == '__main__':
    unittest.main()
