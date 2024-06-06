import unittest
from utils.data_validation import validate_data, sanitize_data, validate_trade_request, validate_close_request

class TestDataValidation(unittest.TestCase):
    def test_validate_data(self):
        data = {
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
        self.assertTrue(validate_data(data, validate_trade_request))

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