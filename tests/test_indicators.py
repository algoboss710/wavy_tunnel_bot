import unittest
from indicators import calculate_ema, calculate_rsi

class TestIndicators(unittest.TestCase):
    def test_calculate_ema(self):
        data = [100, 200, 300, 400, 500]
        period = 3
        expected_ema = [100.0, 150.0, 233.33333333333334, 344.44444444444446, 455.55555555555554]
        self.assertEqual(calculate_ema(data, period), expected_ema)

    def test_calculate_rsi(self):
        data = [100, 200, 150, 300, 250]
        period = 3
        expected_rsi = [100.0, 100.0, 50.0, 100.0, 66.66666666666667]
        self.assertEqual(calculate_rsi(data, period), expected_rsi)

if __name__ == '__main__':
    unittest.main()