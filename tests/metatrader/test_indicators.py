import unittest
from metatrader.indicators import calculate_ema

class TestIndicators(unittest.TestCase):
    def test_calculate_ema(self):
        data = [100, 200, 300, 400, 500]
        period = 3
        expected_ema = [100.0, 150.0, 233.33333333333334, 344.44444444444446, 455.55555555555554]
        self.assertEqual(calculate_ema(data, period), expected_ema)

if __name__ == '__main__':
    unittest.main()
    unittest.main()