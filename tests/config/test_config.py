import unittest
from unittest.mock import patch, MagicMock

class TestConfig(unittest.TestCase):

    @patch.dict('os.environ', {}, clear=True)
    def test_mt5_login_not_set(self):
        with self.assertRaises(ValueError) as context:
            from config import Config
            Config.MT5_LOGIN
        self.assertEqual(str(context.exception), "MT5_LOGIN environment variable is not set.")

    @patch.dict('os.environ', {'MT5_LOGIN': 'valid_value', 'MT5_PASSWORD': 'password', 'MT5_SERVER': 'server', 'MT5_PATH': 'path', 'MT5_TIMEFRAME': 'H1', 'SYMBOLS': 'EURUSD,GBPUSD'})
    def test_mt5_login_set(self):
        from config import Config
        self.assertEqual(Config.MT5_LOGIN, 'valid_value')

    @patch.dict('os.environ', {
        'MT5_LOGIN': 'login',
        'MT5_PASSWORD': 'password',
        'MT5_SERVER': 'server',
        'MT5_PATH': 'path',
        'MT5_TIMEFRAME': 'H1',
        'SYMBOLS': 'EURUSD,GBPUSD',
        'MIN_TP_PROFIT': '50.0',
        'MAX_LOSS_PER_DAY': '1000.0',
        'STARTING_EQUITY': '10000.0',
        'LIMIT_NO_OF_TRADES': '5',
        'RISK_PER_TRADE': '0.01',
        'PIP_VALUE': '1'
    })
    def test_valid_config(self):
        from config import Config
        Config.validate()

if __name__ == '__main__':
    unittest.main()

