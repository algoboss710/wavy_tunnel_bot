import unittest
from unittest.mock import patch
from config import Config

class TestConfig(unittest.TestCase):
    @patch('os.getenv')
    def test_mt5_login_not_set(self, mock_getenv):
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            Config.MT5_LOGIN

    @patch('os.getenv', return_value='valid_value')
    def test_mt5_login_set(self, mock_getenv):
        self.assertEqual(Config.MT5_LOGIN, 'valid_value')

    @patch('os.getenv', side_effect=lambda k, v=None: {
        'MT5_TIMEFRAME': 'H1',
        'SYMBOLS': 'EURUSD,GBPUSD',
        'MIN_TP_PROFIT': '50.0',
        'MAX_LOSS_PER_DAY': '1000.0',
        'STARTING_EQUITY': '10000.0',
        'LIMIT_NO_OF_TRADES': '5',
        'RISK_PER_TRADE': '0.01',
        'PIP_VALUE': '1'
    }.get(k, v))
    def test_valid_config(self, mock_getenv):
        Config.validate()
