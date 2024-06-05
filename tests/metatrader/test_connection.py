import unittest
from unittest.mock import patch
from metatrader.connection import initialize_mt5, shutdown_mt5

class TestConnection(unittest.TestCase):
    @patch('MetaTrader5.initialize', return_value=True)
    def test_initialize_mt5_success(self, mock_initialize):
        self.assertTrue(initialize_mt5())

    @patch('MetaTrader5.initialize', return_value=False)
    def test_initialize_mt5_failure(self, mock_initialize):
        self.assertFalse(initialize_mt5())

    @patch('MetaTrader5.shutdown')
    def test_shutdown_mt5(self, mock_shutdown):
        shutdown_mt5()
        mock_shutdown.assert_called_once()

if __name__ == '__main__':
    unittest.main()