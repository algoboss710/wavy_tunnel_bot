import unittest
from unittest.mock import patch
from utils.logger import setup_logging

class TestLogger(unittest.TestCase):
    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_logging_basicConfig):
        log_level = 20
        log_file = "test.log"
        setup_logging(log_level, log_file)
        mock_logging_basicConfig.assert_called_once()

if __name__ == '__main__':
    unittest.main()