import unittest
from unittest.mock import patch
from utils.error_handling import handle_error, critical_error, warn_error

class TestErrorHandling(unittest.TestCase):
    @patch('logging.error')
    def test_handle_error(self, mock_logging_error):
        error = ValueError("Test error")
        message = "An error occurred"
        handle_error(error, message)
        mock_logging_error.assert_called_with(f"{message}: {str(error)}")

    @patch('logging.critical')
    def test_critical_error(self, mock_logging_critical):
        error = ValueError("Test critical error")
        message = "A critical error occurred"
        with self.assertRaises(SystemExit):
            critical_error(error, message)
        mock_logging_critical.assert_called_with(f"{message}: {str(error)}")

    @patch('logging.warning')
    def test_warn_error(self, mock_logging_warning):
        error = ValueError("Test warning")
        message = "A warning occurred"
        warn_error(error, message)
        mock_logging_warning.assert_called_with(f"{message}: {str(error)}")

if __name__ == '__main__':
    unittest.main()