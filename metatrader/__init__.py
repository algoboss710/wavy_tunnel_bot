from .data_retrieval import get_historical_data
# tests/__init__.py
import unittest

def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.discover('C:\\Users\\16198\\Desktop\\automation\\upwork\\wavy\\wavy_tunnel_bot\\tests', pattern='test_*.py'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())