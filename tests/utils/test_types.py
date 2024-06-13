import unittest
from utils.types import TradeAction, OrderType, OrderFilling, OrderTime, Symbol, Timeframe, LotSize

class TestTypes(unittest.TestCase):
    
    def test_trade_action(self):
        action = TradeAction("BUY")
        self.assertIsInstance(action, str)
        self.assertEqual(action, "BUY")
        
        with self.assertRaises(TypeError):
            TradeAction(123)  # Invalid type

    def test_order_type(self):
        order_type = OrderType("LIMIT")
        self.assertIsInstance(order_type, str)
        self.assertEqual(order_type, "LIMIT")
        
        with self.assertRaises(TypeError):
            OrderType(123)  # Invalid type

    def test_order_filling(self):
        filling = OrderFilling("FOK")
        self.assertIsInstance(filling, str)
        self.assertEqual(filling, "FOK")
        
        with self.assertRaises(TypeError):
            OrderFilling(123)  # Invalid type

    def test_order_time(self):
        order_time = OrderTime("GTC")
        self.assertIsInstance(order_time, str)
        self.assertEqual(order_time, "GTC")
        
        with self.assertRaises(TypeError):
            OrderTime(123)  # Invalid type

    def test_symbol(self):
        symbol = Symbol("EURUSD")
        self.assertIsInstance(symbol, str)
        self.assertEqual(symbol, "EURUSD")
        
        with self.assertRaises(TypeError):
            Symbol(123)  # Invalid type

    def test_timeframe(self):
        timeframe = Timeframe("M1")
        self.assertIsInstance(timeframe, str)
        self.assertEqual(timeframe, "M1")
        
        with self.assertRaises(TypeError):
            Timeframe(123)  # Invalid type

    def test_lot_size(self):
        lot_size = LotSize(1.0)
        self.assertIsInstance(lot_size, float)
        self.assertEqual(lot_size, 1.0)
        
        with self.assertRaises(TypeError):
            LotSize("1.0")  # Invalid type

if __name__ == '__main__':
    unittest.main()
