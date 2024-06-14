class TradeAction(str):
    def __new__(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"TradeAction must be a string, got {type(value).__name__}")
        return str.__new__(cls, value)

class OrderType(str):
    def __new__(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"OrderType must be a string, got {type(value).__name__}")
        return str.__new__(cls, value)

class OrderFilling(str):
    def __new__(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"OrderFilling must be a string, got {type(value).__name__}")
        return str.__new__(cls, value)

class OrderTime(str):
    def __new__(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"OrderTime must be a string, got {type(value).__name__}")
        return str.__new__(cls, value)

class Symbol(str):
    def __new__(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"Symbol must be a string, got {type(value).__name__}")
        return str.__new__(cls, value)

class Timeframe(str):
    def __new__(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"Timeframe must be a string, got {type(value).__name__}")
        return str.__new__(cls, value)

class LotSize(float):
    def __new__(cls, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"LotSize must be a float, got {type(value).__name__}")
        return float.__new__(cls, value)
