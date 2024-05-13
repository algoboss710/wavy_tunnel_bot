from typing import NewType

TradeAction = NewType('TradeAction', str)
OrderType = NewType('OrderType', str)
OrderFilling = NewType('OrderFilling', str)
OrderTime = NewType('OrderTime', str)
# Define custom types
Symbol = NewType("Symbol", str)
Timeframe = NewType("Timeframe", str)
LotSize = NewType("LotSize", float)