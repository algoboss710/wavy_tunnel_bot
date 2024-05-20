from typing import Any
from pydantic import BaseModel, ValidationError
from .logger import setup_logging
from .error_handling import handle_error

setup_logging()

class TradeRequestSchema(BaseModel):
    action: str
    symbol: str
    volume: float
    price: float
    sl: float
    tp: float
    deviation: int
    magic: int
    comment: str
    type: str
    type_filling: str
    type_time: str

class CloseRequestSchema(BaseModel):
    action: str
    position: int
    type: str
    type_filling: str
    type_time: str

def validate_data(data: Any, schema: dict) -> bool:
    try:
        if schema == TradeRequestSchema.schema():
            TradeRequestSchema(**data)
        elif schema == CloseRequestSchema.schema():
            CloseRequestSchema(**data)
        else:
            raise ValueError("Invalid schema provided")
        return True
    except ValidationError as e:
        handle_error(e, "Data validation failed")
        return False

def sanitize_data(data: Any) -> Any:
    try:
        if isinstance(data, dict):
            sanitized_data = {}
            for key, value in data.items():
                sanitized_data[key.strip()] = sanitize_data(value)
            return sanitized_data
        elif isinstance(data, list):
            return [sanitize_data(item) for item in data]
        elif isinstance(data, str):
            return data.strip()
        else:
            return data
    except Exception as e:
        handle_error(e, "Data sanitization failed")
        return None
    
def validate_trade_request(trade_request):
    required_fields = ['action', 'symbol', 'volume', 'price', 'sl', 'tp', 'deviation', 'magic', 'comment', 'type', 'type_filling', 'type_time']
    for field in required_fields:
        if field not in trade_request:
            raise ValueError(f"Missing required field: {field}")

    if trade_request['action'] not in ['BUY', 'SELL']:
        raise ValueError("Invalid trade action. Must be 'BUY' or 'SELL'")

    if trade_request['type'] not in ['ORDER_TYPE_BUY', 'ORDER_TYPE_SELL']:
        raise ValueError("Invalid order type. Must be 'ORDER_TYPE_BUY' or 'ORDER_TYPE_SELL'")

    if trade_request['type_filling'] != 'ORDER_FILLING_FOK':
        raise ValueError("Invalid order filling type. Must be 'ORDER_FILLING_FOK'")

    if trade_request['type_time'] != 'ORDER_TIME_GTC':
        raise ValueError("Invalid order time type. Must be 'ORDER_TIME_GTC'")

def validate_close_request(close_request):
    required_fields = ['action', 'position', 'type', 'type_filling', 'type_time']
    for field in required_fields:
        if field not in close_request:
            raise ValueError(f"Missing required field: {field}")

    if close_request['action'] != 'CLOSE':
        raise ValueError("Invalid close action. Must be 'CLOSE'")

    if close_request['type'] != 'ORDER_TYPE_CLOSE':
        raise ValueError("Invalid order type. Must be 'ORDER_TYPE_CLOSE'")

    if close_request['type_filling'] != 'ORDER_FILLING_FOK':
        raise ValueError("Invalid order filling type. Must be 'ORDER_FILLING_FOK'")

    if close_request['type_time'] != 'ORDER_TIME_GTC':
        raise ValueError("Invalid order time type. Must be 'ORDER_TIME_GTC'")