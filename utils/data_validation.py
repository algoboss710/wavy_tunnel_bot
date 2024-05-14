from typing import Any
from pydantic import BaseModel, ValidationError
import logging

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
    """
    Validate data against a specified schema.
    """
    try:
        if schema == TradeRequestSchema.schema():
            TradeRequestSchema(**data)
        elif schema == CloseRequestSchema.schema():
            CloseRequestSchema(**data)
        else:
            raise ValueError("Invalid schema provided")
        return True
    except ValidationError as e:
        logging.error(f"Data validation failed: {str(e)}")
        return False

def sanitize_data(data: Any) -> Any:
    """
    Sanitize and clean the input data.
    """
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
        logging.error(f"Data sanitization failed: {str(e)}")
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
        raise ValueError("Invalid order time type. Must be 'ORDER_TIME_GTC'")if trade_request['volume'] <= 0:
        raise ValueError("Trade volume must be positive.")
    
    if trade_request['sl'] >= trade_request['price']:
        raise ValueError("Stop loss must be below the entry price for a long trade.")
    
    if trade_request['tp'] <= trade_request['price']:
        raise ValueError("Take profit must be above the entry price for a long trade.")

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
