import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as positions
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
import requests

# OANDA API settings
API_TOKEN = ""
ACCOUNT_ID = ""
API_URL = "https://api-fxtrade.oanda.com/v3"
client = oandapyV20.API(access_token=API_TOKEN)

# Set up logging
logging.basicConfig(filename='wavy_tunnel_strategy_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def get_historical_data(symbol, granularity, num_candles):
    params = {
        "count": num_candles,
        "granularity": granularity
    }
    url = f"{API_URL}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.get(url, headers=headers, params=params).json()

    candles = response['candles']
    df = pd.DataFrame(candles)

    # Extract the relevant fields
    df['time'] = pd.to_datetime(df['time'])
    df['open'] = df['mid']['o'].astype(float)
    df['high'] = df['mid']['h'].astype(float)
    df['low'] = df['mid']['l'].astype(float)
    df['close'] = df['mid']['c'].astype(float)

    logging.info(f"Retrieved {len(df)} candles for {symbol} on {granularity} timeframe")
    return df

def wavy_tunnel_strategy(symbol, granularity):
    data = get_historical_data(symbol, granularity, 200)

    data['wavy_h'] = calculate_ema(data['high'], 34)
    data['wavy_c'] = calculate_ema(data['close'], 34)
    data['wavy_l'] = calculate_ema(data['low'], 34)
    data['tunnel1'] = calculate_ema(data['close'], 144)
    data['tunnel2'] = calculate_ema(data['close'], 169)

    last_row = data.iloc[-1]
    logging.info(f"EMAs for {symbol} on {granularity}:")
    logging.info(f"  wavy_h: {last_row['wavy_h']:.5f}")
    logging.info(f"  wavy_c: {last_row['wavy_c']:.5f}")
    logging.info(f"  wavy_l: {last_row['wavy_l']:.5f}")
    logging.info(f"  tunnel1: {last_row['tunnel1']:.5f}")
    logging.info(f"  tunnel2: {last_row['tunnel2']:.5f}")

    data['long_condition'] = (data['open'] > data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)) & \
                             (data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1) > data[['tunnel1', 'tunnel2']].max(axis=1))

    data['short_condition'] = (data['open'] < data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)) & \
                              (data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1) < data[['tunnel1', 'tunnel2']].min(axis=1))

    data['exit_long'] = data['close'] < data[['wavy_h', 'wavy_c', 'wavy_l']].min(axis=1)
    data['exit_short'] = data['close'] > data[['wavy_h', 'wavy_c', 'wavy_l']].max(axis=1)

    return data.iloc[-1]

def open_position(symbol, side, units):
    order_type = "MARKET" if side == "buy" else "MARKET"
    data = {
        "order": {
            "units": str(units) if side == "buy" else str(-units),
            "instrument": symbol,
            "timeInForce": "FOK",
            "type": order_type,
            "positionFill": "DEFAULT"
        }
    }

    r = orders.OrderCreate(accountID=ACCOUNT_ID, data=data)
    response = client.request(r)
    logging.info(f"Order placed: {response}")
    return response

def close_position(symbol):
    data = {
        "longUnits": "ALL",
        "shortUnits": "ALL"
    }
    r = positions.PositionClose(accountID=ACCOUNT_ID, instrument=symbol, data=data)
    response = client.request(r)
    logging.info(f"Position closed: {response}")
    return response

def trade(symbol, granularity, units=1000):
    last_row = wavy_tunnel_strategy(symbol, granularity)

    logging.info(f"Evaluating {symbol} on {granularity} timeframe")
    logging.info(f"Current price - Open: {last_row['open']:.5f}, Close: {last_row['close']:.5f}, High: {last_row['high']:.5f}, Low: {last_row['low']:.5f}")
    logging.info(f"Long condition: {last_row['long_condition']}")
    logging.info(f"Short condition: {last_row['short_condition']}")
    logging.info(f"Exit long condition: {last_row['exit_long']}")
    logging.info(f"Exit short condition: {last_row['exit_short']}")

    open_positions = client.request(positions.OpenPositions(accountID=ACCOUNT_ID))

    if open_positions['positions']:
        for position in open_positions['positions']:
            if last_row['exit_long'] or last_row['exit_short']:
                logging.info(f"Closing {symbol} position")
                close_position(symbol)
    else:
        if last_row['long_condition']:
            logging.info(f"Opening long position for {symbol}")
            open_position(symbol, "buy", units)
        elif last_row['short_condition']:
            logging.info(f"Opening short position for {symbol}")
            open_position(symbol, "sell", units)
        else:
            logging.info("No trade signal")

if __name__ == "__main__":
    symbol = "EUR_USD"
    timeframes = ['M1', 'M5', 'M15', 'H1']  # Mapping MetaTrader timeframes to OANDA timeframes

    logging.info("Starting Wavy Tunnel Strategy Test with OANDA")
    logging.info(f"Testing on symbol: {symbol}")
    logging.info(f"Timeframes: {timeframes}")
    logging.info("Press Ctrl+C to stop the script")

    try:
        while True:
            for tf in timeframes:
                trade(symbol, tf)
            logging.info("\nWaiting for next tick...\n")
            time.sleep(1)  # Small delay to prevent excessive CPU usage
    except KeyboardInterrupt:
        logging.info("\nStrategy stopped by user")
