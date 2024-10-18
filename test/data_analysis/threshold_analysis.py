import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json
import os
import logging

# Set up logging
logging.basicConfig(filename='threshold_analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MetaTrader5 connection
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    mt5.shutdown()
    quit()

def timeframe_to_string(timeframe):
    timeframe_dict = {
        mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1"
    }
    return timeframe_dict.get(timeframe, str(timeframe))

def get_historical_data(symbol, timeframe, start_date, end_date):
    timezone = pytz.timezone("Etc/UTC")
    start_date = timezone.localize(start_date)
    end_date = timezone.localize(end_date)

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def analyze_volatility(df):
    df['daily_range'] = df['high'] - df['low']
    df['atr'] = calculate_atr(df)

    return {
        'average_daily_range': df['daily_range'].mean(),
        'average_atr': df['atr'].mean(),
        'max_daily_range': df['daily_range'].max()
    }

def analyze_spread(symbol, timeframe, start_date, end_date):
    try:
        ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
        df_ticks = pd.DataFrame(ticks)
        if 'time' not in df_ticks.columns:
            logging.warning(f"'time' column not found in tick data for {symbol}")
            return None
        df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
        df_ticks['spread'] = df_ticks['ask'] - df_ticks['bid']

        avg_spread = df_ticks['spread'].mean()
        max_spread = df_ticks['spread'].max()

        df_ticks['hour'] = df_ticks['time'].dt.hour
        asian_session = df_ticks[(df_ticks['hour'] >= 0) & (df_ticks['hour'] < 8)]
        european_session = df_ticks[(df_ticks['hour'] >= 8) & (df_ticks['hour'] < 16)]
        us_session = df_ticks[(df_ticks['hour'] >= 16) & (df_ticks['hour'] < 24)]

        return {
            'average_spread': avg_spread,
            'max_spread': max_spread,
            'asian_session_avg_spread': asian_session['spread'].mean(),
            'european_session_avg_spread': european_session['spread'].mean(),
            'us_session_avg_spread': us_session['spread'].mean()
        }
    except Exception as e:
        logging.error(f"Error analyzing spread for {symbol}: {str(e)}")
        return None

def analyze_pair(symbol, timeframe, start_date, end_date):
    logging.info(f"Analyzing {symbol} on {timeframe_to_string(timeframe)} timeframe...")

    df = get_historical_data(symbol, timeframe, start_date, end_date)
    volatility_metrics = analyze_volatility(df)
    spread_metrics = analyze_spread(symbol, timeframe, start_date, end_date)

    df['price_change'] = df['close'] - df['open']
    significant_events = df[abs(df['price_change']) > 3 * volatility_metrics['average_atr']]

    return {
        'volatility_metrics': volatility_metrics,
        'spread_metrics': spread_metrics,
        'significant_events': significant_events.to_dict(orient='records')
    }

def suggest_threshold(volatility_metrics, spread_metrics):
    if spread_metrics is None:
        return volatility_metrics['average_atr'] * 0.5  # 50% of ATR
    atr_based = volatility_metrics['average_atr'] * 0.5  # 50% of ATR
    spread_based = spread_metrics['average_spread'] * 3  # 3 times average spread
    return max(atr_based, spread_based)

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)

# Main execution
if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 10, 11)

    results = {}
    threshold_suggestions = {}

    for symbol in symbols:
        results[symbol] = {}
        threshold_suggestions[symbol] = {}
        for timeframe in timeframes:
            tf_name = timeframe_to_string(timeframe)
            try:
                results[symbol][tf_name] = analyze_pair(symbol, timeframe, start_date, end_date)

                # Suggest threshold
                threshold = suggest_threshold(
                    results[symbol][tf_name]['volatility_metrics'],
                    results[symbol][tf_name]['spread_metrics']
                )
                threshold_suggestions[symbol][tf_name] = threshold
                logging.info(f"Suggested threshold for {symbol} on {tf_name}: {threshold:.5f}")
            except Exception as e:
                logging.error(f"Error analyzing {symbol} on {tf_name}: {str(e)}")

    # Save results
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')

    save_results(results, 'analysis_results/historical_analysis.json')
    save_results(threshold_suggestions, 'analysis_results/threshold_suggestions.json')

    # Print threshold suggestions
    print("\nSuggested Initial Thresholds:")
    for symbol, timeframes in threshold_suggestions.items():
        print(f"\n{symbol}:")
        for tf, threshold in timeframes.items():
            print(f"  {tf}: {threshold:.5f}")

    mt5.shutdown()