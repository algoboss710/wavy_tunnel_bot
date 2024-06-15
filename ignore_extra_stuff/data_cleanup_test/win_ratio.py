import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Sample dataset that should trigger both buy and sell conditions
data = {
    'time': ['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00', '2023-01-01 03:00:00'],
    'open': [1.6, 1.2, 1.6, 1.2],
    'high': [1.7, 1.5, 1.7, 1.5],
    'low': [1.5, 1.3, 1.5, 1.3],
    'close': [1.6, 1.4, 1.6, 1.4],
    'tick_volume': [100, 100, 100, 100],
    'spread': [0.1, 0.1, 0.1, 0.1],
    'real_volume': [0, 0, 0, 0],
    'wavy_h': [1.5, 1.5, 1.5, 1.5],
    'wavy_c': [1.4, 1.4, 1.4, 1.4],
    'wavy_l': [1.3, 1.3, 1.3, 1.3],
    'tunnel1': [1.2, 1.6, 1.2, 1.6],
    'tunnel2': [1.1, 1.7, 1.1, 1.7],
}

df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'])

# Function to check buy and sell conditions
def check_conditions(row):
    buy_condition = (
        row['open'] > max(row['wavy_c'], row['wavy_h'], row['wavy_l']) and
        min(row['wavy_c'], row['wavy_h'], row['wavy_l']) > max(row['tunnel1'], row['tunnel2'])
    )
    sell_condition = (
        row['open'] < min(row['wavy_c'], row['wavy_h'], row['wavy_l']) and
        max(row['wavy_c'], row['wavy_h'], row['wavy_l']) < min(row['tunnel1'], row['tunnel2'])
    )
    return buy_condition, sell_condition

# Initialize trade state
in_trade = False
entry_price = 0.0
total_profit = 0.0
max_drawdown = 0.0
win_count = 0
trade_count = 0

# Process each row in the dataframe
for index, row in df.iterrows():
    buy_condition, sell_condition = check_conditions(row)
    logging.info(f"Row {index}: Buy condition: {buy_condition}, Sell condition: {sell_condition}")
    logging.info(f"Indicators: open={row['open']}, wavy_h={row['wavy_h']}, wavy_c={row['wavy_c']}, wavy_l={row['wavy_l']}, tunnel1={row['tunnel1']}, tunnel2={row['tunnel2']}")

    if buy_condition and not in_trade:
        in_trade = True
        entry_price = row['open']
        logging.info(f"Buy condition met at {row['time']} with price {entry_price}")

    if sell_condition and in_trade:
        in_trade = False
        exit_price = row['open']
        profit = exit_price - entry_price
        total_profit += profit
        if profit > 0:
            win_count += 1
        trade_count += 1
        logging.info(f"Sell condition met at {row['time']} with price {exit_price}")
        logging.info(f"Trade closed at {row['time']} with price {exit_price}, profit: {profit}")

if trade_count > 0:
    win_rate = win_count / trade_count
else:
    win_rate = 0.0

# Calculate maximum drawdown (simplified for demonstration)
max_drawdown = min(0, total_profit)

performance_metrics = {
    'Total Profit': total_profit,
    'Win Rate': win_rate,
    'Maximum Drawdown': max_drawdown
}

logging.info(f"Performance metrics: {performance_metrics}")
