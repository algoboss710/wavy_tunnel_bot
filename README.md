# Wavy Tunnel Trading Bot

The Wavy Tunnel Trading Bot is an automated trading bot designed to execute trading strategies using the MetaTrader 5 platform. This bot provides both backtesting and live trading capabilities with a graphical user interface (GUI) for ease of use.

## Features

- Backtesting: Simulate trading strategies on historical data to evaluate performance.
- Live Trading: Execute trading strategies in real-time on the MetaTrader 5 platform.
- Configuration: Easily configure trading parameters through the GUI.
- Real-time Logs: View real-time logs within the GUI.

## Installation

### Prerequisites

- Python 3.8 or later
- MetaTrader 5 installed on your system

### Installing Dependencies

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/wavy_tunnel_bot.git
    cd wavy_tunnel_bot
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Configuration

The configuration settings for the trading bot are loaded from environment variables. You can create a `.env` file in the project root directory to specify these variables. The available configuration options are:

- `MT5_LOGIN`: MetaTrader 5 login ID
- `MT5_PASSWORD`: MetaTrader 5 password
- `MT5_SERVER`: MetaTrader 5 server
- `MT5_PATH`: Path to MetaTrader 5 terminal executable
- `MT5_TIMEFRAME`: Timeframe for trading (e.g., `H1`)
- `SYMBOLS`: List of trading symbols (e.g., `['EURUSD']`)
- `TELEGRAM_TOKEN`: Telegram bot token for notifications
- `TELEGRAM_IDS`: List of Telegram user IDs for notifications
- `MIN_TP_PROFIT`: Minimum take profit
- `MAX_LOSS_PER_DAY`: Maximum loss per day
- `STARTING_EQUITY`: Starting equity for trading
- `LIMIT_NO_OF_TRADES`: Maximum number of trades per day
- `RISK_PER_TRADE`: Risk per trade
- `PIP_VALUE`: Value per pip

Create a `.env` file in the project root directory with the following content:

MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_TIMEFRAME=H1
SYMBOLS=['EURUSD']
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_IDS=['telegram_user_id_1', 'telegram_user_id_2']
MIN_TP_PROFIT=50.0
MAX_LOSS_PER_DAY=1000.0
STARTING_EQUITY=10000.0
LIMIT_NO_OF_TRADES=5
RISK_PER_TRADE=0.02
PIP_VALUE=0.0001


## Usage

### Running the Trading Bot

1. **With GUI**:
    To start the bot with the graphical user interface, run:
    ```sh
    python main.py --ui
    ```

2. **Without GUI**:
    To start the bot without the GUI and choose between backtesting and live trading via command-line prompts, run:
    ```sh
    python main.py
    ```

### GUI Options

1. **Run Backtesting**:
    - This option allows you to simulate trading strategies on historical data. The logs will be displayed in real-time within the GUI.

2. **Run Live Trading**:
    - This option allows you to execute trading strategies in real-time on the MetaTrader 5 platform. The logs will be displayed in real-time within the GUI.

3. **Configuration Settings**:
    - This option opens a window where you can configure various trading parameters such as MT5 login credentials, trading symbols, risk management parameters, etc.

4. **Clear Log File**:
    - This option clears the `app.log` file.

5. **Open Log File**:
    - This option opens the `app.log` file to view the log entries.

### Backtesting

Backtesting is a crucial feature of the Wavy Tunnel Trading Bot. It allows you to simulate trading strategies on historical data to evaluate their performance before applying them to live trading. Here’s an end-to-end breakdown of what happens when the backtest is run:

1. **Initialization**:
    - The backtesting function initializes the MetaTrader 5 (MT5) platform using the credentials and path provided in the configuration.

2. **Data Retrieval**:
    - Historical data for the specified symbols and timeframe is retrieved from the MT5 platform. This data includes open, high, low, close prices, and volumes.
    - **Dates for Historical Data**:
        - The start and end dates for the historical data are currently set in the `main.py` file within the `run_backtest_func` function:
        ```python
        start_date = datetime(2024, 6, 12)
        end_date = datetime.now()
        ```
        - `start_date` is set to June 12, 2024.
        - `end_date` is set to the current date and time when the backtest is run.

3. **Strategy Execution**:
    - **Calculate Indicators**:
        - The strategy calculates the Exponential Moving Average (EMA) for different periods (e.g., 50-period EMA, 200-period EMA) using the historical price data.
    - **Detect Peaks and Dips**:
        - The strategy identifies peaks and dips in the price data based on certain conditions, such as price crossing above or below the EMA.
    - **Check Entry Conditions**:
        - The strategy checks for entry conditions. For example, a typical entry condition might be:
            - A buy signal is generated when the short-term EMA crosses above the long-term EMA, indicating a potential upward trend.
            - A sell signal is generated when the short-term EMA crosses below the long-term EMA, indicating a potential downward trend.
    - **Execute Trades**:
        - When the entry conditions are met, a virtual trade is opened with the specified parameters such as stop loss and take profit.
        - The strategy manages the trade, monitoring the price movements and checking for exit conditions.
    - **Check Exit Conditions**:
        - The strategy checks for exit conditions, such as reaching the take profit level, hitting the stop loss, or reversing signals (e.g., EMA crossover in the opposite direction).

4. **Performance Evaluation**:
    - Throughout the backtesting period, the strategy logs each trade, including entry and exit points, profit and loss, and other relevant metrics.
    - After the backtesting period, the overall performance is evaluated based on metrics such as total profit, number of trades, win rate, maximum drawdown, and risk-reward ratio.

5. **Logging and Results**:
    - All activities and results during the backtest are logged in the `app.log` file.
    - Upon completion, you can view the logs and evaluate the performance of the trading strategy.

## Testing

### Running Unit and Integration Tests

To ensure the functionality and reliability of the Wavy Tunnel Trading Bot, you can run the unit and integration tests included in the `tests` module.

1. **Install Testing Dependencies**:
    Make sure you have installed all the required packages listed in `requirements.txt`.

2. **Run Tests**:
    Use the following command to run all tests:
    ```sh
    pytest
    ```

    You can also run a specific test module or function:
    ```sh
    pytest tests/test_module.py
    pytest tests/test_module.py::test_function
    ```

### Example of Running All Tests

1. Open a terminal and navigate to the project directory.
2. Ensure your virtual environment is activated.
3. Run the tests using `pytest`:
    ```sh
    pytest
    ```

This will execute all the unit and integration tests and provide a summary of the results.

