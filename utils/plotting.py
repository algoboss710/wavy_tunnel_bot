import matplotlib.pyplot as plt

def plot_backtest_results(data, trades):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot price data
    ax1.plot(data['time'], data['close'], label='Price')
    ax1.set_ylabel('Price')
    ax1.set_title('Backtest Results')
    ax1.grid(True)

    # Plot trades
    for trade in trades:
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        exit_time = trade.get('exit_time')
        exit_price = trade.get('exit_price')

        if exit_time is None:
            ax1.plot(entry_time, entry_price, 'g^', markersize=8, label='Entry')
        else:
            ax1.plot(entry_time, entry_price, 'g^', markersize=8)
            ax1.plot(exit_time, exit_price, 'rv', markersize=8)
            ax1.plot([entry_time, exit_time], [entry_price, exit_price], 'k--')

    ax1.legend()

    # Plot account balance
    balance = [trade['balance'] for trade in trades]
    ax2.plot(data['time'], balance, label='Account Balance')
    ax2.set_ylabel('Balance')
    ax2.set_title('Account Balance')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()