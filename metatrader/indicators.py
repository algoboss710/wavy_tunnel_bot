import numpy as np

def calculate_ema(prices, period):
    return np.convolve(prices, np.exp(-np.linspace(-1., 0., period)), mode='valid') / np.sum(np.exp(-np.linspace(-1., 0., period)))

