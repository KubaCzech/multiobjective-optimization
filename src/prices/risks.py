import numpy as np


def calculate_ratios(prices_matrix):
    change_of_prices = np.diff(prices_matrix, axis=1)
    yesterday_prices = prices_matrix[:, :-1]

    return change_of_prices / yesterday_prices


def estimate_risk(ratios_matrix):
    return np.array(np.cov(ratios_matrix), dtype=float)
