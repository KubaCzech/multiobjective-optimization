import numpy as np
from lmfit import Model, Parameters
from scipy.fft import rfft, rfftfreq


def sines_with_trend_model(x, offset, slope, **params):
    # Possible improvement - try A * sin(B * t**2 + C * t + D) instead of A * sin(B * t + C)
    y = (slope * x) + offset  # Added linear trend back in

    # We find how many 'a' parameters exist to know the count of sines
    n = sum(1 for k in params.keys() if k.startswith('a'))

    for i in range(1, n + 1):
        a = params[f'a{i}']
        f = params[f'f{i}']
        p = params[f'p{i}']
        y += a * np.sin(f * x + p)
    return np.where(y > 50, y, np.log(1 + np.exp(y)))


def estimate_price_sum_of_sines(x_data, y_data, n_sines):
    """
    Make price prediction based on sum of sines. To achieve this:
        1. Detrend the data first - fit a simple linear regression to find the trend
        and substract it from the data. This leaves only squiggles (residuals).
        2. Run FFT on detrended data, then ignore the 0th frequency (DC offset that
        should be near 0 anyway).
        3. Create the model
        4. Add N sine parameters dynamically (initial guesses: use FFT for freq and
        std for aplitude).
        5. Fit and predict future data points.
    """
    # 1. Detrend the data
    p = np.polyfit(x_data, y_data, 1)
    slope_guess = p[0]
    intercept_guess = p[1]

    y_detrended = y_data - (slope_guess * x_data + intercept_guess)

    # 2. Run FFT on detrended data
    yf = np.abs(rfft(y_detrended))
    xf = rfftfreq(len(x_data), 1.0)
    omegas = 2 * np.pi * xf

    yf, omegas = yf[1:], omegas[1:]

    topn_idx = np.argsort(yf)[-n_sines:][::-1]
    best_omegas = omegas[topn_idx]

    # 3. Create the Model
    gmodel = Model(sines_with_trend_model)
    params = Parameters()

    params.add('offset', value=intercept_guess)
    params.add('slope', value=slope_guess)

    # 4. Add N sine parameters dynamically
    for i in range(1, n_sines + 1):
        idx = i - 1
        params.add(f'a{i}', value=np.std(y_detrended) / i, min=0)
        params.add(f'f{i}', value=best_omegas[idx], min=1e-5)
        params.add(f'p{i}', value=0.0, min=-np.pi, max=np.pi)

    # 5. Fit and Predict
    result = gmodel.fit(y_data, params, x=x_data)

    target_day = len(x_data) + 100
    x_future = np.arange(target_day)
    y_future = result.eval(x=x_future)

    return x_future, y_future
