import numpy as np
from sklearn.linear_model import LinearRegression


def estimate_price_linear_regression(x_data, y_data):
    x_data = x_data.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_data, y_data)
    x_future = np.arange(len(x_data) + 100).reshape(-1, 1)
    y_future = model.predict(x_future)
    return x_future, y_future
