import matplotlib.pyplot as plt

def plot_prices(name, prices):
    plt.figure(figsize=(10, 5))
    plt.plot(prices, marker='o')
    plt.title(f'Price Evolution for {name}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid()
    plt.show()

def plot_predictions(x_data, y_data, x_future, y_future, name=None):
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data, 'k*', label='Known Prices')

    plt.plot(x_future, y_future, 'b-', linewidth=2, label='FFT + Trend Prediction')
    plt.axvline(x=len(x_data)-1, color='cyan', linestyle='-', label='Start of Forecast')
    plt.axvline(x=len(x_future)-1, color='cyan', linestyle='--', label='End of Forecast')

    color = 'green' if y_data[-1] < y_future[-1] else 'red'
    plt.plot([len(x_data)-1, len(x_future)-1], [y_data[-1], y_future[-1]], color=color, linestyle='--', 
             linewidth=1.5, label='Trend Line')
    
    plt.legend()
    if name:
        plt.title(f"LMFIT Extrapolation with Detrending & FFT Guesses for {name}")
    else:
        plt.title("LMFIT Extrapolation with Detrending & FFT Guesses")
    plt.show()

def plot_evaluation_of_predictions(x_data, y_data, x_future, y_future, name=None):
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data, 'k*', label='Known Prices')

    plt.plot(x_future, y_future, 'b-', linewidth=2, label='FFT + Trend Prediction')
    plt.axvline(x=len(x_data)-101, color='cyan', linestyle='-', label='Start of Forecast')
    plt.axvline(x=len(x_future)-1, color='cyan', linestyle='--', label='End of Forecast')

    color = 'green' if y_data[-101] < y_future[-1] else 'red'
    plt.plot([len(x_data)-101, len(x_future)-1], [y_data[-101], y_future[-1]], color=color, linestyle='--', 
             linewidth=1.5, label='Trend Line')
    
    plt.legend()
    if name:
        plt.title(f"LMFIT Extrapolation with Detrending & FFT Guesses for {name}")
    else:
        plt.title("LMFIT Extrapolation with Detrending & FFT Guesses")
    plt.show()