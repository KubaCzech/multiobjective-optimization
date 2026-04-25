import numpy as np
from pysr import PySRRegressor
from gplearn.genetic import SymbolicRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C


# gplearn - mniejsze wymagania systemowe, ale gorsze wyniki i mniejsza wydajność
def estimate_price_gplearn(x_data, y_data):
    x_data = x_data.reshape(-1, 1)
    model = SymbolicRegressor(
        population_size=1000,
        generations=20,
        random_state=42,
        verbose=0,
        parsimony_coefficient=0.001,
        function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos'],
        tournament_size=20,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
    )
    model.fit(x_data, y_data)
    x_future = np.arange(len(x_data) + 100).reshape(-1, 1)
    y_future = model.predict(x_future)
    return x_future, y_future


# PySR - rodzi problemy bo wymaga Julii
def estimate_price_symbolic_regression(x_data, y_data, depth=0):
    x_data = x_data.reshape(-1, 1)
    model = PySRRegressor(
        niterations=1000,
        populations=30,
        population_size=50,
        ncycles_per_iteration=200,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos", "exp", "sin", "log"],
        model_selection="best",
        verbosity=0,
        progress=False,
        # optimizer_nrestarts=5
    )
    model.fit(x_data, y_data)
    x_future = np.arange(len(x_data) + 100).reshape(-1, 1)
    y_future = model.predict(x_future)
    if any(y_future < 0) and depth < 3:
        print(f'Negative predictions at depth {depth}, retrying with increased depth...')
        return estimate_price_symbolic_regression(x_data, y_data, depth + 1)
    elif any(y_future < 0):
        return estimate_price_gaussian_process(x_data, y_data)
    return x_future, y_future

def estimate_price_gaussian_process(x_data, y_data):
    x_data = x_data.reshape(-1, 1)
    y_log = np.log(np.clip(y_data, 1e-5, None))
    
    # DEFINICJA JĄDRA (Kernel):
    # RBF - odpowiada za gładki trend (logarytmy, wielomiany)
    # ExpSineSquared - odpowiada za okresowość (sinusy, cosinusy)
    # WhiteKernel - pochłania szum (raz mniejszy, raz większy)
    kernel = (C(1.0) * RBF(length_scale=10.0) + 
              ExpSineSquared(periodicity=50.0) + 
              WhiteKernel(noise_level=0.1))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    

    gp.fit(x_data, y_log)
    
    x_future = np.arange(len(x_data) + 100).reshape(-1, 1)
    y_future_log, sigma = gp.predict(x_future, return_std=True)
    
    y_future = np.exp(y_future_log)
    
    # lower_bound = np.exp(y_future_log - 1.96 * sigma)
    # upper_bound = np.exp(y_future_log + 1.96 * sigma)
    
    return x_future, y_future
