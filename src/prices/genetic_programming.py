import numpy as np
from pysr import PySRRegressor
from gplearn.genetic import SymbolicRegressor

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
def estimate_price_symbolic_regression(x_data, y_data):
    x_data = x_data.reshape(-1, 1)
    model = PySRRegressor(
        niterations=100,
        populations=30,
        population_size=100,
        ncyclesperiteration=700,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos", "exp", "sin", "log"],
        model_selection="accuracy",
        verbosity=0,
        progress=False,
        # optimizer_nrestarts=5
    )
    model.fit(x_data, y_data)
    x_future = np.arange(len(x_data) + 100).reshape(-1, 1)
    y_future = model.predict(x_future)
    return x_future, y_future