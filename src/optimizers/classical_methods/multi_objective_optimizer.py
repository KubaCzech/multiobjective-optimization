import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix

# Suppress cvxopt output for cleaner terminal
solvers.options['show_progress'] = False


class OptimizerType(Enum):
    ECM = "Epsilon Constraint Method"
    WSM = "Weighted Sum Method"


class MultiObjectiveOptimizer:
    # f1 -> risk
    # f2 -> return
    def __init__(self, expected_returns, ratios_matrix, n_points):
        self.q = expected_returns
        self.P = ratios_matrix

        self.n = len(expected_returns)
        self.pareto_front = []
        self.method_name = None

        self.is_normalized = False
        self.pareto_front_normalized = None

        self.n_points = n_points

        self.define_constraints()
        self.calculate_bounds()

    def define_constraints(self):
        # Inequality constraint: x >= 0 (No negative allocations)
        # Formatted for cvxopt as Gx <= h -> -Ix <= 0
        self.G = matrix(-np.eye(self.n))
        self.h = matrix(np.zeros(self.n))

        # Equality constraint: sum(x) = 1 (Total allocation must be 100%)
        # Formatted for cvxopt as Ax = b
        self.A = matrix(np.ones((1, self.n)))
        self.b = matrix(1.0)

    def calculate_bounds(self):
        # 1. Find MIN risk
        sol_min_risk = solvers.qp(matrix(2 * self.P), matrix(np.zeros(self.n)), self.G, self.h, self.A, self.b)
        x_min_risk = np.array(sol_min_risk['x']).flatten()

        # 2. Find MAX return
        sol_max_gain = solvers.qp(matrix(np.zeros((self.n, self.n))), matrix(-self.q), self.G, self.h, self.A, self.b)
        x_max_gain = np.array(sol_max_gain['x']).flatten()

        # Bounds on risk (f1)
        self.f1_min = 0.5 * np.dot(x_min_risk.T, np.dot(self.P, x_min_risk))
        self.f1_max = 0.5 * np.dot(x_max_gain.T, np.dot(self.P, x_max_gain))

        # Bounds on return (f2)
        self.f2_min = np.dot(self.q, x_min_risk)
        self.f2_max = np.dot(self.q, x_max_gain)

    def normalize(self):
        risks_min = min(self.pareto_front, key=lambda x: x[0])[0]
        risks_max = max(self.pareto_front, key=lambda x: x[0])[0]

        returns_min = min(self.pareto_front, key=lambda x: x[1])[1]
        returns_max = max(self.pareto_front, key=lambda x: x[1])[1]

        self.pareto_front_normalized = [
            ((i[0] - risks_min) / (risks_max - risks_min), (i[1] - returns_min) / (returns_max - returns_min), i[2])
            for i in self.pareto_front
        ]

    def optimize(self):
        # placeholder to override for later classes
        pass

    def choose_strategy(self, arg):
        pass

    def plot_pareto_front(self):
        plt.figure(figsize=(8, 5))
        plt.plot(
            [i[1] for i in self.pareto_front],
            [i[0] for i in self.pareto_front],
            'b.-',
            markersize=10,
            label='Pareto Optimal Solutions',
        )
        plt.title(f'Pareto Front: Risk vs. Gain ({self.method_name})')
        plt.ylabel('Risk (Variance)')
        plt.xlabel('Gain (Expected Return)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
