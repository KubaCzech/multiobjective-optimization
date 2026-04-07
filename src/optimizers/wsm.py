import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from .multi_objective_optimizer import MultiObjectiveOptimizer, OptimizerType

class WeightedSumMethodOptimizer(MultiObjectiveOptimizer):
    def __init__(self, expected_returns, ratios_matrix, n_points=21):
        super().__init__(expected_returns, ratios_matrix, n_points)
        self.method_name = OptimizerType.WSM.value
        self.weights = None

    def optimize(self):
        # Apply WSM:
        weights = np.linspace(0.0, 1.0, self.n_points)

        for w in weights:
            # WSM Combining: Minimize w*(Risk) - (1-w)*(Gain)
            risk_scale = (self.f1_max - self.f1_min)
            gain_scale = (self.f2_max - self.f2_min)

            # We multiply by 2 because cvxopt minimizes (1/2)*x^T P x.
            P_opt = matrix(2 * w * self.P / risk_scale)
            q_opt = matrix(-(1.0 - w) * self.q / gain_scale)

            # Solve the Quadratic Program
            sol = solvers.qp(P_opt, q_opt, self.G, self.h, self.A, self.b)
            x = np.array(sol['x']).flatten()

            # Calculate the actual Base Risk and Base Gain for this optimal 'x'
            actual_risk = 0.5 * np.dot(x.T, np.dot(self.P, x))
            actual_gain = np.dot(self.q, x)

            self.pareto_front.append((actual_risk, actual_gain, x))
        self.weights = weights
        return self.pareto_front, weights
    
    def plot_pareto_front(self):
        super().plot_pareto_front()
        plt.show()

    def choose_strategy(self, arg):
        assert self.weights is not None
        assert 0 <= arg <= 1
        index = min(range(len(self.pareto_front)), key=lambda i: abs(self.weights[i] - arg))
        return self.pareto_front[index]