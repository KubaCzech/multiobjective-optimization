import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from .multi_objective_optimizer import MultiObjectiveOptimizer, OptimizerType

class EpsilonConstraintMethodOptimizer(MultiObjectiveOptimizer):
    def __init__(self, expected_returns, ratios_matrix, n_points=21):
        super().__init__(expected_returns, ratios_matrix, n_points)
        self.method_name = OptimizerType.ECM.value
        self.epsilons = None

    def optimize(self):
        # Epsilon means: "I want minimal risk with gain at least X"
        epsilons = np.linspace(self.f2_min, self.f2_max, self.n_points)

        for eps in epsilons:            
            # self.G and self.h are the constraints x >= 0
            G_extra = matrix(-self.q.reshape(1, self.n))
            h_extra = matrix([-float(eps)])
            
            G_total = matrix(np.vstack([self.G, G_extra]))
            h_total = matrix(np.vstack([self.h, h_extra]))

            # 3. Minimize risk (P_norm) for a given gain
            try:
                # Goal: min (1/2) * x^T * (2 * P_norm) * x
                sol = solvers.qp(matrix(2 * self.P), 
                                 matrix(np.zeros(self.n)), 
                                 G_total, h_total, self.A, self.b)
                
                if sol['status'] == 'optimal':
                    x = np.array(sol['x']).flatten()
                    
                    # Calculate raw (not normalized) gains and risks
                    actual_risk = 0.5 * np.dot(x.T, np.dot(self.P, x))
                    actual_gain = np.dot(self.q, x)
                    
                    self.pareto_front.append((actual_risk, actual_gain, x))
            except: # infeasible solution
                continue

        self.epsilons = epsilons
        # Normalize the front so the results are in [0, 1]
        return self.pareto_front, epsilons
    
    def plot_pareto_front(self):
        super().plot_pareto_front()
        if self.method_name == OptimizerType.ECM.value:
            for eps in self.epsilons:
                plt.axvline(x=(eps - self.f2_min)/(self.f2_max - self.f2_min), color='r', linestyle='--')
        plt.show()

    def choose_strategy(self, arg):
        assert self.epsilons is not None
        assert 0 <= arg <= 1
        index = min(
            range(len(self.pareto_front)), 
            key=lambda i: abs((self.epsilons[i]-self.f2_min)/(self.f2_max - self.f2_min) - arg)
        )
        return self.pareto_front[index]