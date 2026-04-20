import numpy as np
import matplotlib.pyplot as plt

from ..nsga import CrossoverMethod, MutationMethod
from .nsga_ii import NSGAII


class NSGAIITwoObjectives(NSGAII):
    def __init__(
        self,
        prices,
        risk_matrix,
        pop_size,
        dirichlet_alpha=0.2,
        crossover_prob=0.8,
        mutation_prob=0.2,
        eta_c=5,
        eta_m=15,
        crossover_method=CrossoverMethod.SBX,
        mutation_method=MutationMethod.polynomial,
    ):
        super().__init__(
            prices=prices,
            risk_matrix=risk_matrix,
            pop_size=pop_size,
            n_objectives=2,
            dirichlet_alpha=dirichlet_alpha,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            eta_c=eta_c,
            eta_m=eta_m,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            directions=[+1, -1],
        )

    def fitness_function(self, sol):
        price = sol @ self.prices
        risk = sol @ self.risk_matrix @ sol
        return price, risk  # (maximize, minimize)

    def dominates(self, sol1, sol2):
        e1 = self.fitness_function(sol1)
        e2 = self.fitness_function(sol2)
        if e1[0] > e2[0] and e1[1] <= e2[1]:
            return True
        if e1[0] >= e2[0] and e1[1] < e2[1]:
            return True
        return False

    def plot_pareto_front(self, title='Price vs Risk Pareto Front (NSGA-II for Two Objectives)'):
        scores = np.array(self.scores)
        prices = scores[:, 0]
        risks = scores[:, 1]

        plt.figure(figsize=(5, 5))
        plt.scatter(prices, risks, c='blue', marker='o')

        plt.title(title)
        plt.xlabel('Price [max]')
        plt.ylabel('Risk [min]')
        plt.grid()
        plt.show()
