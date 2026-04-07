from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

from .nsga_iii import NSGAIII

class NSGAIIIThreeObjectives(NSGAIII):
    def __init__(self, prices, risk_matrix, eps, pop_size, p, nr_of_iterations, dirichlet_alpha=0.2, mutation_prob=0.1, crossover_prob=0.8):
        super().__init__(prices, risk_matrix, pop_size, nr_of_iterations, 3, p, dirichlet_alpha, mutation_prob, crossover_prob)
        self.eps = eps

    def dominates(self, sol1, sol2):
        eval1 = self.fitness_function(sol1)
        eval2 = self.fitness_function(sol2)
        if eval1[0] > eval2[0] and eval1[1] <= eval2[1] and eval1[2] >= eval2[2]:
            return True
        if eval1[0] >= eval2[0] and eval1[1] < eval2[1] and eval1[2] >= eval2[2]:
            return True
        if eval1[0] >= eval2[0] and eval1[1] <= eval2[1] and eval1[2] > eval2[2]:
            return True
        return False

    def fitness_function(self, sol):
        risk = sol @ self.risk_matrix @ sol
        price = sol @ self.prices
        diversity_coeff = len(set(np.where(sol > self.eps)[0])) / self.n
        return -price, risk, -diversity_coeff


    def plot_pareto_front(self):
        # TODO: zrobic interaktywny wykres 3D z możliwością obracania, żeby lepiej zobaczyć rozkład punktów
        prices = [score[0] for score in self.scores]
        risks = [score[1] for score in self.scores]
        diversities = [-score[2] for score in self.scores]

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(risks, prices, diversities, c='blue', marker='o')
        ax.set_title('Pareto Front - Price vs Risk vs Diversity')
        ax.set_xlabel('Risk [min]')
        ax.set_ylabel('-Price [min]')
        ax.set_zlabel('Diversity Coefficient')
        plt.show()