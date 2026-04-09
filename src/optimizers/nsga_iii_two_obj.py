from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

from .nsga_iii import NSGAIII

class NSGAIIITwoObjectives(NSGAIII):
    def __init__(self, prices, risk_matrix, pop_size, p, nr_of_iterations, dirichlet_alpha=0.2, mutation_prob=0.3, crossover_prob=0.8):
        super().__init__(prices, risk_matrix, pop_size, nr_of_iterations, 2, p, dirichlet_alpha, mutation_prob, crossover_prob)

    def generate_reference_points(self):
        ref_points = []
        # Das-Dennis method (Stars and Bars)
        for combo in combinations(range(self.p + self.n_objectives - 1), self.n_objectives - 1):
            combo = [-1] + list(combo) + [self.p + self.n_objectives - 1]
            
            point = []
            for i in range(len(combo) - 1):
                val = (combo[i+1] - combo[i] - 1) / self.p
                point.append(val)
            
            ref_points.append(point)
            
        return np.array(ref_points)

    def dominates(self, sol1, sol2):
        eval1 = self.fitness_function(sol1)
        eval2 = self.fitness_function(sol2)
        if eval1[0] > eval2[0] and eval1[1] <= eval2[1]:
            return True
        if eval1[0] >= eval2[0] and eval1[1] < eval2[1]:
            return True
        return False

    def fitness_function(self, sol):
        risk = sol @ self.risk_matrix @ sol
        price = sol @ self.prices
        return price, risk

    def plot_pareto_front(self, title='Pareto Front - Price vs Risk'):
        prices = [score[0] for score in self.scores]
        risks = [score[1] for score in self.scores]

        plt.figure(figsize=(5, 5))
        plt.scatter(prices, risks, c='blue', marker='o')

        p_min, p_max = min(prices), max(prices)
        r_min, r_max = min(risks), max(risks)

        scale = 2.0 

        # (Utopia Point) -> p_max, r_min
        start_p = p_max
        start_r = r_min

        for pt in self.reference_points:
            target_p = p_min + pt[0] * (p_max - p_min)
            target_r = r_min + pt[1] * (r_max - r_min)
            
            vec_p = target_p - start_p
            vec_r = target_r - start_r
            
            plt.plot([start_p, start_p + vec_p * scale], 
                    [start_r, start_r + vec_r * scale], 
                    color='red', linestyle='--', alpha=0.2, zorder=1)

        plt.xlim(p_min * 0.9, p_max * 1.1)
        plt.ylim(r_min * 0.9, r_max * 1.1)
        plt.title(title)
        plt.ylabel('Risk [min]')
        plt.xlabel('Price')
        plt.grid()
        plt.show()
    