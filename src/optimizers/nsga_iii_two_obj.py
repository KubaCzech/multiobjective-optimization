import matplotlib.pyplot as plt
import numpy as np

from .nsga_iii import NSGAIII

class NSGAIIITwoObjectives(NSGAIII):
    def __init__(self, prices, risk_matrix, pop_size, p, dirichlet_alpha=0.2, mutation_prob=0.2, crossover_prob=0.8, elitism=True):
        super().__init__(
            prices, 
            risk_matrix, 
            pop_size, 
            n_objectives=2, 
            p=p, 
            dirichlet_alpha=dirichlet_alpha, 
            mutation_prob=mutation_prob, 
            crossover_prob=crossover_prob, 
            directions=[+1, -1],
            elitism=elitism
        )

    def on_before_normalization(self, temp_scores):
        temp_scores[:, 1] = np.sqrt(temp_scores[:, 1])
        return temp_scores

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

        utopia_p = p_max 
        utopia_r = r_min

        scale = 2.0 

        for pt in self.reference_points:
            vec_p = -pt[0] * (p_max - p_min)
            vec_r =  pt[1] * (r_max - r_min)
            
            plt.plot([utopia_p, utopia_p + vec_p * scale], 
                    [utopia_r, utopia_r + vec_r * scale], 
                    color='red', linestyle='--', alpha=0.2, zorder=1)

        plt.xlim(p_min, p_max)
        plt.ylim(r_min, r_max * 1.1)
        plt.title(title)
        plt.ylabel('Risk [min]')
        plt.xlabel('Price')
        plt.grid()
        plt.show()
    