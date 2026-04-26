import numpy as np
import matplotlib.pyplot as plt

from .nsga_gradient import GradientNSGAIII
from ...nsga import CrossoverMethod, MutationMethod

class GradientNSGATwoObjectives(GradientNSGAIII):
    def __init__(self,
        prices,
        risk_matrix,
        pop_size,
        p,
        dirichlet_alpha=0.2,
        crossover_prob=0.8,
        mutation_prob=0.2,
        eta_c=5,
        eta_m=15,
        crossover_method=CrossoverMethod.SBX,
        mutation_method=MutationMethod.polynomial,
        refinement_interval = 10, 
        refinement_steps = 10, 
        refinement_prob = 0.1, 
        lr=0.01
    ):
        super().__init__(
            prices=prices,
            risk_matrix=risk_matrix,
            pop_size=pop_size,
            n_objectives=2,
            p=p,
            dirichlet_alpha=dirichlet_alpha,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            eta_c=eta_c,
            eta_m=eta_m,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            directions=[+1, -1],
            refinement_interval=refinement_interval,
            refinement_steps=refinement_steps,
            refinement_prob=refinement_prob,
            lr=lr,
        )

    def get_gradient_directions(self):
        # Ranking zysku tworzy wachlarz kierunków na froncie
        current_returns = np.array([np.dot(w, self.prices) for w in self.population])
        sorted_indices = np.argsort(current_returns)
        
        alphas = np.linspace(0.01, 0.99, self.pop_size)
        directions = [None] * self.pop_size
        
        for i, idx in enumerate(sorted_indices):
            # [Waga Zysku, Waga Ryzyka]
            directions[idx] = [alphas[i], 1.0 - alphas[i]]
        return directions
    
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

    def plot_pareto_front(self, title='Price vs Risk Pareto Front (NSGA-III for Two Objectives)', show=True, col='blue', mark='o', label=None, ax=None):
        scores = np.array(self.scores)
        prices = scores[:, 0]
        risks = scores[:, 1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
    
        ax.scatter(prices, risks, c=col, marker=mark, label=label, zorder=2)

        # plt.figure(figsize=(5, 5))
        # plt.scatter(prices, risks, c=col, marker=mark, label=label)

        p_min, p_max = min(prices), max(prices)
        r_min, r_max = min(risks), max(risks)

        utopia_p = p_max
        utopia_r = r_min

        scale = 2.0

        for pt in self.reference_points:
            vec_p = -pt[0] * (p_max - p_min)
            vec_r = pt[1] * (r_max - r_min)

            ax.plot(
                [utopia_p, utopia_p + vec_p * scale],
                [utopia_r, utopia_r + vec_r * scale],
                color='red',
                linestyle='--',
                alpha=0.2,
                zorder=1,
            )

        ax.set_xlim(p_min, p_max)
        ax.set_ylim(r_min, r_max)
        ax.set_title(title)
        ax.set_ylabel('Risk [min]')
        ax.set_xlabel('Price [max]')
        ax.grid()

        if label:
            ax.legend()

        if show:
            plt.show()