import numpy as np
import matplotlib.pyplot as plt

from .nsga_island import NSGAIsland
from ...nsga import CrossoverMethod, MutationMethod, NSGAIIITwoObjectives

class NSGAIslandTwoObjectives(NSGAIsland):
    def __init__(
            self, 
            prices, 
            risk_matrix, 
            island_pop_size=20, 
            p=19, 
            n_migrants=3, 
            migration_step=100
        ):
        self.islands_names = ['Exploration Island', 'Exploitation Island', 'Balanced Island']
        model1 = NSGAIIITwoObjectives(
            prices=prices,
            risk_matrix=risk_matrix,
            pop_size=island_pop_size,
            p=p,
            crossover_prob=0.8,
            mutation_prob=0.3,
            eta_c=3.0,
            eta_m=5.0,
            crossover_method=CrossoverMethod.SBX,
            mutation_method=MutationMethod.polynomial,
        ) # Exploration Island

        model2 = NSGAIIITwoObjectives(
            prices=prices,
            risk_matrix=risk_matrix,
            pop_size=island_pop_size,
            p=p,
            crossover_prob=1.0,
            mutation_prob=0.05,
            eta_c=30.0,
            eta_m=100.0,
            crossover_method=CrossoverMethod.arithmetic,
            mutation_method=MutationMethod.polynomial,
        ) # Exploitation Island

        model_3 = NSGAIIITwoObjectives(
            prices=prices,
            risk_matrix=risk_matrix,
            pop_size=island_pop_size,
            p=p,
            crossover_prob=0.9,
            mutation_prob=0.1,
            eta_c=20.0,
            eta_m=20.0,
            crossover_method=CrossoverMethod.SBX,
            mutation_method=MutationMethod.polynomial,
        ) # Balanced Island
        super().__init__(model1, model2, model_3, n_migrants=n_migrants, migration_step=migration_step)

    def plot_islands(self):
        styles = list(zip(['blue', 'red', 'green'], ['o', 's', '^']))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, model in enumerate(self.models):
            c, m = styles[idx]
            model.plot_pareto_front(show=False, col=c, mark=m, label=self.islands_names[idx], ax=axes[idx], title=self.islands_names[idx])
        plt.tight_layout()
        plt.show()

    def plot_pareto_front(self, title='Price vs Risk Pareto Front (NSGA-III for Two Objectives)', show=True, col='blue', mark='o', label=None, ax=None):
        scores = np.array(self.sc)
        prices = scores[:, 0]
        risks = scores[:, 1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
    
        ax.scatter(prices, risks, c=col, marker=mark, label=label, zorder=2)

        p_min, p_max = min(prices), max(prices)
        r_min, r_max = min(risks), max(risks)

        utopia_p = p_max
        utopia_r = r_min

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