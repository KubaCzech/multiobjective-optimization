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