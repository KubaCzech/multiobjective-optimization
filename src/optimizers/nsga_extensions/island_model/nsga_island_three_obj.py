from .nsga_island import NSGAIsland
from ...nsga import CrossoverMethod, MutationMethod

class NSGAIslandThreeObjectives(NSGAIsland):
    def __init__(self, expected_returns, risks_matrix, island_p=20, p=19, n_migrants=5,
                 crossover_prob=0.8, mutation_prob=0.3, crossover_method=CrossoverMethod.SBX, mutation_method=MutationMethod.polynomial):
        pass