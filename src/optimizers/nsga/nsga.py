import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class CrossoverMethod(Enum):
    SBX = 'sbx'
    one_point = 'one_point'
    arithmetic = 'arithmetic'

class MutationMethod(Enum):
    SBX = 'sbx'
    polynomial = 'polynomial'
    swap = 'swap'
    flow = 'flow'

class UtopianPointArchive:
    # Used for normalization
    def __init__(self, directions):
        self.directions = directions
        self.fitness_values = np.array([-np.inf if dir==1 else np.inf for dir in directions])

    @property
    def utopian_point(self):
        return self.fitness_values

    def add_solution(self, fitness_vals):
        fitness_vals = np.array(fitness_vals)
        self.fitness_values[self.directions == 1] = np.maximum(self.fitness_values[self.directions == 1], fitness_vals[self.directions == 1])
        self.fitness_values[self.directions == -1] = np.minimum(self.fitness_values[self.directions == -1], fitness_vals[self.directions == -1])

class EliteSolutionsArchive:
    # Used to preserve best solutions
    def __init__(self, directions):
        self.directions = np.array(directions)
        self.extreme_solutions = [None] * len(directions)
        self.fitness_values = np.array([[-np.inf if dir==1 else np.inf for dir in directions] for _ in range(len(directions))])

    @property
    def elite_solutions(self):
        return [sol for sol in self.extreme_solutions if sol is not None]
    
    def dominates(self, fit1, fit2):
        diff = (fit1 - fit2) * self.directions
        return np.all(diff >= 0) and np.any(diff > 0)
            
    def add_solution(self, sol: list, fitness_vals: list):
        fit1 = np.array(fitness_vals)
        for idx, dir in enumerate(self.directions):
            fit2 = self.fitness_values[idx]
            domination_flag = self.dominates(fit1, fit2) or (not self.dominates(fit1, fit2) and not self.dominates(fit2, fit1)) # either dominates or non-comparability
            if dir == 1:
                if self.fitness_values[idx][idx] < fitness_vals[idx] and domination_flag:
                    self.extreme_solutions[idx] = sol.copy()
                    self.fitness_values[idx] = fit1
            elif dir == -1:
                if self.fitness_values[idx][idx] > fitness_vals[idx] and domination_flag:
                    self.extreme_solutions[idx] = sol.copy()
                    self.fitness_values[idx] = fit1
            else:
                raise ValueError