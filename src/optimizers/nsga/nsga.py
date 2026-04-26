import random
import numpy as np
from enum import Enum


class CrossoverMethod(Enum):
    SBX = 'sbx'
    one_point = 'one_point'
    arithmetic = 'arithmetic'


class MutationMethod(Enum):
    distribute = 'distribute'
    polynomial = 'polynomial'
    swap = 'swap'
    flow = 'flow'


class UtopianPointArchive:
    # Used for normalization
    def __init__(self, directions):
        self.directions = directions
        self.fitness_values = np.array([-np.inf if dir == 1 else np.inf for dir in directions])

    @property
    def utopian_point(self):
        return self.fitness_values

    def add_solution(self, fitness_vals):
        fitness_vals = np.array(fitness_vals)

        self.fitness_values[self.directions == 1] = np.maximum(
            self.fitness_values[self.directions == 1], fitness_vals[self.directions == 1]
        )
        self.fitness_values[self.directions == -1] = np.minimum(
            self.fitness_values[self.directions == -1], fitness_vals[self.directions == -1]
        )


def project_onto_simplex(v, s=1.0):
    """
    Michelot method
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def check_constraints(sol):
    eps = 1e-10
    if np.any(sol < 0):
        return False
    if np.sum(sol) > 1.000+eps or np.sum(sol) < 1.000-eps:
        return False
    return True

class NSGA:
    def __init__(
        self,
        prices,
        risk_matrix,
        pop_size,
        n_objectives,
        dirichlet_alpha,
        crossover_prob,
        mutation_prob,
        eta_c,
        eta_m,
        crossover_method,
        mutation_method,
        directions,
    ):
        self.prices = prices
        self.risk_matrix = risk_matrix
        self.scores = []
        self.n_objectives = n_objectives
        # directions: +1 = maximize, -1 = minimize (one per objective)
        self.directions = np.array(directions) if directions is not None else np.full(n_objectives, -1)

        self.population = []
        self.pop_size = pop_size
        self.n = len(prices)
        self.dirichlet_alpha = dirichlet_alpha

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        self.eta_m = eta_m
        self.eta_c = eta_c

        self.history = []  # for each generation: (return, risk, optionally: div_coeff, solution)

        self.create_initial_population()

    # ---
    # Creating and evaluating population
    # ---
    def get_population(self):
        return self.population

    def evaluate_population(self):
        self.scores = []
        for sol in self.population:
            score = self.fitness_function(sol)
            self.scores.append(score)
        self.scores = np.array(self.scores)

    def create_random_kraemer_solution(self):
        cuts = np.random.rand(self.n - 1)
        cuts.sort()
        points = np.concatenate(([0.0], cuts, [1.0]))
        weights = np.diff(points)  # Weights are the intervals between the sorted random cuts, ensuring they sum to 1

        return weights

    def create_random_dirichlet_solution(self):
        weights = np.random.dirichlet([self.dirichlet_alpha] * self.n)
        return weights

    def create_initial_population(self):
        # To avoid situations where all weights are close to 1/n which is due to the nature of random numbers generator
        # We can generate 75% of the solutions using Kraemer method and 25% using Dirichlet distribution to ensure more
        # diversity in the population and let algorithm find solutions dominated by one or few stocks
        assert not self.population

        kraemer_solution_number = int(self.pop_size * 0.75)
        dirichlet_solution_number = self.pop_size - kraemer_solution_number

        for _ in range(kraemer_solution_number):
            new_sol = self.create_random_kraemer_solution()
            self.population.append(new_sol)

        for _ in range(dirichlet_solution_number):
            new_sol = self.create_random_dirichlet_solution()
            self.population.append(new_sol)
    
    def add_individual_to_population(self, sol):
        self.population.append(sol)

    def add_multiple_individuals_to_population(self, sols):
        self.population.extend(sols)

    # ---
    # History
    # ---
    @property
    def hist(self):
        return self.history

    # ---
    # Mutation Operators
    # ---
    def mutation_flow(self, sol):
        sol = np.copy(sol)
        idx1, idx2 = random.sample(range(self.n), 2)
        delta = random.uniform(0, min(sol[idx1], sol[idx2]))
        sol[idx1] -= delta
        sol[idx2] += delta
        return sol

    def mutation_swap(self, sol):
        sol = np.copy(sol)
        idx1, idx2 = random.sample(range(self.n), 2)
        sol[idx1], sol[idx2] = sol[idx2], sol[idx1]
        return sol

    def mutation_polynomial(self, sol):
        new_sol = np.copy(sol)

        u = np.random.rand(self.n)
        eta = self.eta_m
        mask_low = u <= 0.5
        mask_high = u > 0.5

        delta = np.zeros(self.n)
        delta[mask_low] = (2 * u[mask_low]) ** (1 / (eta + 1)) - 1
        delta[mask_high] = 1 - (2 * (1 - u[mask_high])) ** (1 / (eta + 1))

        mutation_mask = np.random.rand(self.n) < (1.0 / self.n)
        if not np.any(mutation_mask):  # Mutate at least one gene
            mutation_mask[np.random.randint(0, self.n)] = True

        new_sol[mutation_mask] += delta[mutation_mask]
        return project_onto_simplex(new_sol)

    def mutation_distribute(self, sol):
        new_sol = sol.copy()
        non_zero_indices = np.where(new_sol > 1e-6)[0]

        # Must be more than 2 non-zero weights
        if len(non_zero_indices) < 2:
            return new_sol

        idx_to_zero = random.choice(non_zero_indices)
        weight_to_distribute = new_sol[idx_to_zero]

        new_sol[idx_to_zero] = 0.0

        portion = weight_to_distribute / (len(non_zero_indices) - 1)
        for i in non_zero_indices:
            if i != idx_to_zero:
                new_sol[i] += portion

        return new_sol

    def mutation(self, sol):
        assert self.mutation_method in MutationMethod
        return getattr(self, f'mutation_{self.mutation_method.value}')(sol)

    # ---
    # Crossover Operators
    # ---
    def crossover_arithmetic(self, sol1, sol2):
        # Arithmetic crossover: new_sol = alpha * sol1 + (1 - alpha) * sol2
        alpha = random.random()
        child1 = alpha * np.copy(sol1) + (1 - alpha) * np.copy(sol2)
        child2 = alpha * np.copy(sol2) + (1 - alpha) * np.copy(sol1)
        return child1, child2

    def crossover_one_point(self, sol1, sol2):
        # One-point crossover - very bad idea
        idx = random.randint(1, self.n - 1)
        child1 = np.concatenate((sol1[:idx], sol2[idx:]))
        child2 = np.concatenate((sol2[:idx], sol1[idx:]))
        return project_onto_simplex(child1), project_onto_simplex(child2)

    def crossover_sbx(self, sol1, sol2):
        # SBX - Simulated Binary Crossover
        if random.random() < 0.5:
            sol1, sol2 = sol2, sol1

        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (self.eta_c + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1))

        offspring1 = 0.5 * ((1 + beta) * sol1 + (1 - beta) * sol2)
        offspring2 = 0.5 * ((1 - beta) * sol1 + (1 + beta) * sol2)

        return project_onto_simplex(offspring1), project_onto_simplex(offspring2)

    def crossover(self, sol1, sol2):
        assert self.crossover_method in CrossoverMethod
        return getattr(self, f'crossover_{self.crossover_method.value}')(sol1, sol2)

    # ---
    # Reproduction of population
    # ---
    def create_offspring(self, sol1, sol2):
        if random.random() < self.crossover_prob:
            child1, child2 = self.crossover(sol1, sol2)
        else:
            child1 = np.copy(sol1)
            child2 = np.copy(sol2)

        if random.random() < self.mutation_prob:
            child1 = self.mutation(child1)

        if random.random() < self.mutation_prob:
            child2 = self.mutation(child2)

        return child1, child2

    def fitness_function(self, sol):
        # to be overridden in subclass
        pass

    def dominates(self, sol1, sol2):
        # to be overridden in subclass
        pass

    def find_pareto_fronts(self):
        n = len(self.population)
        domination_counts = [0] * n
        dominated_solutions = [set() for _ in range(n)]
        fronts = [[]]

        # Calculate domination relations
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(self.population[i], self.population[j]):
                    dominated_solutions[i].add(j)
                elif self.dominates(self.population[j], self.population[i]):
                    domination_counts[i] += 1

            if domination_counts[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        return fronts

    def choose_new_population(self):
        pass

    # ---
    # Main Loop
    # ---
    def evolve(self):
        pass

    # ---
    # Plotting
    # ---
    def plot_pareto_front(self, title=None):
        pass

    def plot_initial_population(self):
        self.evaluate_population()
        self.plot_pareto_front(title='Initial Population')
