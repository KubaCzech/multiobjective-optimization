import random
import numpy as np
from enum import Enum
from itertools import combinations

from ..nsga import CrossoverMethod, MutationMethod, UtopianPointArchive, EliteSolutionsArchive

class NSGAIII:
    def __init__(
            self, 
            prices, 
            risk_matrix, 
            pop_size, 
            n_objectives, 
            p, 
            dirichlet_alpha=0.2, 
            mutation_prob=0.1, 
            crossover_prob=0.8, 
            crossover_method=CrossoverMethod.SBX,
            mutation_method=MutationMethod.polynomial,
            eta_c=5,
            eta_m=20,
            directions=None, 
            elitism=True
        ):
        self.prices = prices
        self.risk_matrix = risk_matrix
        self.scores = []
        self.normalized_scores = []
        self.n_objectives = n_objectives
        # directions: +1 = maximize, -1 = minimize (one per objective)
        self.directions = np.array(directions) if directions is not None else np.full(n_objectives, -1)

        self.population = []
        self.pop_size = pop_size
        self.n = len(prices) # number of assets
        self.p = p
        self.dirichlet_alpha = dirichlet_alpha

        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method

        self.eta_c = eta_c
        self.eta_m = eta_m

        if elitism:
            self.elite_solutions_archive = EliteSolutionsArchive(directions)
        self.elitism = elitism
        self.utopian_point_archive = UtopianPointArchive(directions)
        self.nadir_point = []

        self.history = [] # for each generation: (return, risk, optionally: div_coeff, solution)

        self.reference_points = self.generate_reference_points()
        self.create_initial_population()

    def get_population(self):
        return self.population

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

    def create_random_kraemer_solution(self):
        cuts = np.random.rand(self.n - 1)
        cuts.sort()
        points = np.concatenate(([0.0], cuts, [1.0]))
        weights = np.diff(points) # Weights are the intervals between the sorted random cuts, ensuring they sum to 1
            
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
        eta_m = 1.3
        sol = np.copy(sol)
        for i in range(self.n):
            if random.random() < (1.0 / self.n):
                u = random.random()
                delta = (2*u)**(1/(eta_m+1)) - 1 if u <= 0.5 else 1 - (2*(1-u))**(1/(eta_m+1))
                sol[i] += delta
        
        sol = np.maximum(sol, 1e-6)
        return sol / np.sum(sol)
    
    def mutation(self, sol):
        assert self.mutation_method in MutationMethod
        return getattr(self, f'mutation_{self.mutation_method.value}')(sol)

    def crossover_arithmetic(self, sol1, sol2):
        # Arithmetic crossover: new_sol = alpha * sol1 + (1 - alpha) * sol2
        alpha = random.random()
        child1 = alpha * np.copy(sol1) + (1 - alpha) * np.copy(sol2)
        child2 = alpha * np.copy(sol2) + (1 - alpha) * np.copy(sol1)
        return child1, child2

    def crossover_one_point(self, sol1, sol2):
        # One-point crossover - bardzo kiepski pomysl
        idx = random.randint(1, self.n - 1)
        child1 = np.concatenate((sol1[:idx], sol2[idx:]))
        child2 = np.concatenate((sol2[:idx], sol1[idx:]))
        return child1/sum(child1), child2/sum(child2)

    def crossover_sbx(self, sol1, sol2):
        # SBX - Simulated Binary Crossover
        eta_c = 1.5
        child1 = np.zeros(self.n)
        child2 = np.zeros(self.n)
        
        for i in range(self.n):
            if random.random() <= 0.9:
                if abs(sol1[i] - sol2[i]) > 1e-9:
                    u = random.random()
                    beta = (2 * u)**(1.0/(eta_c + 1)) if u <= 0.5 else (1.0/(2*(1-u)))**(1.0/(eta_c + 1))
                    child1[i] = 0.5 * ((1 + beta) * sol1[i] + (1 - beta) * sol2[i])
                    child2[i] = 0.5 * ((1 - beta) * sol1[i] + (1 + beta) * sol2[i])
                else:
                    child1[i], child2[i] = sol1[i], sol2[i]
            else:
                child1[i], child2[i] = sol1[i], sol2[i]

        # Additive correction
        def repair(child):
            child = np.maximum(child, 0)
            s = np.sum(child)
            # If sum == 0, we assign equal weights -> highly unlikely
            if s < 1e-12:
                return np.full(self.n, 1.0 / self.n)
            return child / s

        return repair(child1), repair(child2)
    
    def crossover(self, sol1, sol2):
        assert self.crossover_method in CrossoverMethod
        return getattr(self, f'crossover_{self.crossover_method.value}')(sol1, sol2)
    
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

    def evaluate_population(self):
        self.scores = []
        for sol in self.population:
            score = self.fitness_function(sol)
            self.utopian_point_archive.add_solution(score)
            if self.elitism:
                self.elite_solutions_archive.add_solution(sol, list(score))
            self.scores.append(score)

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
    
    def on_before_normalization(self, temp_scores):
        return temp_scores

    def normalize_population(self):
        # Normalize the population's objective values to [0, 1] range for niching.
        # Raw scores are preserved in self.scores for plotting and domination checks.
        # Maximized objectives are flipped so all objectives become minimized,
        # placing the ideal point at (0, 0, ...) as required by NSGA-III reference point geometry.
        temp_scores = self.on_before_normalization(np.copy(self.scores))
        scores_array = np.zeros_like(self.scores)
        ideal_point = self.on_before_normalization(self.utopian_point_archive.utopian_point.reshape(1, -1)).flatten()

        min_scores = np.min(temp_scores, axis=0)
        max_scores = np.max(temp_scores, axis=0).reshape(-1, 1)

        for dim in range(scores_array.shape[1]):
            z_ideal = ideal_point[dim]
            z_nadir = max_scores[dim] if self.directions[dim] == -1 else min_scores[dim]

            denom = abs(z_ideal - z_nadir) + 1e-9

            if self.directions[dim] == 1:
                normalized = (z_ideal - temp_scores[:, dim]) / denom
            elif self.directions[dim] == -1:
                normalized = (temp_scores[:, dim] - z_ideal)/denom
            else:
                raise ValueError
            scores_array[:, dim] = normalized
        self.normalized_scores = scores_array

    def niching_selection(self, current_scores, last_front_scores, solution_counter):
        # 1. Helper function to associate solutions with reference points based on perpendicular distance
        def associate(scores):
            assoc_indices = []
            assoc_dists = []
            for s in scores:
                s_val = np.array(s)
                dists = []
                for w in self.reference_points:
                    # Perpendicular distance from s to w
                    w_norm = np.linalg.norm(w)
                    if w_norm == 0:
                        dist = np.linalg.norm(s_val)
                    else:
                        # Projection of s onto w
                        projection = (np.dot(s_val, w) / (w_norm**2)) * w
                        dist = np.linalg.norm(s_val - projection)
                    dists.append(dist)
                
                assoc_indices.append(np.argmin(dists))
                assoc_dists.append(np.min(dists))
            return np.array(assoc_indices), np.array(assoc_dists)

        # 2. Associate current scores and last front scores with reference points
        niche_counts = np.zeros(len(self.reference_points))
        
        if len(current_scores) > 0:
            current_assoc, _ = associate(current_scores)
            for a in current_assoc:
                niche_counts[a] += 1
        
        last_assoc, last_dists = associate(last_front_scores)
        
        # 3. Niche selection - choose 'solution_counter' solutions from last_front based on niche counts
        chosen_from_last = []
        available_indices = list(range(len(last_front_scores)))
        
        # Copy niche counts to modify during selection without affecting the original counts
        temp_niche_counts = niche_counts.copy()

        while len(chosen_from_last) < solution_counter:
            # Find niches with the minimum count
            min_val = np.min(temp_niche_counts)
            min_indices = np.where(temp_niche_counts == min_val)[0]
            chosen_ref_idx = random.choice(min_indices)
            
            # Find individuals from last_front associated with this specific niche
            candidates = [i for i in available_indices if last_assoc[i] == chosen_ref_idx]
            
            if candidates:
                if temp_niche_counts[chosen_ref_idx] == 0:
                    # If the niche is empty, we choose the individual closest to the reference vector
                    best_cand_idx = candidates[np.argmin([last_dists[i] for i in candidates])]
                else:
                    # If the niche has someone already, we choose randomly (for diversity)
                    best_cand_idx = random.choice(candidates)
                
                chosen_from_last.append(best_cand_idx)
                available_indices.remove(best_cand_idx)
                temp_niche_counts[chosen_ref_idx] += 1
            else:
                # If there is no one in this niche in last_front, we exclude it from the minimum search
                temp_niche_counts[chosen_ref_idx] = np.inf
                
            # If all niches have become inf, and we still need people, reset the counters
            if np.all(temp_niche_counts == np.inf) and available_indices:
                temp_niche_counts = niche_counts.copy() + 1e-3 # małe zaburzenie

        return chosen_from_last

    def choose_new_population(self, fronts):
        new_population_indices = []
        
        # If possible, add entire fronts until we reach the population size
        for front in fronts:
            if len(new_population_indices) + len(front) <= self.pop_size:
                new_population_indices.extend(front)
            else:
                last_front = front
                break
        
        # If new population is full, return it
        if len(new_population_indices) == self.pop_size:
            return [self.population[i] for i in new_population_indices], [self.scores[i] for i in new_population_indices]

        # If we have a partially filled population, we need to select from the last front
        solution_counter = self.pop_size - len(new_population_indices)

        # Niching uses normalized scores for fair comparison across objectives
        current_scores = np.array([self.normalized_scores[i] for i in new_population_indices])
        last_front_scores = np.array([self.normalized_scores[i] for i in last_front])

        chosen_from_last_front = self.niching_selection(current_scores, last_front_scores, solution_counter)
        final_indices = new_population_indices + [last_front[i] for i in chosen_from_last_front]

        # Update population — keep raw scores for plotting
        scores = [self.scores[i] for i in final_indices]
        population = [self.population[i] for i in final_indices]

        assert len(population) == self.pop_size, f"Expected population size {self.pop_size}, got {len(population)}"
        assert len(scores) == self.pop_size, f"Expected scores size {self.pop_size}, got {len(scores)}"

        return population, scores

    def evolve(self, nr_of_iterations, plot=False, log_after=25):
        if not log_after and plot:
            raise ValueError
        for it_number in range(nr_of_iterations):
            if not log_after:
                pass
            elif (it_number+1) % log_after == 0:
                print(f"Iteration {it_number+1}/{nr_of_iterations}")
                if plot:
                    self.plot_pareto_front(title = f'Pareto front in {it_number+1} iteration')
            new_population = []
            while len(new_population) < self.pop_size:
                parent1, parent2 = random.sample(self.population, 2)
                child1, child2 = self.create_offspring(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)

            self.population.extend(new_population)
            self.evaluate_population()

            if self.elitism:
                elite = self.elite_solutions_archive.elite_solutions
                for sol in elite:
                    self.population.append(sol)
                    self.scores.append(self.fitness_function(sol))
            self.scores = np.array(self.scores)

            fronts = self.find_pareto_fronts()
            self.normalize_population()

            self.population, self.scores = self.choose_new_population(fronts)
            self.history.append(self.scores)

        return self.history

    def normalize_pareto_front(self):
        # to be overriden in subclass
        pass

    def plot_pareto_front(self, title):
        # to be overriden in subclass
        pass

    def plot_initial_population(self):
        self.evaluate_population()
        self.plot_pareto_front(title='Initial Population')