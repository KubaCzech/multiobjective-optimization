import random
import numpy as np

# * sprawdzic random solution: TODO
# * crossover: TODO (obecnie jest troche biased, rada Tomczyka: sbx crossover)
# * mutation: TODO (ale mozna zrobic a la sbx)
# * wystrzegac sie dzielenia przez sume: TODO
# * narysowac reference lines: TODO
# * poprawic normalizacje: TODO
# * narysowac pareto fronty dobrze: TODO
# * porownac z benchmarkami (np. equal weights, ECM, WSM): TODO
class NSGAIII:
    def __init__(self, prices, risk_matrix, pop_size, nr_of_iterations, n_objectives, p, dirichlet_alpha=0.2, mutation_prob=0.1, crossover_prob=0.8):
        self.prices = prices
        self.risk_matrix = risk_matrix
        self.scores = []
        self.n_objectives = n_objectives

        self.population = []
        self.pop_size = pop_size
        self.nr_of_iterations = nr_of_iterations
        self.n = len(prices) # number of assets
        self.p = p
        self.dirichlet_alpha = dirichlet_alpha

        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        self.reference_points = self.generate_reference_points()
        self.create_initial_population()

    def get_population(self):
        return self.population

    def generate_reference_points(self):
        pass

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

    def mutation(self, sol):
        sol = np.copy(sol)
        idx1, idx2 = random.sample(range(self.n), 2)
        delta = random.uniform(0, min(sol[idx1], sol[idx2]))
        sol[idx1] -= delta
        sol[idx2] += delta
        return sol

    def crossover(self, sol1, sol2):
        # Arithmetic crossover: new_sol = alpha * sol1 + (1 - alpha) * sol2
        alpha = random.random()
        child1 = alpha * np.copy(sol1) + (1 - alpha) * np.copy(sol2)
        child2 = alpha * np.copy(sol2) + (1 - alpha) * np.copy(sol1)
        return child1, child2

        # One-point crossover
        # idx = random.randint(1, self.n - 1)
        # child1 = np.concatenate((sol1[:idx], sol2[idx:]))
        # child2 = np.concatenate((sol2[:idx], sol1[idx:]))
        # return child1/sum(child1), child2/sum(child2)
    
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

    def normalize_population(self):
        # Normalize the population's objective values to [0, 1] range
        scores_array = np.array(self.scores)
        min_scores = np.min(scores_array, axis=0)
        max_scores = np.max(scores_array, axis=0)
        normalized_scores = (scores_array - min_scores) / (max_scores - min_scores + 1e-9)
        self.scores = [tuple(i) for i in normalized_scores.tolist()]

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
        
        # Prepare indices for niching selection
        # Use only the already selected individuals and those from the last front
        current_scores = np.array([self.scores[i] for i in new_population_indices])
        last_front_scores = np.array([self.scores[i] for i in last_front])
        
        chosen_from_last_front = self.niching_selection(current_scores, last_front_scores, solution_counter)
        final_indices = new_population_indices + [last_front[i] for i in chosen_from_last_front]
        
        # Update population
        scores = [self.scores[i] for i in final_indices]
        population = [self.population[i] for i in final_indices]

        assert len(population) == self.pop_size, f"Expected population size {self.pop_size}, got {len(population)}"
        assert len(scores) == self.pop_size, f"Expected scores size {self.pop_size}, got {len(scores)}"

        return population, scores

    def evolve(self):
        for it_number in range(self.nr_of_iterations):
            if (it_number+1) % 50 == 0:
                print(f"Iteration {it_number+1}/{self.nr_of_iterations}")
            new_population = []
            while len(new_population) < self.pop_size:
                parent1, parent2 = random.sample(self.population, 2)
                child1, child2 = self.create_offspring(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)

            self.population.extend(new_population)
            self.evaluate_population()

            fronts = self.find_pareto_fronts()
            self.normalize_population()

            self.population, self.scores = self.choose_new_population(fronts)
        # self.normalize_population()
        return self.population

    def normalize_pareto_front(self):
        # to be overriden in subclass
        pass

    def plot_initial_population(self):
        self.evaluate_population()
        self.plot_pareto_front(title='Initial Population')