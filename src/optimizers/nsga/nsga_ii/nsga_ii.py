import random
import numpy as np
import matplotlib.pyplot as plt

from ..nsga import CrossoverMethod, MutationMethod, EliteSolutionsArchive


class NSGAII:
    def __init__(
            self,
            prices,
            risk_matrix,
            pop_size,
            n_objectives,
            dirichlet_alpha=0.2,
            mutation_prob=0.2,
            crossover_prob=0.8,
            crossover_method=CrossoverMethod.SBX,
            mutation_method=MutationMethod.polynomial, 
            directions = None,
            elitism=True
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

        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        if elitism:
            self.elite_solutions_archive = EliteSolutionsArchive(directions)
        self.elitism = elitism

        self.history = [] # for each generation: (return, risk, optionally: div_coeff, solution)

        self.create_initial_population()

    def get_population(self):
        return self.population

    def create_random_kraemer_solution(self):
        cuts = np.random.rand(self.n - 1)
        cuts.sort()
        points = np.concatenate(([0.0], cuts, [1.0]))
        return np.diff(points)

    def create_random_dirichlet_solution(self):
        return np.random.dirichlet([self.dirichlet_alpha] * self.n)

    def create_initial_population(self):
        kraemer_n = int(self.pop_size * 0.75)
        dirichlet_n = self.pop_size - kraemer_n
        for _ in range(kraemer_n):
            self.population.append(self.create_random_kraemer_solution())
        for _ in range(dirichlet_n):
            self.population.append(self.create_random_dirichlet_solution())

    # -------------------------------------------------------------------------
    # Fitness and dominance
    # -------------------------------------------------------------------------

    def fitness_function(self, sol):
        pass

    def dominates(self, sol1, sol2):
        pass

    def evaluate_population(self):
        self.scores = []
        for sol in self.population:
            score = self.fitness_function(sol)
            if self.elitism:
                self.elite_solutions_archive.add_solution(sol, list(score))
            self.scores.append(score)

    # -------------------------------------------------------------------------
    # Non-dominated sorting
    # -------------------------------------------------------------------------

    def find_pareto_fronts(self):
        n = len(self.population)
        domination_counts = [0] * n
        dominated_solutions = [set() for _ in range(n)]
        fronts = [[]]

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

    # -------------------------------------------------------------------------
    # Crowding distance
    # -------------------------------------------------------------------------

    def crowding_distance_assignment(self, front_indices):
        """
        Compute crowding distance for each solution in the front.
        Returns distances array aligned to positions in front_indices.
        """
        n = len(front_indices)
        distances = np.zeros(n)

        if n <= 2:
            distances[:] = np.inf
            return distances

        scores = np.array(self.scores)[front_indices]

        for obj in range(self.n_objectives):
            sorted_pos = np.argsort(scores[:, obj])
            distances[sorted_pos[0]] = np.inf
            distances[sorted_pos[-1]] = np.inf

            obj_min = scores[sorted_pos[0], obj]
            obj_max = scores[sorted_pos[-1], obj]
            denom = obj_max - obj_min
            if denom < 1e-12:
                continue

            for k in range(1, n - 1):
                distances[sorted_pos[k]] += (
                    scores[sorted_pos[k + 1], obj] - scores[sorted_pos[k - 1], obj]
                ) / denom

        return distances

    # -------------------------------------------------------------------------
    # Variation operators
    # -------------------------------------------------------------------------

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
        eta_m = 15
        sol = np.copy(sol)
        for i in range(self.n):
            if random.random() < (1.0 / self.n):
                u = random.random()
                delta = (2 * u) ** (1 / (eta_m + 1)) - 1 if u <= 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                sol[i] += delta
        sol = np.maximum(sol, 1e-6)
        return sol / np.sum(sol)

    def mutation(self, sol):
        return getattr(self, f'mutation_{self.mutation_method.value}')(sol)

    def crossover_arithmetic(self, sol1, sol2):
        alpha = random.random()
        child1 = alpha * sol1 + (1 - alpha) * sol2
        child2 = alpha * sol2 + (1 - alpha) * sol1
        return child1, child2

    def crossover_one_point(self, sol1, sol2):
        idx = random.randint(1, self.n - 1)
        child1 = np.concatenate((sol1[:idx], sol2[idx:]))
        child2 = np.concatenate((sol2[:idx], sol1[idx:]))
        return child1 / child1.sum(), child2 / child2.sum()

    def crossover_sbx(self, sol1, sol2):
        eta_c = 5
        child1 = np.zeros(self.n)
        child2 = np.zeros(self.n)

        for i in range(self.n):
            if random.random() <= 0.9:
                if abs(sol1[i] - sol2[i]) > 1e-9:
                    u = random.random()
                    beta = (2 * u) ** (1.0 / (eta_c + 1)) if u <= 0.5 else (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    child1[i] = 0.5 * ((1 + beta) * sol1[i] + (1 - beta) * sol2[i])
                    child2[i] = 0.5 * ((1 - beta) * sol1[i] + (1 + beta) * sol2[i])
                else:
                    child1[i], child2[i] = sol1[i], sol2[i]
            else:
                child1[i], child2[i] = sol1[i], sol2[i]

        def repair(child):
            child = np.maximum(child, 0)
            s = np.sum(child)
            return np.full(self.n, 1.0 / self.n) if s < 1e-12 else child / s

        return repair(child1), repair(child2)

    def crossover(self, sol1, sol2):
        return getattr(self, f'crossover_{self.crossover_method.value}')(sol1, sol2)

    def create_offspring(self, sol1, sol2):
        if random.random() < self.crossover_prob:
            child1, child2 = self.crossover(sol1, sol2)
        else:
            child1, child2 = np.copy(sol1), np.copy(sol2)

        if random.random() < self.mutation_prob:
            child1 = self.mutation(child1)
        if random.random() < self.mutation_prob:
            child2 = self.mutation(child2)

        return child1, child2

    # -------------------------------------------------------------------------
    # Tournament selection
    # -------------------------------------------------------------------------

    def tournament_select(self, ranks, distances):
        """
        Binary tournament: lower rank wins; ties broken by higher crowding distance.
        Returns index into self.population.
        """
        a, b = random.sample(range(len(self.population)), 2)
        if ranks[a] < ranks[b]:
            return a
        elif ranks[b] < ranks[a]:
            return b
        else:
            return a if distances[a] >= distances[b] else b

    # -------------------------------------------------------------------------
    # Next generation selection
    # -------------------------------------------------------------------------

    def choose_new_population(self, fronts):
        """
        Fill P_{t+1} greedily by fronts; resolve the last partial front
        by crowding distance (descending).
        """
        new_indices = []

        for front in fronts:
            if len(new_indices) + len(front) <= self.pop_size:
                new_indices.extend(front)
            else:
                needed = self.pop_size - len(new_indices)
                dists = self.crowding_distance_assignment(front)
                sorted_by_dist = sorted(range(len(front)), key=lambda i: dists[i], reverse=True)
                new_indices.extend(front[sorted_by_dist[i]] for i in range(needed))
                break

        population = [self.population[i] for i in new_indices]
        scores = [self.scores[i] for i in new_indices]

        assert len(population) == self.pop_size
        return population, scores

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def evolve(self, nr_of_iterations, plot=False, log_after=25):
        if not log_after and plot:
            raise ValueError
        self.evaluate_population()

        for it_number in range(nr_of_iterations):
            if not log_after:
                pass
            elif (it_number+1) % log_after == 0:
                print(f"Iteration {it_number+1}/{nr_of_iterations}")
                if plot:
                    self.plot_pareto_front(title = f'Pareto front in {it_number+1} iteration')

            # Rank and crowding distance over P_t for tournament
            fronts = self.find_pareto_fronts()
            ranks = np.zeros(len(self.population), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            all_distances = np.zeros(len(self.population))
            for front in fronts:
                dists = self.crowding_distance_assignment(front)
                for pos, idx in enumerate(front):
                    all_distances[idx] = dists[pos]

            # Generate Q_t via binary tournament selection
            offspring = []
            while len(offspring) < self.pop_size:
                p1_idx = self.tournament_select(ranks, all_distances)
                p2_idx = self.tournament_select(ranks, all_distances)
                c1, c2 = self.create_offspring(self.population[p1_idx], self.population[p2_idx])
                offspring.append(c1)
                offspring.append(c2)

            # R_t = P_t ∪ Q_t, evaluate R_t
            self.population = list(self.population) + offspring
            self.evaluate_population()
            self.scores = np.array(self.scores)

            # Select P_{t+1} from R_t
            fronts = self.find_pareto_fronts()
            self.population, self.scores = self.choose_new_population(fronts)
            self.history.append(self.scores)

        return self.history

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_initial_population(self):
        self.evaluate_population()
        self.plot_pareto_front(title='Initial Population')

    def plot_pareto_front(self, title='Pareto Front - Price vs Risk (NSGA-II)'):
        scores = np.array(self.scores)
        prices = scores[:, 0]
        risks = scores[:, 1]

        plt.figure(figsize=(5, 5))
        plt.scatter(prices, risks, c='blue', marker='o')
        plt.title(title)
        plt.xlabel('Price [max]')
        plt.ylabel('Risk [min]')
        plt.grid()
        plt.show()
