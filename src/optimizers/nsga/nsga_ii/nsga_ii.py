import random
import numpy as np

from ..nsga import CrossoverMethod, MutationMethod, NSGA

# mutacja: wrzucasz wage assetow po rowno do niezerowych wektorow

class NSGAII(NSGA):
    def __init__(
            self,
            prices,
            risk_matrix,
            pop_size,
            n_objectives,
            dirichlet_alpha=0.2,
            crossover_prob=0.8,
            mutation_prob=0.2,
            eta_c=5,
            eta_m=15,
            crossover_method=CrossoverMethod.SBX,
            mutation_method=MutationMethod.polynomial, 
            directions = None,
        ):
        super().__init__(
            prices=prices,
            risk_matrix=risk_matrix,
            pop_size=pop_size,
            n_objectives=n_objectives,
            dirichlet_alpha=dirichlet_alpha,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            eta_c=eta_c,
            eta_m=eta_m,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            directions=directions

        )

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
