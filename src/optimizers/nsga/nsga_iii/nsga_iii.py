import random
import numpy as np
from enum import Enum
from itertools import combinations

from ..nsga import UtopianPointArchive, NSGA, project_onto_simplex


class NSGAIII(NSGA):
    def __init__(
        self,
        prices,
        risk_matrix,
        pop_size,
        n_objectives,
        p,
        dirichlet_alpha,
        crossover_prob,
        mutation_prob,
        eta_c,
        eta_m,
        crossover_method,
        mutation_method,
        directions,
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
            directions=directions,
        )
        self.normalized_scores = []

        self.p = p

        self.utopian_point_archive = UtopianPointArchive(directions)
        self.nadir_point = []

        self.reference_points = self.generate_reference_points()

    def generate_reference_points(self):
        ref_points = []
        # Das-Dennis method (Stars and Bars)
        for combo in combinations(range(self.p + self.n_objectives - 1), self.n_objectives - 1):
            combo = [-1] + list(combo) + [self.p + self.n_objectives - 1]

            point = []
            for i in range(len(combo) - 1):
                val = (combo[i + 1] - combo[i] - 1) / self.p
                point.append(val)

            ref_points.append(point)

        return np.array(ref_points)

    def evaluate_population(self):
        super().evaluate_population()
        for score in self.scores:
            self.utopian_point_archive.add_solution(score)

    def on_before_normalization(self, temp_scores):
        temp_scores[:, 1] = np.sqrt(temp_scores[:, 1])
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

            if self.directions[dim] == 1:
                if np.isinf(z_ideal):
                    z_ideal = np.max(temp_scores[:, dim])
                normalized = (z_ideal - temp_scores[:, dim]) / abs(z_ideal - z_nadir + 1e-9)
            elif self.directions[dim] == -1:
                if np.isinf(z_ideal):
                    z_ideal = np.min(temp_scores[:, dim])
                normalized = (temp_scores[:, dim] - z_ideal) / abs(z_ideal - z_nadir + 1e-9)
            else:
                raise ValueError
            scores_array[:, dim] = normalized
        self.normalized_scores = scores_array

    def niching_selection(self, current_scores, last_front_scores, solution_counter):
        # 1. Helper function to associate solutions with reference points based on perpendicular distance
        def associate(scores):
            ref_norms = np.linalg.norm(self.reference_points, axis=1)

            scalar_products = np.dot(scores, self.reference_points.T)
            s_norms = np.linalg.norm(scores, axis=1)[:, np.newaxis]

            ref_norms[ref_norms == 0] = 1e-10
            distances = np.sqrt(np.abs(s_norms**2 - (scalar_products / ref_norms) ** 2))

            assoc_indices = np.argmin(distances, axis=1)
            assoc_dists = np.min(distances, axis=1)
            return assoc_indices, assoc_dists

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
                temp_niche_counts = niche_counts.copy() + 1e-3  # małe zaburzenie

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
            return [self.population[i] for i in new_population_indices], [
                self.scores[i] for i in new_population_indices
            ]

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
            elif (it_number + 1) % log_after == 0:
                print(f"Iteration {it_number+1}/{nr_of_iterations}")
                if plot:
                    self.plot_pareto_front(title=f'Pareto front in {it_number+1} iteration')
            new_population = []
            while len(new_population) < self.pop_size:
                parent1, parent2 = random.sample(self.population, 2)
                child1, child2 = self.create_offspring(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)

            self.population.extend(new_population)
            self.evaluate_population()

            self.scores = np.array(self.scores)

            fronts = self.find_pareto_fronts()
            self.normalize_population()

            self.population, self.scores = self.choose_new_population(fronts)
            self.history.append(self.scores)
        return self.history
