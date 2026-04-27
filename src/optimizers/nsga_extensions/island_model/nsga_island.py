import random
import numpy as np
from ...nsga import NSGA

class NSGAIsland:
    def __init__(self, *models: NSGA, n_migrants: int = 2, migration_step: int = 100):
        assert len(models) >= 2, "Island model requires at least 2 islands"
        self.models = list(models)
        self.n_migrants = n_migrants
        self.migration_step = migration_step
        self.history = []

    @property
    def scores(self):
        sc = []
        for model in self.models:
            sc.extend(model.scores)
        return sc

    def get_population(self):
        all_islands = []
        for m in self.models:
            all_islands.extend(m.get_population())
        return all_islands

    def evolve(self, nr_of_iterations: int = 2000, plot_: bool = False, **kwargs):
        assert nr_of_iterations % self.migration_step == 0
        for it_idx in range(nr_of_iterations // self.migration_step):
            for model in self.models:
                model.evolve(nr_of_iterations=self.migration_step, log_after=0)
            if plot_:
                self.plot_islands()
            if ((it_idx+1) * self.migration_step) < nr_of_iterations:
                self._migrate()

        all_hists = [model.hist for model in self.models]
        self.history = [
            np.concatenate(gen_scores, axis=0)
            for gen_scores in zip(*all_hists)
        ]
        return self.history

    def _migrate(self):
        # Ring topology: island i receives migrants from island (i-1) % n
        n = len(self.models)
        migrants = [self.choose_best_individual(m) for m in self.models]
        for i, model in enumerate(self.models):
            model.add_multiple_individuals_to_population(migrants[(i - 1) % n])

    def choose_best_individual(self, model: NSGA):
        fronts = model.find_pareto_fronts()
        chosen = []
        for front in fronts:
            if len(chosen) + len(front) <= self.n_migrants:
                chosen.extend(front)
            else:
                chosen.extend(random.sample(front, self.n_migrants - len(chosen)))
                break
        return [model.get_population()[i] for i in chosen]
    
    def plot_islands(self):
        pass