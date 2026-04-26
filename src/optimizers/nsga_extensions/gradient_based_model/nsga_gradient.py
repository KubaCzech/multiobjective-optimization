import random
import numpy as np

from ...nsga import NSGA, check_constraints, project_onto_simplex
from ...nsga.nsga_iii.nsga_iii import NSGAIII

class GradientNSGAIII(NSGAIII):
    def __init__(self, *args, refinement_interval = 10, refinement_steps = 10, refinement_prob = 0.1, lr=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.refinement_interval = refinement_interval
        self.refinement_steps = refinement_steps
        self.refinement_prob = refinement_prob
        self.lr = lr

    # def manual_gradient_step(self, weights, alpha_vector):
    #     """
    #     Single step of gradient-based refinement for a single solution.
    #     """    
    #     w_profit, w_risk = alpha_vector
    #     # 1. Calculation of gradients
    #     grad_return = self.prices
    #     grad_risk = 2 * np.dot(self.risk_matrix, weights)
        
    #     # 2. Scalarization and combination of gradients
    #     total_gradient = w_risk * grad_risk - w_profit * grad_return
        
    #     delta_w = total_gradient - np.mean(total_gradient)
        
    #     decreasing_indices = delta_w > 0
    #     if np.any(decreasing_indices):
    #         max_possible_steps = weights[decreasing_indices] / delta_w[decreasing_indices]
    #         limit_eta = np.min(max_possible_steps)
            
    #         safe_step = min(self.lr, limit_eta * 0.95)
    #     else:
    #         safe_step = self.lr

    #     new_weights = weights - safe_step * delta_w
    #     if not check_constraints(new_weights):
    #         return project_onto_simplex(new_weights)  # As a last resort, project onto simplex; this is needed only because of numerical issues
    #     return new_weights

    def manual_gradient_step(self, weights, alpha_vector):
        w_profit, w_risk = alpha_vector
        
        # 1. Calculation of gradients
        grad_return = self.prices
        grad_risk = 2 * np.dot(self.risk_matrix, weights)
        
        # 2. Normalization with L2 norm
        norm_ret = np.linalg.norm(grad_return)
        norm_risk = np.linalg.norm(grad_risk)
        
        if norm_ret > 0: grad_return /= norm_ret
        if norm_risk > 0: grad_risk /= norm_risk
        
        # 3. Scalarization (Direction: we minimize risk, maximize return)
        # We subtract the return gradient because we want to go "up" the return
        total_gradient = w_risk * grad_risk - w_profit * grad_return
        
        # 4. Gradient step
        new_weights = weights - self.lr * total_gradient
        
        # 5. Simplex projection to ensure feasibility (handles both constraints at once)
        return project_onto_simplex(new_weights)

    def get_gradient_directions(self):
        pass

    def perform_gradient_iteration(self):
        # Sort population by current returns to assign alpha values (we make sure that solutions 
        # with lower returns get smaller alpha, thus focusing more on risk reduction)
        directions = self.get_gradient_directions()
        
        new_population = self.population[:]
        for i in range(int(self.pop_size * self.refinement_prob)):
            # if random.random() <= self.refinement_prob:
            w = np.copy(self.population[i])
            alpha_vec = directions[i]
                
            for _ in range(self.refinement_steps):
                w = self.manual_gradient_step(w, alpha_vec)
            new_population[i] = w
                
        self.population = new_population
        self.evaluate_population()

    def evolve(self, nr_of_iterations,  log_after=25):
        for it_number in range(1, nr_of_iterations + 1):
            # Every `refinement_interval` iterations, perform gradient-based refinement instead of regular NSGA-III steps
            if it_number % self.refinement_interval == 0:
                self.perform_gradient_iteration()
            else:
                new_offspring = []
                while len(new_offspring) < self.pop_size:
                    p1, p2 = random.sample(self.population, 2)
                    c1, c2 = self.create_offspring(p1, p2)
                    new_offspring.append(c1)
                    new_offspring.append(c2)

                self.population.extend(new_offspring)
                self.evaluate_population()
                
                fronts = self.find_pareto_fronts()
                self.normalize_population()
                self.population, self.scores = self.choose_new_population(fronts)
            
            self.history.append(self.scores)
            
            if log_after and it_number % log_after == 0:
                print(f"Iteration {it_number}/{nr_of_iterations} {'[GRADIENT]' if it_number % self.refinement_interval == 0 else ''}")
        return self.history