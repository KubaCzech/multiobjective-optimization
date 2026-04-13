import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ..nsga import CrossoverMethod, MutationMethod
from .nsga_iii import NSGAIII

class NSGAIIIThreeObjectives(NSGAIII):
    def __init__(
            self, 
            prices, 
            risk_matrix, 
            pop_size, 
            p, 
            eps=0.001, 
            dirichlet_alpha=0.2, 
            mutation_prob=0.2, 
            crossover_prob=0.8, 
            mutation_method=MutationMethod.polynomial, 
            crossover_method=CrossoverMethod.SBX,
            eta_c=1.5,
            eta_m=1.3,
            elitism=True
            ):
        super().__init__(
            prices, 
            risk_matrix, 
            pop_size, 
            n_objectives=3, 
            p=p, 
            dirichlet_alpha=dirichlet_alpha, 
            mutation_prob=mutation_prob, 
            crossover_prob=crossover_prob, 
            directions=[+1, -1, +1],
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            eta_m=eta_m,
            eta_c=eta_c,
            elitism=elitism
        )
        self.eps = eps

    def dominates(self, sol1, sol2):
        eval1 = self.fitness_function(sol1)
        eval2 = self.fitness_function(sol2)
        if eval1[0] > eval2[0] and eval1[1] <= eval2[1] and eval1[2] >= eval2[2]:
            return True
        if eval1[0] >= eval2[0] and eval1[1] < eval2[1] and eval1[2] >= eval2[2]:
            return True
        if eval1[0] >= eval2[0] and eval1[1] <= eval2[1] and eval1[2] > eval2[2]:
            return True
        return False
    
    def on_before_normalization(self, temp_scores):
        temp_scores[:, 1] = np.sqrt(temp_scores[:, 1])
        return temp_scores

    def fitness_function(self, sol):
        risk = sol @ self.risk_matrix @ sol
        price = sol @ self.prices
        diversity_coeff = len(set(np.where(sol > self.eps)[0])) / self.n
        return price, risk, diversity_coeff


    def plot_pareto_front(self, title='Price vs Risk vs Diversity Coefficient Pareto Front (NSGA-III for Two Objectives)'):
        prices = [score[0] for score in self.scores]
        risks = [score[1] for score in self.scores]
        diversities = [score[2] for score in self.scores]

        fig = go.Figure(data=[go.Scatter3d(
            x=risks,
            y=prices,
            z=diversities,
            mode='markers',
            marker=dict(
                size=5,
                color=prices,                # Kolorowanie po cenie
                colorscale='Viridis',        # Ładna paleta barw
                opacity=0.8
            ),
            text=[f"Price: {p:.2f}<br>Risk: {r:.4f}<br>Div: {d:.2f}" 
                for p, r, d in zip(prices, risks, diversities)]
        )])

        fig.update_layout(
            title='3D Pareto Front',
            scene=dict(
                xaxis_title='Risk (Min)',
                yaxis_title='Price (Max)',
                zaxis_title='Diversity (Max)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()