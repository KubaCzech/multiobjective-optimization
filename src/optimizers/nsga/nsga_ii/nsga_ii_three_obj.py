import numpy as np
import plotly.graph_objects as go

from ..nsga import CrossoverMethod, MutationMethod
from .nsga_ii import NSGAII


class NSGAIIThreeObjectives(NSGAII):
    def __init__(
        self,
        prices,
        risk_matrix,
        pop_size,
        eps=0.001,
        dirichlet_alpha=0.2,
        crossover_prob=0.8,
        mutation_prob=0.2,
        eta_c=5,
        eta_m=15,
        crossover_method=CrossoverMethod.SBX,
        mutation_method=MutationMethod.polynomial,
    ):
        super().__init__(
            prices=prices,
            risk_matrix=risk_matrix,
            pop_size=pop_size,
            n_objectives=3,
            dirichlet_alpha=dirichlet_alpha,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            eta_c=eta_c,
            eta_m=eta_m,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            directions=[+1, -1, +1],
        )
        self.eps = eps

    def fitness_function(self, sol):
        price = sol @ self.prices
        risk = sol @ self.risk_matrix @ sol
        diversity_coeff = len(set(np.where(sol > self.eps)[0])) / self.n
        return price, risk, diversity_coeff  # (maximize, minimize, maximize)

    def dominates(self, sol1, sol2):
        e1 = self.fitness_function(sol1)
        e2 = self.fitness_function(sol2)
        # price: max, risk: min, diversity: max
        better_or_equal = e1[0] >= e2[0] and e1[1] <= e2[1] and e1[2] >= e2[2]
        strictly_better = e1[0] > e2[0] or e1[1] < e2[1] or e1[2] > e2[2]
        return better_or_equal and strictly_better

    def plot_pareto_front(
        self, title='Price vs Risk vs Diversity Coefficient Pareto Front (NSGA-II for three objectives)'
    ):
        prices = [score[0] for score in self.scores]
        risks = [score[1] for score in self.scores]
        diversities = [score[2] for score in self.scores]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=risks,
                    y=prices,
                    z=diversities,
                    mode='markers',
                    marker=dict(size=5, color=prices, colorscale='Viridis', opacity=0.8),
                    text=[
                        f"Price: {p:.2f}<br>Risk: {r:.4f}<br>Div: {d:.2f}"
                        for p, r, d in zip(prices, risks, diversities)
                    ],
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='Risk (Min)', yaxis_title='Price (Max)', zaxis_title='Diversity (Max)'),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()
