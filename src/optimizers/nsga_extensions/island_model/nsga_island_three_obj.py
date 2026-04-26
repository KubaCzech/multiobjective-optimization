import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .nsga_island import NSGAIsland
from ...nsga import CrossoverMethod, MutationMethod, NSGAIIIThreeObjectives


class NSGAIslandThreeObjectives(NSGAIsland):
    def __init__(
            self,
            prices,
            risk_matrix,
            eps=0.001,
            island_pop_size=36,
            p=7,
            n_migrants=5,
            migration_step=100,
        ):
        self.islands_names = ['Exploration Island', 'Exploitation Island', 'Balanced Island']

        model1 = NSGAIIIThreeObjectives(
            prices=prices,
            risk_matrix=risk_matrix,
            eps=eps,
            pop_size=island_pop_size,
            p=p,
            crossover_prob=0.8,
            mutation_prob=0.3,
            eta_c=3.0,
            eta_m=5.0,
            crossover_method=CrossoverMethod.SBX,
            mutation_method=MutationMethod.polynomial,
        )  # Exploration Island

        model2 = NSGAIIIThreeObjectives(
            prices=prices,
            risk_matrix=risk_matrix,
            eps=eps,
            pop_size=island_pop_size,
            p=p,
            crossover_prob=1.0,
            mutation_prob=0.05,
            eta_c=30.0,
            eta_m=100.0,
            crossover_method=CrossoverMethod.arithmetic,
            mutation_method=MutationMethod.polynomial,
        )  # Exploitation Island

        model3 = NSGAIIIThreeObjectives(
            prices=prices,
            risk_matrix=risk_matrix,
            eps=eps,
            pop_size=island_pop_size,
            p=p,
            crossover_prob=0.9,
            mutation_prob=0.1,
            eta_c=20.0,
            eta_m=20.0,
            crossover_method=CrossoverMethod.SBX,
            mutation_method=MutationMethod.polynomial,
        )  # Balanced Island

        super().__init__(model1, model2, model3, n_migrants=n_migrants, migration_step=migration_step)

    def plot_islands(self):
        colors = ['Blues', 'Reds', 'Greens']
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=self.islands_names,
        )
        for col_idx, (model, colorscale) in enumerate(zip(self.models, colors), start=1):
            scores = np.array(model.scores)
            prices, risks, diversities = scores[:, 0], scores[:, 1], scores[:, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=risks, y=prices, z=diversities,
                    mode='markers',
                    marker=dict(size=4, color=prices, colorscale=colorscale, opacity=0.8),
                    text=[f"Price: {p:.2f}<br>Risk: {r:.4f}<br>Div: {d:.2f}" for p, r, d in zip(prices, risks, diversities)],
                    showlegend=False,
                ),
                row=1, col=col_idx,
            )
        axis = dict(xaxis_title='Risk', yaxis_title='Price', zaxis_title='Diversity')
        fig.update_layout(
            scene=axis, scene2=axis, scene3=axis,
            width=1400, height=550,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()

    def plot_pareto_front(self, title='Price vs Risk vs Diversity (Island Model, Three Objectives)'):
        scores = np.array(self.scores)
        prices = scores[:, 0]
        risks = scores[:, 1]
        diversities = scores[:, 2]

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
