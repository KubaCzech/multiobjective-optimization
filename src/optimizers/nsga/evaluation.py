import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

def generational_distance(p, p_star):
    # p: our front (N x M), p_star: ideal front (K x M)
    distances = []
    for point in p:
        # Min euclidean distance
        d = np.min(np.linalg.norm(p_star - point, axis=1))
        distances.append(d**2)
    
    return np.sqrt(np.sum(distances)) / len(p)

def inverted_generational_distance(p, p_star):
    # p: our front, p_star: ideal front
    distances = []
    for point_star in p_star:
        # how close to our front is ideal point
        d = np.min(np.linalg.norm(p - point_star, axis=1))
        distances.append(d**2)
        
    return np.sqrt(np.sum(distances)) / len(p_star)

def nadir_point(dim):
    return np.array([1.1 for _ in range(dim)])

def normalize_scores(scores, f_min, f_max):
    return (scores - f_min) / (f_max - f_min + 1e-10)

def hypervolume(scores, ref_point):
    scores_ = np.array(scores)
    return HV(ref_point=ref_point)(scores_)

def sensitivity_analysis_plot(params1: list, params2: list, scores: list, names: list, algo_name: str):
    # x -> generations
    # y -> population sizes
    n = len(params1)
    m = len(params2)
    scores_ = np.array(scores).reshape((m, n))
    
    df = pd.DataFrame(scores_, index=params2, columns=params1)

    plt.figure(figsize=(10, 8))
        
    ax = sns.heatmap(df, 
                     annot=True, 
                     fmt=".3f", 
                     cmap="Blues", 
                     cbar_kws={'label': 'Hypervolume'},
                     linewidths=.5)

    plt.title(f'Sensitivity analysis: {names[0]} vs {names[1]} for {algo_name}', pad=20)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def single_convergence_plot(history, label="Algorithm", ax=None, color=None, title="Convergence Plot", show=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title)

    history = np.array(history)
    mean_scores = np.mean(history, axis=1)
    std_scores = np.std(history, axis=1)
    generations = np.arange(len(mean_scores)) + 1

    if show==True:
        color = 'blue'
    line, = ax.plot(generations, mean_scores, label=label, linewidth=2, color=color)
    current_color = line.get_color()
    
    ax.fill_between(generations, mean_scores - std_scores, mean_scores + std_scores, 
                     alpha=0.2, color=current_color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    if show:
        plt.show()
    
    return ax

def multiple_convergence_plot(histories, labels, title="Comparison of Algorithms"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for history, label in zip(histories, labels):
        single_convergence_plot(history, label=label, ax=ax, show=False)

    ax.set_title(title)
    plt.show()

def plot_multiple_populations(populations, pop_names, title="Populations Comparison"):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.get_cmap('tab10').colors
    markers = ['o', 'x', 's', '^', 'v', '<', '>', 'p', '*', 'h']

    for i, (pop, name) in enumerate(zip(populations, pop_names)):
        pts = np.array(pop)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.scatter(
            pts[:, 0], pts[:, 1], 
            c=[color],
            marker=marker, 
            alpha=0.6, 
            s=40, 
            label=name
        )

    plt.title(title, fontsize=14)
    plt.xlabel('Expected Return [$\max$]', fontsize=12)
    plt.ylabel('Risk [$\min$]', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.show()