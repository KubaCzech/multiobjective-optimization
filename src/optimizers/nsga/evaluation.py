import numpy as np
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

def nadir_point(scores):
    nadir = np.zeros(scores.shape[1])

    for idx in range(scores.shape[1]):
        val = np.max(scores[:, idx])
        nadir[idx] = val + abs(val * 0.01) + 1e-6
    return nadir

def hypervolume(scores, directions, nadir_p = None):
    # Logic: if we know the nadir -> use it, else -> calculate it
    # TODO: a moze lepiej zawsze przyjac reference point jako (1.1)^N
    # TODO: trzeba pomyslec jak zrobic normalizacje i kiedy (w funkcji czy przed)
    scores_ = np.array(scores) * np.array(directions)
    if nadir_p is None:
        nadir = nadir_point(scores_)
    else:
        nadir = nadir_p

    indicator = HV(ref_point=nadir) 
    return indicator(scores_)

def sensitivity_analysis_plot(params1: list, params2: list, scores: list, names: list):
    # x -> generations
    # y -> population sizes
    n = len(params1)
    m = len(params2)
    scores_ = np.array(scores).reshape((m, n))

    plt.imshow(scores_, cmap='inferno', interpolation='nearest', aspect='auto')

    plt.xticks(np.arange(len(params1)), params1, rotation=45)
    plt.yticks(np.arange(len(params2)), params2)

    plt.colorbar(label='Hypervolume')
    plt.title(f'Sensitivity analysis of {names[0]} vs {names[1]}')
    plt.xlabel(names[0])
    plt.ylabel(names[1])

    threshold = scores_.max() / 2.
    for i in range(m):
        for j in range(n):
            color = "white" if scores_[i, j] < threshold else "black"
            plt.text(j, i, f"{scores_[i, j]:.3f}", 
                    ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.show()

def average_convergence_plot(history, title="Average Convergence Plot"):
    mean_scores = np.mean(history, axis=1)
    std_scores = np.std(history, axis=1)
    generations = np.arange(len(mean_scores))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_scores, label='Mean Metric', color='blue')
    
    plt.fill_between(generations, mean_scores - std_scores, mean_scores + std_scores, 
                     alpha=0.2, color='blue', label='Standard Deviation')

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Metric Value (e.g. Hypervolume)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()