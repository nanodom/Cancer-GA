import numpy as np

def plot_convergence(history, save_path=None, show=True):
    """Grafica la curva de convergencia para problemas mono-objetivo."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["best_fitness"], color="#2E86AB", linewidth=2)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor valor objetivo")
    ax.set_title("Convergencia del algoritmo")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")
    if show:
        plt.show()
    return fig, ax


def plot_pareto_front(elite_archive, n_obj, sign, knee_result=None, save_path=None, show=True):
    """Grafica el frente de Pareto actual (M=2 en 2D, M=3 en 3D)."""
    if n_obj < 2 or n_obj > 3:
        raise ValueError("plot_pareto_front() soporta problemas de 2 o 3 objetivos.")

    import matplotlib.pyplot as plt

    if not elite_archive:
        raise ValueError("No hay individuos en el frente de Pareto élite para graficar.")
        
    objs = np.array([ind.obj * sign for ind in elite_archive])  # Escala original

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(objs[:, 0], objs[:, 1], c="#E63946", s=45,
                   edgecolor="white", zorder=3, label="Frente de Pareto")
        if knee_result is not None:
            _, knee_obj, _ = knee_result
            ax.scatter(knee_obj[0], knee_obj[1], marker="*", s=350, c="gold",
                       edgecolor="black", zorder=5, label="Punto codo")
        ax.set_xlabel("Objetivo 1")
        ax.set_ylabel("Objetivo 2")
    else:  # n_obj == 3
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], c="#E63946", s=40,
                   label="Frente de Pareto")
        if knee_result is not None:
            _, knee_obj, _ = knee_result
            ax.scatter(knee_obj[0], knee_obj[1], knee_obj[2], marker="*", s=350,
                       c="gold", edgecolor="black", label="Punto codo")
        ax.set_xlabel("Objetivo 1")
        ax.set_ylabel("Objetivo 2")
        ax.set_zlabel("Objetivo 3")

    ax.set_title(f"Frente de Pareto ({len(elite_archive)} soluciones)")
    ax.legend()
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")
    if show:
        plt.show()
    return fig, ax