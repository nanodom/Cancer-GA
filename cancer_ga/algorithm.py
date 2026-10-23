import numpy as np
from .individual import Individual
from .operators.mutation import adaptive_mutation
from .operators.chemotaxis import chemotaxis
from .operators.duel import biological_duel
from .operators.crossover import crossover
from .operators.proliferation import niche_proliferation, compute_fibroblast_field
from .operators.metastasis import metastasis
from .operators.starvation import starvation
from .utils.nsga2 import assign_rank_and_crowding, fast_non_dominated_sort, crowding_distance


class CancerGA:
    """Algoritmo Genético Co-Evolutivo Bio-Inspirado (CancerGA)."""

    def __init__(
        self,
        objective_function,
        n_dim,
        bounds,
        n_obj=1,
        constraints=None,
        eq_constraints=None,
        is_integer=None,
        n_cancer=60,
        n_lymph=20,
        e_init=1.0,
        sigma_init=0.5,
        sigma_min=1e-5,
        growth_rate=0.4,
        niche_capacity=15,
        niche_radius=1.5,
        crossover_rate=0.7,
        blx_alpha=0.5,
        p_meta0=0.08,
        levy_scale=0.3,
        levy_beta=1.5,
        e_maintenance=0.1,
        e_gain=0.3,
        e_comp=0.05,
        chemo_step=0.2,
        chemo_beta=0.3,
        chemo_fd_eps=1e-5,
        duel_radius=1.0,
        duel_k=2.0,
        use_fibroblasts=True,
        fibro_k=3.0,
        fibro_beta_boost=0.4,
        fibro_shield=0.3,
        fibro_energy_boost=0.1,
        feas_eps=1e-6,
        eq_tol=1e-4,
        use_gradient=True,
        mutation_prob=None,
        archive_size=100,
        seed=None,
    ):
        self.obj_func = objective_function
        self.n_dim = n_dim
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim == 1:
            self.bounds = np.tile(self.bounds, (n_dim, 1))

        self.n_obj = n_obj
        self.constraints = constraints or []
        self.eq_constraints = eq_constraints or []
        self.is_integer = np.asarray(is_integer if is_integer is not None else [False] * n_dim, dtype=bool)

        self.n_cancer = n_cancer
        self.n_lymph = n_lymph
        self.e_init = e_init
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.growth_rate = growth_rate
        self.niche_capacity = niche_capacity
        self.niche_radius = niche_radius
        self.crossover_rate = crossover_rate
        self.blx_alpha = blx_alpha
        self.p_meta0 = p_meta0
        self.levy_scale = levy_scale
        self.levy_beta = levy_beta
        self.e_maintenance = e_maintenance
        self.e_gain = e_gain
        self.e_comp = e_comp
        self.chemo_step = chemo_step
        self.chemo_beta = chemo_beta
        self.chemo_fd_eps = chemo_fd_eps
        self.duel_radius = duel_radius
        self.duel_k = duel_k
        self.use_fibroblasts = use_fibroblasts
        self.fibro_k = fibro_k
        self.fibro_beta_boost = fibro_beta_boost
        self.fibro_shield = fibro_shield
        self.fibro_energy_boost = fibro_energy_boost
        self.feas_eps = feas_eps
        self.eq_tol = eq_tol
        # False -> la quimiotaxis no calcula gradientes (útil con decodificadores constantes a trozos)
        self.use_gradient = use_gradient
        # None -> mutar todos los genes (original); p in (0,1] -> cada gen muta con probabilidad p
        self.mutation_prob = mutation_prob
        # Tamaño máximo del archivo élite de Pareto (solo n_obj > 1). 0 lo desactiva.
        self.archive_size = archive_size

        self.rng = np.random.default_rng(seed)
        self.fibro_phi = {}
        self.archive = []

    def _repair(self, x):
        """Proyecta x a los límites [bounds] y redondea variables enteras."""
        x_clamped = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        if np.any(self.is_integer):
            x_clamped[self.is_integer] = np.round(x_clamped[self.is_integer])
        return x_clamped

    def _constraint_violation(self, x):
        """Calcula la suma de violaciones de restricciones g(x) <= 0 y |h(x)| <= tol."""
        total_cv = 0.0
        for g in self.constraints:
            val = g(x)
            if val > 0:
                total_cv += val
        for h in self.eq_constraints:
            val = abs(h(x))
            if val > self.eq_tol:
                total_cv += val
        return total_cv

    def _evaluate_objectives(self, x):
        """Evalúa la función objetivo devolviendo un arreglo 1D numpy."""
        raw = self.obj_func(x)
        # Las funciones tipo benchmark devuelven (objetivos, restricciones); aquí solo interesan los objetivos
        objs = raw[0] if isinstance(raw, tuple) else raw

        objs_arr = np.atleast_1d(np.asarray(objs, dtype=float))
        return objs_arr

    def evaluate(self, pop):
        """Evalúa restricciones, vectores de objetivos y asigna rangos NSGA-II."""
        for ind in pop:
            ind.cv = self._constraint_violation(ind.x)
            ind.obj = self._evaluate_objectives(ind.x)
            if ind.weights is None and self.n_obj > 1:
                ind.weights = self.rng.dirichlet(np.ones(self.n_obj))
            elif self.n_obj == 1:
                ind.weights = np.array([1.0])

        assign_rank_and_crowding(pop, self.n_obj, feas_eps=self.feas_eps)

    def _phi_of(self, ind):
        """Devuelve el nivel de activación de fibroblastos para el nicho del individuo."""
        if not self.use_fibroblasts or ind.niche_id == -1:
            return 0.0
        return self.fibro_phi.get(ind.niche_id, 0.0)

    def _init_population(self, size):
        """Inicializa una subpoblación con posiciones aleatorias uniformes."""
        pop = []
        for _ in range(size):
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            x = self._repair(x)
            w = self.rng.dirichlet(np.ones(self.n_obj)) if self.n_obj > 1 else np.array([1.0])
            pop.append(Individual(x, self.sigma_init, self.e_init, weights=w))
        return pop

    def _update_archive(self, pop):
        """Mantiene un archivo élite con los individuos factibles no dominados vistos hasta ahora."""
        if self.n_obj < 2 or self.archive_size <= 0:
            return
        candidates = list(self.archive)
        for ind in pop:
            if ind.cv <= self.feas_eps:
                c = Individual(ind.x.copy(), ind.sigma, ind.energy, weights=ind.weights)
                c.obj = ind.obj.copy()
                c.cv = ind.cv
                candidates.append(c)
        if not candidates:
            return

        seen, unique = set(), []
        for c in candidates:
            key = tuple(np.round(c.obj, 9))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        front = fast_non_dominated_sort(unique, feas_eps=self.feas_eps)[0]
        if len(front) > self.archive_size:
            crowd = crowding_distance(front, self.n_obj)
            front = sorted(front, key=lambda i: crowd[id(i)], reverse=True)[: self.archive_size]
        self.archive = front

    def pareto_front(self):
        """Devuelve el archivo élite: individuos factibles no dominados (solo n_obj > 1)."""
        return list(self.archive)

    def run(self, n_generations=100, verbose=True):
        """Ejecuta el ciclo co-evolutivo de CancerGA."""
        self.archive = []
        cancer_pop = self._init_population(self.n_cancer)
        lymph_pop = self._init_population(self.n_lymph)

        self.evaluate(cancer_pop)
        self.evaluate(lymph_pop)

        history = {"best_fitness": [], "cancer_size": [], "lymph_size": []}

        for gen in range(n_generations):
            # 1. Microambiente tumoral (fibroblastos)
            if self.use_fibroblasts:
                self.fibro_phi = compute_fibroblast_field(cancer_pop, self)

            # 2. Mutación adaptativa
            adaptive_mutation(cancer_pop, self)
            adaptive_mutation(lymph_pop, self)

            # 3. Quimiotaxis guiada
            chemotaxis(cancer_pop, self)
            chemotaxis(lymph_pop, self)

            # 4. Evaluación de la población mutada/desplazada
            self.evaluate(cancer_pop)
            self.evaluate(lymph_pop)

            # 5. Duelo biológico
            cancer_pop, lymph_pop = biological_duel(cancer_pop, lymph_pop, self)

            # 6. Cruce
            cancer_pop = crossover(cancer_pop, self)
            # Los hijos nacen sin fitness: evaluar antes de ordenar por nichos (si no, sorted() falla con None)
            self.evaluate(cancer_pop)

            # 7. Proliferación por nichos
            cancer_pop = niche_proliferation(cancer_pop, self.n_cancer, self)

            # 8. Metástasis
            metastasis(cancer_pop, self)

            # 9. Evaluación previa a la inanición
            self.evaluate(cancer_pop)
            self.evaluate(lymph_pop)

            # 10. Inanición y limpieza energética
            cancer_pop = starvation(cancer_pop, self)
            lymph_pop = starvation(lymph_pop, self)

            # Reabastecimiento mínimo si alguna población colapsa
            if len(cancer_pop) < 5:
                cancer_pop.extend(self._init_population(self.n_cancer - len(cancer_pop)))
            if len(lymph_pop) < 2:
                lymph_pop.extend(self._init_population(self.n_lymph - len(lymph_pop)))

            self.evaluate(cancer_pop)
            self.evaluate(lymph_pop)

            # Archivo élite de Pareto (multiobjetivo)
            self._update_archive(cancer_pop)
            self._update_archive(lymph_pop)

            # Registro de historia
            feasible_c = [ind for ind in cancer_pop if ind.cv <= self.feas_eps]
            if feasible_c:
                best_ind = max(feasible_c, key=lambda i: i.fitness)
                best_val = best_ind.obj[0] if self.n_obj == 1 else best_ind.fitness
            else:
                best_ind = min(cancer_pop, key=lambda i: i.cv)
                best_val = -best_ind.cv

            history["best_fitness"].append(best_val)
            history["cancer_size"].append(len(cancer_pop))
            history["lymph_size"].append(len(lymph_pop))

            if verbose and (gen == 0 or (gen + 1) % 10 == 0 or gen == n_generations - 1):
                status = "Factible" if feasible_c else "Infactible"
                extra = f" | Pareto: {len(self.archive):3d}" if self.n_obj > 1 else ""
                print(
                    f"Gen {gen + 1:3d}/{n_generations} | Cáncer: {len(cancer_pop):3d} | "
                    f"Linf: {len(lymph_pop):3d} | Valor: {best_val:.6f} ({status}){extra}"
                )

        self.history = history
        feasible_c = [ind for ind in cancer_pop if ind.cv <= self.feas_eps]
        if feasible_c:
            best_ind = max(feasible_c, key=lambda i: i.fitness if self.n_obj > 1 else -i.obj[0])
            return best_ind.x, best_ind.obj[0] if self.n_obj == 1 else best_ind.obj, True
        elif cancer_pop:
            best_ind = min(cancer_pop, key=lambda i: i.cv)
            return best_ind.x, best_ind.obj[0] if self.n_obj == 1 else best_ind.obj, False

        return None
