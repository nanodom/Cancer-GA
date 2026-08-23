class Individual:
    """Representa una célula (cancerosa o linfocito)."""

    __slots__ = (
        "x", "sigma", "energy", "fitness", "niche_id", 
        "obj", "rank", "crowd", "weights", "cv"
    )

    def __init__(self, x, sigma, energy, weights=None):
        self.x = x                # Posición (genotipo) en R^n
        self.sigma = sigma        # Paso de mutación auto-adaptativo
        self.energy = energy      # Reserva de energía
        self.fitness = None       # Score compuesto
        self.niche_id = -1        # Nicho asignado
        self.obj = None           # Vector de objetivos
        self.rank = None          # Rango de no-dominancia
        self.crowd = None         # Distancia de aglomeración
        self.weights = weights    # Pesos de escalarización
        self.cv = 0.0             # Violación total de restricciones