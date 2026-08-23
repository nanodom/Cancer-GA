def get_mw_problem(name, n_var=10):
    """Wrapper para importar dinámicamente las funciones del benchmark MW desde pymoo.
    
    Requiere pymoo instalado (pip install pymoo).
    """
    try:
        from pymoo.problems import get_problem
    except ImportError as err:
        raise ImportError(
            "Para usar la familia MW es necesario instalar pymoo: `pip install pymoo`"
        ) from err

    problem = get_problem(name.lower(), n_var=n_var)

    def mw_func(x):
        out = {}
        problem._evaluate(x, out)
        objs = out["F"]
        
        # Pymoo maneja las restricciones como G <= 0
        constraints = out.get("G", None)
        if constraints is not None:
            return objs, [float(g) for g in constraints]
        return objs, []

    return mw_func