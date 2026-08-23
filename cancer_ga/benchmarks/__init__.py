from .single_objective import (
    sphere,
    rastrigin,
    rosenbrock,
    ackley,
    constrained_mixed_example,
)
from .ctp import CTP1, CTP2, CTP3, CTP4, CTP5, CTP6, CTP7, CTP8
from .cf import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10
from .mw import get_mw_problem

__all__ = [
    # Mono-objetivo
    "sphere",
    "rastrigin",
    "rosenbrock",
    "ackley",
    "constrained_mixed_example",
    # CTP
    "CTP1",
    "CTP2",
    "CTP3",
    "CTP4",
    "CTP5",
    "CTP6",
    "CTP7",
    "CTP8",
    # CF
    "CF1",
    "CF2",
    "CF3",
    "CF4",
    "CF5",
    "CF6",
    "CF7",
    "CF8",
    "CF9",
    "CF10",
    # MW
    "get_mw_problem",
]