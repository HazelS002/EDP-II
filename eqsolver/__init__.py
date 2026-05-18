"""
eqsolver - Solución simbólica de ecuaciones diferenciales mediante diversos métodos.
Actualmente implementa el método de descomposición de Adomian (ADM) para EDOs.
"""

from .core.equation import Equation, Condition
from .core.solver_base import Solver
from .methods.adomian.adomian_solver import AdomianMethod
from .methods.adomian.adomian_polynomials import AdomianPolynomialsCalculator

__all__ = [
    "Equation",
    "Condition",
    "Solver",
    "AdomianMethod",
    "AdomianPolynomialsCalculator",
]