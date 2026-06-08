"""
Clase base abstracta para todos los solvers.
"""

from abc import ABC, abstractmethod
from typing import Any
from .equation import Equation


class Solver(ABC):
    """Interfaz común para cualquier método de resolución de EDOs/EDPs."""

    @abstractmethod
    def solve(self, equation: Equation, **kwargs) -> Any:
        """
        Resuelve la ecuación diferencial y retorna la solución (simbólica,
        numérica, serie, etc.)
        equation: instancia de Equation
        kwargs: parámetros específicos del método (número de términos,
        tolerancia, etc.)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"