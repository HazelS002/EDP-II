"""
Representación de una ecuación diferencial ordinaria o parcial.
"""

from typing import List, Tuple, Union, Optional, Any
import sympy as sp
from sympy import Function, Symbol, Derivative, Expr


class Condition:
    """Condición inicial o de contorno."""
    def __init__(self, var: Symbol, value: Expr, at_point: Expr, is_initial: bool = True):
        """
        var: variable independiente (ej. t)
        value: valor de la función (o derivada) en at_point
        at_point: punto donde se aplica la condición
        is_initial: True si es condición inicial, False si es de contorno
        """
        self.var = var
        self.value = value
        self.at_point = at_point
        self.is_initial = is_initial

    def __repr__(self):
        return f"Condition({self.var}={self.value} @ {self.at_point}, initial={self.is_initial})"


class Equation:
    """
    Representa una ecuación diferencial de la forma:
        L(u) + R(u) + N(u) = g
    donde:
        L : operador lineal invertible (principal, de mayor orden)
        R : resto lineal (operadores de orden inferior)
        N : término no lineal (función de u y sus derivadas)
        g : término fuente (función de las variables independientes)

    También almacena condiciones y las variables involucradas.
    """

    def __init__(
        self,
        L: Expr,
        R: Expr,
        N: Expr,
        g: Expr,
        var: List[Symbol],
        dep_var: Function,
        conditions: Optional[List[Condition]] = None,
    ):
        """
        L, R, N, g: expresiones sympy que representan los operadores.
        var: lista de variables independientes (ej. [t] para EDO, [t, x] para EDP)
        dep_var: función dependiente, ej. Function('u')(t, x)
        conditions: lista de condiciones (iniciales o de contorno)
        """
        self.L = L
        self.R = R
        self.N = N
        self.g = g
        self.var = var
        self.dep_var = dep_var
        self.conditions = conditions or []

        # Verificación básica de que dep_var aparece en los operadores
        # (se puede mejorar)
        # self._validate()

    @classmethod
    def from_expression(
        cls,
        expr: Expr,
        var: List[Symbol],
        dep_var: Function,
        conditions: Optional[List[Condition]] = None,
    ):
        """
        Constructor alternativo que acepta una expresión completa del tipo:
            expr = 0
        donde expr contiene L(u)+R(u)+N(u)-g.
        Intenta descomponer automáticamente los términos lineales y no lineales.
        Este método es básico y puede fallar para casos complejos; se recomienda
        usar el constructor directo con componentes separados.
        """
        # Por simplicidad, esta implementación asume que expr ya está expandida
        # y que todo término que no sea lineal en dep_var y sus derivadas se considera no lineal.
        # En la práctica, se necesita un análisis más cuidadoso.
        raise NotImplementedError(
            "La descomposición automática no está implementada aún. "
            "Use el constructor explícito con L, R, N, g."
        )

    def order(self) -> int:
        """Retorna el orden de la ecuación (mayor orden de derivada en L)."""
        # Buscar la derivada de mayor orden en L (asumiendo que L contiene el operador principal)
        from sympy import Derivative

        def max_order(expr):
            orders = []
            for arg in sp.preorder_traversal(expr):
                if isinstance(arg, Derivative) and arg.expr == self.dep_var:
                    orders.append(len(arg.variables))
            return max(orders) if orders else 0

        return max_order(self.L)

    def __repr__(self):
        return f"Equation(L={self.L}, R={self.R}, N={self.N}, g={self.g}, var={self.var}, dep_var={self.dep_var})"