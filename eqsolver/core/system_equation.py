"""
Definición de un sistema de ecuaciones diferenciales.
"""

import sympy as sp
from typing import List, Optional
from .equation import Equation, Condition


class SystemEquation:
    """
    Representa un sistema de ecuaciones diferenciales de la forma:
        L(u_i) + R_i(u) + N_i(u) = g_i, para i=1..m
    donde u = (u_1, ..., u_m) es el vector de funciones dependientes.
    L es el mismo operador lineal invertible para todas las ecuaciones
    (generalmente ∂^p/∂t^p, derivada temporal pura).
    """

    def __init__(
        self,
        equations: List[Equation],
        dep_vars: List[sp.Function],
        conditions: Optional[List[Condition]] = None,
    ):
        """
        equations: lista de objetos Equation (cada uno con su L, R, N, g, pero L debe ser idéntico para todos)
        dep_vars: lista de funciones dependientes (ej. [u, v])
        conditions: lista de condiciones (iniciales) que pueden aplicarse a cualquier dep_var.
                    Es recomendable incluir las condiciones directamente en cada Equation,
                    pero este parámetro permite condiciones cruzadas.
        """
        self.equations = equations
        self.dep_vars = dep_vars
        self.conditions = conditions or []

        # Verificar que todas las ecuaciones tengan el mismo L (mismo operador)
        if len(equations) > 1:
            L0 = equations[0].L
            for eq in equations[1:]:
                if eq.L != L0:
                    raise ValueError("Todas las ecuaciones deben tener el mismo operador L.")

        # Reunir todas las condiciones de las ecuaciones
        self.all_conditions = self.conditions[:]
        for eq in equations:
            self.all_conditions.extend(eq.conditions)

    def get_time_variable(self):
        """Obtiene la variable temporal común (la primera derivada encontrada en L)."""
        if not self.equations:
            return None
        return self.equations[0].get_time_variable()

    def get_order(self):
        """Orden del operador L (común)."""
        if not self.equations:
            return 0
        return self.equations[0].get_order()

    def __repr__(self):
        return f"SystemEquation({len(self.equations)} equations, dep_vars={self.dep_vars})"