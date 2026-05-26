"""
Definición de la ecuación diferencial y sus condiciones.
"""

import sympy as sp
from sympy import Expr, Function, Symbol, Derivative
from typing import List, Union, Dict, Optional, Any


class Condition:
    """
    Representa una condición inicial o de contorno.
    
    Atributos:
        var: la función o derivada a la que se aplica la condición
        (ej. u, u.diff(t))
        value: valor de la condición (expresión simbólica)
        at_point: punto donde se aplica (número, símbolo o diccionario 
        {var: punto})
        is_initial: True si es condición inicial (en tiempo), False si es
        de contorno
    """
    def __init__(self, var: Any, value: Expr, at_point:\
                 Union[Expr, Dict[Symbol, Expr]], is_initial: bool = True):
        self.var = var
        self.value = value
        self.at_point = at_point
        self.is_initial = is_initial

    def __repr__(self):
        return f"Condition({self.var} = {self.value} @ {self.at_point}, initial={self.is_initial})"


class Equation:
    """
    Representa una ecuación diferencial de la forma:
        L(u) + R(u) + N(u) = g
    donde:
        L : operador lineal invertible (principal, típicamente derivada de
        mayor orden respecto al tiempo)
        R : resto lineal (operadores lineales de orden inferior)
        N : término no lineal (función de u y sus derivadas)
        g : término fuente (función de las variables independientes)
        var : lista de variables independientes (ej. [t] para EDO, [t, x]
        para EDP)
        dep_var : función dependiente (ej. Function('u')(t, x))
        conditions : lista de objetos Condition (iniciales y/o de contorno)

    El usuario debe proporcionar explícitamente la descomposición L, R, N, g.
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
        self.L = L
        self.R = R
        self.N = N
        self.g = g
        self.var = var
        self.dep_var = dep_var
        self.conditions = conditions or []

        # Conversión automática de ceros a SymPy Integer
        if self.L == 0: self.L = sp.S(0)
        if self.R == 0: self.R = sp.S(0)
        if self.N == 0: self.N = sp.S(0)
        if self.g == 0: self.g = sp.S(0)

        # Validación básica: L debe contener al menos una derivada de dep_var
        self._validate()

    def _validate(self):
        """Verifica que L contenga una derivada de dep_var
        (para poder invertir)."""
        has_derivative = False
        for arg in sp.preorder_traversal(self.L):
            if isinstance(arg, Derivative) and arg.expr == self.dep_var:
                has_derivative = True
                break
    
        if not has_derivative and self.L != 0:
            raise ValueError("L debe tener al menos una derivada")

    def get_order(self) -> int:
        """Retorna el orden de la ecuación (grado de la derivada
        más alta en L)."""
        max_order = 0
        for arg in sp.preorder_traversal(self.L):
            if isinstance(arg, Derivative) and arg.expr == self.dep_var:
                order = len(arg.variables)
                if order > max_order: max_order = order

        return max_order

    def get_time_variable(self) -> Optional[Symbol]:
        """
        Intenta determinar la variable temporal (aquella respecto a la cual se
        deriva en L). Si L tiene múltiples derivadas (caso EDP), se toma la
        primera derivada encontrada.
        Retorna None si no se encuentra.
        """
        for arg in sp.preorder_traversal(self.L):
            if isinstance(arg, Derivative) and arg.expr == self.dep_var:
                return arg.variables[0]
        return None

    def get_spatial_variables(self) -> List[Symbol]:
        """Retorna las variables independientes que no son la temporal."""
        t_var = self.get_time_variable()
        if t_var is None: return self.var[:]
        return [v for v in self.var if v != t_var]

    def is_ode(self) -> bool:
        """Retorna True si es una EDO (solo una variable independiente)."""
        return len(self.var) == 1

    def is_pde(self) -> bool:
        """Retorna True si es una EDP (más de una variable independiente)."""
        return len(self.var) > 1

    def __repr__(self):
        return (f"Equation(L={self.L}, R={self.R}, N={self.N}, g={self.g}, "
                f"var={self.var}, dep_var={self.dep_var}, conditions={self.conditions})")