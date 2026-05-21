"""
Cálculo de polinomios de Adomian para una función no lineal N(u) y para sistemas.
"""

import sympy as sp
from sympy import Function, Symbol, Expr, Derivative
from typing import List, Dict


class AdomianPolynomialsCalculator:
    """Calculador de polinomios de Adomian para una sola ecuación."""

    @staticmethod
    def compute(N_expr: Expr, u_components: List[Expr], n: int, dep_var: Function) -> Expr:
        # (código ya existente, sin cambios)
        if N_expr == 0 or N_expr == sp.Integer(0):
            return sp.S(0)
        if not N_expr.has(dep_var):
            return N_expr if n == 0 else sp.S(0)
        λ = sp.Symbol('λ')
        u_series = sum(uk * λ**k for k, uk in enumerate(u_components[:n+1]))
        N_series = N_expr.subs(dep_var, u_series)
        deriv = sp.diff(N_series, λ, n)
        An = (deriv / sp.factorial(n)).subs(λ, 0)
        return sp.simplify(An)

    @staticmethod
    def compute_sequence(N_expr: Expr, u_components: List[Expr], max_n: int, dep_var: Function) -> List[Expr]:
        return [AdomianPolynomialsCalculator.compute(N_expr, u_components, n, dep_var) for n in range(max_n+1)]

    # Nuevo método para sistemas
    @staticmethod
    def compute_for_system(
        N_expr: Expr,
        components_dict: Dict[Function, List[Expr]],
        n: int,
        dep_vars: List[Function]
    ) -> Expr:
        """
        Calcula el polinomio de Adomian A_n para una expresión no lineal N_expr
        que puede depender de múltiples funciones dependientes.
        
        Parámetros:
        N_expr: expresión sympy que involucra las funciones en dep_vars.
        components_dict: diccionario {dep_var: [u0, u1, ..., un]} para cada variable.
        n: índice del polinomio.
        dep_vars: lista de funciones dependientes (orden relevante).
        
        Retorna A_n como expresión sympy.
        """
        if N_expr == 0 or N_expr == sp.Integer(0):
            return sp.S(0)
        
        # Verificar si N_expr depende de alguna de las variables
        depends = any(N_expr.has(var) for var in dep_vars)
        if not depends:
            # Si no depende de ninguna, A_0 = N_expr, A_n=0 para n>0
            return N_expr if n == 0 else sp.S(0)
        
        λ = sp.Symbol('λ')
        # Sustituir cada función dependiente por su serie en λ
        subs_dict = {}
        for var in dep_vars:
            comps = components_dict.get(var, [])
            # Construir serie truncada hasta n: sum_{k=0}^n u_k λ^k
            series = sum(comps[k] * λ**k for k in range(min(n+1, len(comps))))
            subs_dict[var] = series
        N_series = N_expr.subs(subs_dict)
        deriv = sp.diff(N_series, λ, n)
        An = (deriv / sp.factorial(n)).subs(λ, 0)
        return sp.simplify(An)