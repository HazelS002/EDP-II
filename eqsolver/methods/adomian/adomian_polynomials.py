"""
Cálculo de polinomios de Adomian para una función no lineal N(u).
"""

import sympy as sp
from sympy import Function, Symbol, Expr, Derivative
from typing import List


class AdomianPolynomialsCalculator:
    """
    Calcula los polinomios de Adomian A_n para una función no lineal N(u).
    """

    @staticmethod
    def compute(N_expr: Expr, u_components: List[Expr], n: int, dep_var: Function) -> Expr:
        """
        N_expr: expresión simbólica que depende de dep_var (ej. u**2, sin(u), etc.)
        u_components: lista de [u0, u1, ..., un] (expresiones simbólicas)
        n: índice del polinomio A_n que se desea calcular
        dep_var: función dependiente

        Retorna A_n como expresión sympy.
        """
        # Si N_expr es 0 (cero constante), todos los polinomios son cero
        if N_expr == 0 or N_expr == sp.Integer(0): return sp.S(0)
        
        # Si N_expr no depende de dep_var, entonces A_0 = N_expr, A_n = 0 para n>0
        if not N_expr.has(dep_var): return N_expr if n == 0 else sp.S(0)
        
        # Creamos una variable l simbólica
        l = sp.Symbol('l')
        # Construimos la serie truncada: u0 + u1*l + u2*l^2 + ... + un*l^n
        u_series = sum(uk * l**k for k, uk in enumerate(u_components[:n+1]))
        # Sustituimos dep_var por la serie en N_expr
        N_series = N_expr.subs(dep_var, u_series)
        # Derivamos n veces respecto a l
        deriv = sp.diff(N_series, l, n)
        # Evaluamos en l=0 y dividimos por n!
        An = (deriv / sp.factorial(n)).subs(l, 0)
        # Simplificamos
        return sp.simplify(An)

    @staticmethod
    def compute_sequence(N_expr: Expr, u_components: List[Expr], max_n: int, dep_var: Function) -> List[Expr]:
        """
        Calcula la lista de polinomios A_0, A_1, ..., A_{max_n}.
        """
        return [AdomianPolynomialsCalculator.compute(N_expr, u_components, n, dep_var) for n in range(max_n+1)]