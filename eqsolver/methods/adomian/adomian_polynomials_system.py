# eqsolver/methods/adomian/adomian_polynomials_system.py
import sympy as sp
from sympy import Function, Expr
from typing import List, Dict


class AdomianPolynomialsSystem:
    @staticmethod
    def compute(
        N_expr: Expr,
        u_components_dict: Dict[Function, List[Expr]],
        n: int,
        dep_vars: List[Function],
        lambda_symbol=None
    ) -> Expr:
        if N_expr == 0 or N_expr == sp.Integer(0):
            return sp.S(0)
        if not any(N_expr.has(dv) for dv in dep_vars):
            return N_expr if n == 0 else sp.S(0)

        lam = lambda_symbol or sp.Symbol('λ')
        # Construir series para cada variable
        series_map = {}
        for dv, comps in u_components_dict.items():
            series = sum(uk * lam**k for k, uk in enumerate(comps[:n+1]))
            series_map[dv] = series

        # Sustituir
        N_series = N_expr
        for dv, ser in series_map.items():
            N_series = N_series.subs(dv, ser)

        deriv = sp.diff(N_series, lam, n)
        An = (deriv / sp.factorial(n)).subs(lam, 0)
        return sp.simplify(An)

    @staticmethod
    def compute_sequence(N_expr, u_components_dict, max_n, dep_vars):
        return [AdomianPolynomialsSystem.compute(N_expr, u_components_dict,\
                                    n, dep_vars) for n in range(max_n + 1)]