import sympy as sp
from sympy import Function, Symbol, Derivative
from typing import List

from ...core.equation import Equation, Condition
from ...core.solver_base import Solver
from ...utils import inverse_operator, get_base_point
from .adomian_polynomials import AdomianPolynomialsCalculator


class AdomianMethod(Solver):
    """
    Método de descomposición de Adomian para resolver EDOs/EDPs de evolución.
    """

    def __init__(self, n_terms: int = 5, simplify: bool = True):
        """
        n_terms: número de términos de la serie (u0 + u1 + ... + u_{n_terms-1})
        simplify: si True, simplifica expresiones simbólicas
        """
        self.n_terms = n_terms
        self.simplify = simplify

    def _replace_dep_var(self, expr, u_expr, dep_var, time_var):
        """Reemplaza la función dependiente y sus derivadas por u_expr."""
        if expr == dep_var: return u_expr

        if isinstance(expr, Derivative):
            if expr.expr == dep_var:
                new_expr = u_expr
                for v in expr.variables: new_expr = sp.Derivative(new_expr, v)
                return new_expr
            else:
                return expr.func(*[self._replace_dep_var(arg, u_expr, dep_var,\
                                                time_var) for arg in expr.args])
        if hasattr(expr, 'args') and expr.args:
            return expr.func(*[self._replace_dep_var(arg, u_expr, dep_var,\
                                            time_var) for arg in expr.args])
        return expr

    def solve(self, equation: Equation, **kwargs) -> sp.Expr:
        # Obtener componentes de la ecuación
        L = equation.L
        R = equation.R
        N = equation.N
        g = equation.g
        variables = equation.var
        dep_var = equation.dep_var
        conditions = equation.conditions

        if N == 0: N = sp.S(0)
        if g == 0: g = sp.S(0)

        # Determinar variable temporal y orden de L
        time_var = None
        order = 0
        for arg in sp.preorder_traversal(L):
            if isinstance(arg, Derivative) and arg.expr == dep_var:
                time_var = arg.variables[0]
                order = len(arg.variables)
                break
        if time_var is None:
            raise ValueError("No se encontró derivada temporal en L.")

        # Punto base
        point = get_base_point(conditions, default=0)
        if isinstance(point, dict): point = point.get(time_var, 0)

        # Inversa de L
        L_inverse = inverse_operator(L, dep_var, variables, order,
                                     conditions, time_var)

        # Extraer valores iniciales
        init_vals = {}
        for cond in conditions:
            if cond.is_initial:
                if cond.var == dep_var:
                    dorder = 0
                elif isinstance(cond.var, Derivative) and cond.var.expr\
                    == dep_var:
                    dorder = len(cond.var.variables)
                else:
                    continue
                init_vals[dorder] = cond.value

        # Construir solución homogénea phi con condiciones
        spatial_vars = [v for v in variables if v != time_var]
        C_symbols = []
        for k in range(order):
            if spatial_vars: Ck = sp.Function(f'C{k}')(*spatial_vars)
            else: Ck = sp.Symbol(f'C{k}')
            C_symbols.append(Ck)
        phi = sum(C_symbols[k] * (time_var - point)**k for k in range(order))
        eqs = []
        for k in range(order):
            val = init_vals.get(k, sp.S(0))
            lhs = sp.factorial(k) * C_symbols[k]
            rhs = val
            eqs.append(sp.Eq(lhs, rhs))
        sol = sp.solve(eqs, C_symbols)
        phi = phi.subs(sol)

        # u0
        u0 = phi + L_inverse(g)
        if self.simplify: u0 = sp.simplify(u0)
        u_components = [u0]

        # Recursión
        for m in range(1, self.n_terms):
            R_um = self._replace_dep_var(R,u_components[m-1],dep_var,time_var)
            if N != sp.S(0):
                A_m1 = AdomianPolynomialsCalculator.compute(N, u_components,
                                                             m-1, dep_var)
            else:
                A_m1 = sp.S(0)
            term1 = L_inverse(R_um)
            term2 = L_inverse(A_m1)
            u_m = -term1 - term2

            if self.simplify: u_m = sp.simplify(u_m)
            u_components.append(u_m)

        # Suma final
        approx = sum(u_components)
        if self.simplify: approx = sp.simplify(approx)
        return approx