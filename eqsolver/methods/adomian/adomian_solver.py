"""
Implementación del método de descomposición de Adomian (ADM) para EDOs con condiciones iniciales.
"""

import sympy as sp
from typing import List
from sympy import Function, Symbol, Derivative, Integral

from ...core.equation import Equation, Condition
from ...core.solver_base import Solver
from .adomian_polynomials import AdomianPolynomialsCalculator


class AdomianMethod(Solver):
    """
    Resuelve ecuaciones del tipo L(u) + R(u) + N(u) = g mediante ADM.
    """

    def __init__(self, n_terms: int = 5, simplify: bool = True):
        self.n_terms = n_terms
        self.simplify = simplify

    def _replace_dep_var(self, expr, u_expr, dep_var, t):
        """
        Reemplaza dep_var y sus derivadas por u_expr y sus derivadas.
        """
        if expr == dep_var:
            return u_expr
        elif isinstance(expr, Derivative):
            # Verificar si es derivada de dep_var
            if expr.expr == dep_var:
                new_expr = u_expr
                for var in expr.variables:
                    new_expr = sp.Derivative(new_expr, var)
                return new_expr
            else:
                # Derivada de otra cosa, recursión
                return expr.func(*[self._replace_dep_var(arg, u_expr, dep_var, t) for arg in expr.args])
        elif hasattr(expr, 'args') and expr.args:
            # Recursión sobre argumentos
            return expr.func(*[self._replace_dep_var(arg, u_expr, dep_var, t) for arg in expr.args])
        else:
            return expr

    def _apply_conditions_to_phi(self, order, t, point, conditions):
        """
        Construye la solución homogénea phi que satisface las condiciones iniciales.
        """
        # Extraer valores de condiciones
        init_vals = {}
        for cond in conditions:
            if not cond.is_initial:
                continue
            
            # Determinar el orden de derivada
            if cond.var == cond.var.func:
                # Es la función misma
                order_deriv = 0
            elif isinstance(cond.var, Derivative):
                # Es una derivada
                order_deriv = len(cond.var.variables)
            else:
                # Intentar extraer de otra forma
                order_deriv = 0
            
            init_vals[order_deriv] = cond.value

        # Asegurar valores por defecto
        for k in range(order):
            if k not in init_vals:
                init_vals[k] = sp.S(0)

        # Construir phi como polinomio
        C = sp.symbols(f'C0:{order}')
        phi = sum(C[k] * (t - point)**k for k in range(order))
        
        # Crear ecuaciones para las condiciones
        eqs = []
        for j in range(order):
            deriv_j = phi.diff(t, j).subs(t, point)
            eqs.append(sp.Eq(deriv_j, init_vals[j]))
        
        # Resolver y sustituir
        sol = sp.solve(eqs, C)
        phi = phi.subs(sol)
        
        return phi

    def solve(self, equation: Equation, **kwargs) -> sp.Expr:
        """
        Retorna la solución aproximada como suma de u0 + u1 + ... + u_{n_terms-1}
        """
        # Extraer componentes
        L = equation.L
        R = equation.R
        N = equation.N
        g = equation.g
        var = equation.var
        dep_var = equation.dep_var
        conditions = equation.conditions

        # Asegurar que N es una expresión SymPy
        if N == 0:
            N = sp.S(0)
        if g == 0:
            g = sp.S(0)

        # Verificar EDO
        if len(var) != 1:
            raise NotImplementedError("Por ahora solo EDOs con una variable independiente.")
        t = var[0]

        # Determinar orden de L
        def find_order(expr):
            orders = []
            for arg in sp.preorder_traversal(expr):
                if isinstance(arg, Derivative) and arg.expr == dep_var:
                    orders.append(len(arg.variables))
            return max(orders) if orders else 0

        order = find_order(L)
        if order == 0:
            # Intentar con L como derivada explícita
            if isinstance(L, Derivative) and L.expr == dep_var:
                order = len(L.variables)
            else:
                raise ValueError(f"No se pudo determinar el orden de L: {L}")

        # Extraer condiciones iniciales
        init_conds = [c for c in conditions if c.is_initial]
        if not init_conds:
            # Si no hay condiciones, asumir punto base = 0 y valores cero
            point = sp.S(0)
            # Crear condiciones por defecto
            init_conds = [Condition(var=dep_var, value=sp.S(0), at_point=point, is_initial=True)]
            for k in range(1, order):
                init_conds.append(Condition(var=sp.Derivative(dep_var, t, k), value=sp.S(0), at_point=point, is_initial=True))
        else:
            point = init_conds[0].at_point

        # Definir operador inverso L^{-1} (integración múltiple desde point)
        def L_inverse(expr):
            result = expr
            for _ in range(order):
                result = sp.Integral(result, (t, point, t))
            return result

        # Construir phi (solución homogénea con condiciones)
        phi = self._apply_conditions_to_phi(order, t, point, init_conds)
        
        # Calcular u0 = phi + L^{-1}(g)
        u0 = phi + L_inverse(g)
        if self.simplify:
            u0 = sp.simplify(u0)

        # Lista de componentes
        u_components = [u0]

        # Recursión de Adomian
        for m in range(1, self.n_terms):
            # Calcular R(u_{m-1})
            R_um = self._replace_dep_var(R, u_components[m-1], dep_var, t)
            
            # Calcular A_{m-1} (polinomio de Adomian)
            if N != sp.S(0):
                A_m1 = AdomianPolynomialsCalculator.compute(N, u_components, m-1, dep_var)
            else:
                A_m1 = sp.S(0)
            
            # Aplicar L^{-1}
            term1 = L_inverse(R_um)
            term2 = L_inverse(A_m1)
            
            u_m = -term1 - term2
            if self.simplify:
                u_m = sp.simplify(u_m)
            
            u_components.append(u_m)

        # Sumar todos los componentes
        approx_solution = sum(u_components)
        if self.simplify:
            approx_solution = sp.simplify(approx_solution)

        return approx_solution