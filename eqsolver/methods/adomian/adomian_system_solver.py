"""
Método de descomposición de Adomian para sistemas de EDOs/EDPs de evolución.
"""

import sympy as sp
from sympy import Function, Symbol, Derivative, Expr
from typing import List, Dict

from ...core.system_equation import SystemEquation
from ...core.equation import Condition
from ...core.solver_base import Solver
from ...utils import inverse_operator, get_base_point, build_homogeneous_solution
from .adomian_polynomials import AdomianPolynomialsCalculator


class AdomianSystemSolver(Solver):
    """
    Resuelve un sistema de ecuaciones de la forma:
        L(u_i) + R_i(u) + N_i(u) = g_i, i=1..m
    donde L es el mismo operador diferencial temporal para todos.
    """

    def __init__(self, n_terms: int = 5, simplify: bool = True):
        self.n_terms = n_terms
        self.simplify = simplify

    def _replace_dep_vars(self, expr, u_map: Dict[Function, Expr], time_var: Symbol):
        """
        Reemplaza las funciones dependientes y sus derivadas en expr
        por las expresiones dadas en u_map (que son aproximaciones u_comp).
        u_map: {dep_var: expresión sustituta}
        """
        if expr in u_map:
            return u_map[expr]
        if isinstance(expr, Derivative):
            # Verificar si la base es una función dependiente
            if expr.expr in u_map:
                base = u_map[expr.expr]
                # Derivar base respecto a las mismas variables
                new_expr = base
                for var in expr.variables:
                    new_expr = sp.Derivative(new_expr, var)
                return new_expr
            else:
                # Recursión en argumentos
                return expr.func(*[self._replace_dep_vars(arg, u_map, time_var) for arg in expr.args])
        if hasattr(expr, 'args') and expr.args:
            return expr.func(*[self._replace_dep_vars(arg, u_map, time_var) for arg in expr.args])
        return expr

    def solve(self, system: SystemEquation, **kwargs) -> Dict[Function, sp.Expr]:
        """
        Resuelve el sistema y retorna un diccionario {dep_var: solución_aproximada}.
        """
        equations = system.equations
        dep_vars = system.dep_vars
        conditions = system.all_conditions

        # Tomamos la primera ecuación para obtener información común (L, variables, orden)
        first_eq = equations[0]
        L = first_eq.L
        variables = first_eq.var
        time_var = system.get_time_variable()
        if time_var is None:
            raise ValueError("No se pudo determinar la variable temporal en el sistema.")

        order = system.get_order()

        # Punto base
        point = get_base_point(conditions, default=0)
        if isinstance(point, dict):
            point = point.get(time_var, 0)

        # Inversa de L
        L_inverse = inverse_operator(L, dep_vars[0], variables, order, conditions, time_var)

        # Extraer condiciones iniciales por variable
        init_vals_dict = {var: {} for var in dep_vars}
        for cond in conditions:
            if not cond.is_initial:
                continue
            # Determinar a qué variable pertenece la condición
            for var in dep_vars:
                if cond.var == var:
                    dorder = 0
                    init_vals_dict[var][dorder] = cond.value
                    break
                elif isinstance(cond.var, Derivative) and cond.var.expr == var:
                    dorder = len(cond.var.variables)
                    init_vals_dict[var][dorder] = cond.value
                    break

        # Construir phi para cada variable (solución homogénea)
        spatial_vars = [v for v in variables if v != time_var]
        phi_dict = {}
        for var in dep_vars:
            # Coeficientes pueden ser funciones de las variables espaciales
            C_symbols = []
            for k in range(order):
                if spatial_vars:
                    Ck = sp.Function(f'C_{var.name}_{k}')(*spatial_vars)
                else:
                    Ck = sp.Symbol(f'C_{var.name}_{k}')
                C_symbols.append(Ck)
            phi = sum(C_symbols[k] * (time_var - point)**k for k in range(order))
            # Imponer condiciones
            init_vals = init_vals_dict.get(var, {})
            eqs = []
            for k in range(order):
                val = init_vals.get(k, sp.S(0))
                lhs = sp.factorial(k) * C_symbols[k]
                rhs = val
                eqs.append(sp.Eq(lhs, rhs))
            sol = sp.solve(eqs, C_symbols)
            phi_dict[var] = phi.subs(sol)

        # Inicializar componentes para cada variable
        u_components = {var: [] for var in dep_vars}

        # u_i0 = phi_i + L^{-1}(g_i)
        for eq, var in zip(equations, dep_vars):
            u0 = phi_dict[var] + L_inverse(eq.g)
            if self.simplify:
                u0 = sp.simplify(u0)
            u_components[var].append(u0)

        # Recursión
        for m in range(1, self.n_terms):
            for idx, (eq, var) in enumerate(zip(equations, dep_vars)):
                # Calcular R_i(u_{m-1}) para esta variable
                # Necesitamos un mapa de sustitución para todas las variables,
                # usando el componente (m-1)-ésimo de cada una.
                subs_map = {}
                for v in dep_vars:
                    if len(u_components[v]) > m-1:
                        subs_map[v] = u_components[v][m-1]
                    else:
                        subs_map[v] = sp.S(0)  # si no hay componente, cero
                R_um = self._replace_dep_vars(eq.R, subs_map, time_var)

                # Calcular polinomio de Adomian A_{m-1} para N_i
                # Necesitamos los componentes hasta m-1 para todas las variables
                comps_dict = {v: u_components[v] for v in dep_vars}
                A_m1 = AdomianPolynomialsCalculator.compute_for_system(
                    eq.N, comps_dict, m-1, dep_vars
                )

                term1 = L_inverse(R_um)
                term2 = L_inverse(A_m1)
                u_m = -term1 - term2
                if self.simplify:
                    u_m = sp.simplify(u_m)
                u_components[var].append(u_m)

        # Construir soluciones aproximadas sumando componentes
        solutions = {}
        for var in dep_vars:
            solutions[var] = sum(u_components[var])
            if self.simplify:
                solutions[var] = sp.simplify(solutions[var])
        return solutions