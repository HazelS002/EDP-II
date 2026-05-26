# eqsolver/methods/adomian/adomian_system_solver.py
import sympy as sp
from sympy import Function, Symbol, Derivative, Expr
from typing import List, Dict
from ...core.system_equation import SystemEquation
from ...core.solver_base import Solver
from ...utils import inverse_operator, get_base_point
from .adomian_polynomials_system import AdomianPolynomialsSystem


class AdomianSystemSolver(Solver):
    def __init__(self, n_terms: int = 5, simplify: bool = True):
        self.n_terms = n_terms
        self.simplify = simplify

    def _replace_dep_vars(self, expr: Expr, u_map: Dict[Function, Expr],
                          time_var: Symbol) -> Expr:
        if isinstance(expr, Function) and expr in u_map: return u_map[expr]

        if isinstance(expr, Derivative):
            if expr.expr in u_map:
                new_base = u_map[expr.expr]
                for var in expr.variables:
                    new_base = sp.Derivative(new_base, var)
                return new_base
            else:
                return expr.func(*[self._replace_dep_vars(arg, u_map, time_var)\
                                   for arg in expr.args])
        if hasattr(expr, 'args') and expr.args:
            return expr.func(*[self._replace_dep_vars(arg, u_map, time_var)\
                               for arg in expr.args])
        return expr

    def solve(self, system: SystemEquation, **kwargs) -> List[Expr]:
        eqs = system.equations
        dep_vars = system.dep_vars
        var = system.var
        time_var = system.time_var
        if time_var is None:
            raise ValueError("No se pudo determinar la variable temporal.")
        conditions = system.all_conditions

        orders = [eq.get_order() for eq in eqs]
        if len(set(orders)) != 1:
            raise ValueError("Las ecuaciones deben tener mismo orden temporal.")
        order = orders[0]

        point = get_base_point(conditions, default=0)
        if isinstance(point, dict): point = point.get(time_var, 0)

        # Tomamos la primera ecuación para construir L_inverse (asumimos que L es d^order/dt^order)
        L0 = eqs[0].L
        L_inverse = inverse_operator(L0, dep_vars[0], var, order,
                                     conditions, time_var)

        # Extraer condiciones iniciales por variable
        init_vals = {dv: {} for dv in dep_vars}
        for cond in conditions:
            if not cond.is_initial: continue

            for dv in dep_vars:
                if cond.var == dv:
                    init_vals[dv][0] = cond.value
                    break
                elif isinstance(cond.var, Derivative) and cond.var.expr == dv:
                    dorder = len(cond.var.variables)
                    init_vals[dv][dorder] = cond.value
                    break

        # Construir phi para cada variable (solución homogénea)
        spatial_vars = [v for v in var if v != time_var]
        phi_dict = {}
        for dv in dep_vars:
            C_syms = []
            for k in range(order):
                Ck = sp.Function(f'C_{dv.name}_{k}')(*spatial_vars)\
                    if spatial_vars else sp.Symbol(f'C_{dv.name}_{k}')
                C_syms.append(Ck)

            phi = sum(C_syms[k] * (time_var - point)**k for k in range(order))
            eqs_cond = []
            for k in range(order):
                val = init_vals.get(dv, {}).get(k, sp.S(0))
                lhs = sp.factorial(k) * C_syms[k]
                eqs_cond.append(sp.Eq(lhs, val))
            sol = sp.solve(eqs_cond, C_syms)
            phi = phi.subs(sol)
            phi_dict[dv] = phi

        # Inicializar componentes
        components = {dv: [] for dv in dep_vars}
        for idx, eq in enumerate(eqs):
            dv = dep_vars[idx]
            u0 = phi_dict[dv] + L_inverse(eq.g)
            if self.simplify: u0 = sp.simplify(u0)
            components[dv].append(u0)

        # Recursión
        for m in range(1, self.n_terms):
            u_next = {}
            for idx, eq in enumerate(eqs):
                dv = dep_vars[idx]
                # Mapa de sustitución para R: usamos u_{m-1} de cada variable
                subst_map = {dvi: comps[m-1] for dvi, comps\
                             in components.items() if m-1 < len(comps)}
                R_um = self._replace_dep_vars(eq.R, subst_map, time_var)

                # Polinomio de Adomian para N
                comps_dict = {dvi: comps_i for dvi, comps_i\
                              in components.items()}
                if eq.N != sp.S(0):
                    A = AdomianPolynomialsSystem.compute(eq.N, comps_dict,\
                                                         m-1, dep_vars) 
                else:
                    A = sp.S(0)

                term1 = L_inverse(R_um)
                term2 = L_inverse(A)
                u_m = -term1 - term2
                if self.simplify: u_m = sp.simplify(u_m)
                u_next[dv] = u_m

            for dv, u_m in u_next.items(): components[dv].append(u_m)

        # Soluciones finales
        solutions = []
        for dv in dep_vars:
            sol = sum(components[dv])
            if self.simplify: sol = sp.simplify(sol)
            solutions.append(sol)
        return solutions