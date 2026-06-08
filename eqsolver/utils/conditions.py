"""
Manejo de condiciones iniciales y de contorno.
"""

import sympy as sp
from sympy import Symbol, Expr, Derivative, Function, Eq, solve
from typing import List, Dict, Union
from ..core.equation import Condition


def extract_initial_conditions(conditions: List[Condition], var: Symbol)\
    -> Dict[int, Expr]:
    """
    Extrae las condiciones iniciales en un diccionario: {orden_derivada: valor}.
    Asume que todas las condiciones están en el mismo punto.
    """
    init_dict = {}
    for cond in conditions:
        if not cond.is_initial:
            continue
        # Determinar el orden de derivada
        if cond.var == cond.var.func:  # es la función (sin derivar)
            order = 0
        elif isinstance(cond.var, Derivative)\
            and cond.var.expr == cond.var.expr.func:
            # Caso típico: Derivative(u, t)
            order = len(cond.var.variables)
        else:
            # Si es otra cosa, intentar buscar
            order = 0
        init_dict[order] = cond.value
    return init_dict


def build_homogeneous_solution(order: int, var: Symbol, point: Expr, init_vals:
                               Dict[int, Expr]) -> Expr:
    """
    Construye la solución homogénea phi = sum_{k=0}^{order-1} C_k * (var -
    point)^k y determina las constantes C_k usando init_vals (diccionario
    orden->valor).
    """
    C = sp.symbols(f'C0:{order}')
    phi = sum(C[k] * (var - point)**k for k in range(order))
    eqs = []
    for k in range(order):
        val = init_vals.get(k, sp.S(0))
        # La derivada k-ésima de phi en var=point es k! * C_k
        lhs = sp.factorial(k) * C[k]
        rhs = val
        eqs.append(Eq(lhs, rhs))
    sol = solve(eqs, C)
    return phi.subs(sol)


def get_base_point(conditions: List[Condition], default=0)\
    -> Union[Expr, Dict[Symbol, Expr]]:
    """Obtiene el punto base común de las condiciones iniciales."""
    for cond in conditions:
        if cond.is_initial: return cond.at_point
    return default


def apply_initial_conditions(expr: Expr, var: Symbol, conditions:\
                             List[Condition]) -> Expr:
    """
    Dada una expresión que puede contener constantes indeterminadas
    (C1, C2, ...), aplica las condiciones iniciales para determinar las
    constantes.
    Útil cuando se tienen integrales indefinidas.
    """
    # Recolectar símbolos que empiecen con 'C' (constantes)
    free_syms = [s for s in expr.free_symbols if str(s).startswith('C')]
    if not free_syms: return expr

    # Construir sistema de ecuaciones
    eqs = []
    for cond in conditions:
        if not cond.is_initial: continue

        # Determinar orden de derivada
        if cond.var == cond.var.func:
            deriv_order = 0
            expr_eval = expr.subs(var, cond.at_point)
        elif isinstance(cond.var, Derivative)\
            and cond.var.expr == cond.var.expr.func:
            deriv_order = len(cond.var.variables)
            expr_eval = expr.diff(var, deriv_order).subs(var, cond.at_point)
        else:
            continue
        eqs.append(Eq(expr_eval, cond.value))

    if not eqs: return expr

    sol = solve(eqs, free_syms)
    return expr.subs(sol)