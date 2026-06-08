"""
Funciones auxiliares para manipulación simbólica: integrales definidas,
inversa de operadores, etc.
"""

import sympy as sp
from sympy import Function, Symbol, Derivative, Integral, Eq, solve
from typing import List, Union, Dict, Optional
from ..core.equation import Condition


def inverse_operator(L_expr: sp.Expr, dep_var: Function, variables:\
                     List[Symbol], order: int, conditions: List[Condition],\
                        time_var: Symbol = None):
    """
    Construye la inversa del operador L, asumiendo que L es d^m/dt^m (derivada
    pura respecto al tiempo).
    Para EDOs, variables es [t]; para EDPs, la inversa solo integra en tiempo.
    conditions: lista de Condition con is_initial=True, que definen el punto
    base t0.
    Retorna una función que aplica la inversa a una expresión f.
    """
    # Determinar variable temporal
    if time_var is None:
        if len(variables) == 1: time_var = variables[0]
        else:
            for arg in sp.preorder_traversal(L_expr):
                if isinstance(arg, Derivative) and arg.expr == dep_var:
                    time_var = arg.variables[0]
                    break
            if time_var is None:
                raise ValueError("No se pudo determinar la variable temporal.")

    # Extraer punto base de las condiciones iniciales
    init_conds = [c for c in conditions if c.is_initial]
    if init_conds:
        point = init_conds[0].at_point
        if isinstance(point, dict): point = point.get(time_var, 0)
    else:
        point = 0

    def L_inverse(expr):
        # Integrar expr desde point hasta time_var, order veces
        result = expr
        for _ in range(order):
            result = Integral(result, (time_var, point, time_var))
        return result

    return L_inverse


def integrate_with_conditions(expr: sp.Expr, var: Symbol, order: int,
                              conditions: List[Condition]) -> sp.Expr:
    """
    Integra expr order veces respecto a var, y luego determina las constantes
    de integración usando las condiciones dadas (deben ser condiciones iniciales
    en el punto base).
    Retorna la primitiva que satisface las condiciones.
    """
    # Realizar integrales indefinidas
    result = expr
    constants = []
    for i in range(order):
        result = Integral(result, var)
        Ci = sp.Symbol(f'C{i}')
        constants.append(Ci)
        result = result + Ci

    # Si no hay condiciones, devolver con constantes
    if not conditions: return result

    # Extraer punto base
    point = conditions[0].at_point
    if isinstance(point, dict):
        # Para EDPs, necesitamos la variable temporal; asumimos que var es la temporal
        point = point.get(var, 0)

    # Construir ecuaciones
    eqs = []
    for cond in conditions:
        if not cond.is_initial: continue

        # Determinar orden de derivada
        if cond.var == cond.var.func:
            deriv_order = 0
            expr_eval = result.subs(var, point)
        elif isinstance(cond.var, Derivative) and cond.var.expr == cond.var.expr.func:
            deriv_order = len(cond.var.variables)
            expr_eval = result.diff(var, deriv_order).subs(var, point)
        else:
            continue
        eqs.append(Eq(expr_eval, cond.value))

    if eqs:
        sol = solve(eqs, constants)
        result = result.subs(sol)
    return result


def apply_initial_conditions(expr: sp.Expr, var: Symbol, conditions: List[Condition]) -> sp.Expr:
    """
    Dada una expresión que puede contener constantes indeterminadas (C1, C2, ...),
    aplica las condiciones iniciales para determinar las constantes.
    """
    free_syms = [s for s in expr.free_symbols if str(s).startswith('C')]
    if not free_syms: return expr

    eqs = []
    for cond in conditions:
        if not cond.is_initial:
            continue
        if cond.var == cond.var.func:
            deriv_order = 0
            expr_eval = expr.subs(var, cond.at_point)
        elif isinstance(cond.var, Derivative) and cond.var.expr == cond.var.expr.func:
            deriv_order = len(cond.var.variables)
            expr_eval = expr.diff(var, deriv_order).subs(var, cond.at_point)
        else:
            continue
        eqs.append(Eq(expr_eval, cond.value))

    if not eqs: return expr
    sol = solve(eqs, free_syms)
    return expr.subs(sol)