# eqsolver/core/system_equation.py
from typing import List, Optional
import sympy as sp
from sympy import Function
from .equation import Equation, Condition


class SystemEquation:
    def __init__(
        self,
        equations: List[Equation],
        dep_vars: List[Function],
        conditions: Optional[List[Condition]] = None,
    ):
        self.equations = equations
        self.dep_vars = dep_vars
        self.conditions = conditions or []

        # Verificar que todas tengan las mismas variables independientes
        if equations:
            vars0 = equations[0].var
            for eq in equations[1:]:
                if eq.var != vars0:
                    raise ValueError("Distintas variables independientes.")
            self.var = vars0

        # Determinar variable temporal (la primera que aparezca en derivadas)
        self.time_var = None
        for eq in equations:
            tv = eq.get_time_variable()
            if tv is not None:
                self.time_var = tv ; break

        # Reunir todas las condiciones (globales + las de cada ecuación)
        self.all_conditions = self.conditions[:]
        for eq in equations: self.all_conditions.extend(eq.conditions)

    def __len__(self): return len(self.equations)

    def __repr__(self):
        return f"SystemEquation({len(self.equations)} equations, vars={self.var})"