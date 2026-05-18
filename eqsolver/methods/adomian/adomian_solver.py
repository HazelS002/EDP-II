"""
Implementación del método de descomposición de Adomian (ADM) para EDOs con condiciones iniciales.
"""

import sympy as sp
from typing import List, Optional
from sympy import Function, Symbol, Derivative, Integral, exp, oo, Eq

from ...core.equation import Equation, Condition
from ...core.solver_base import Solver
from ...utils.symbolic_helpers import inverse_operator
from .adomian_polynomials import AdomianPolynomialsCalculator


class AdomianMethod(Solver):
    """
    Resuelve ecuaciones del tipo L(u) + R(u) + N(u) = g mediante ADM,
    donde L es un operador diferencial lineal invertible (generalmente el de mayor orden),
    R es el resto lineal, N es no lineal, y g es el término fuente.
    Condiciones iniciales: se proporcionan en Equation.conditions.
    """

    def __init__(self, n_terms: int = 5, simplify: bool = True):
        """
        n_terms: número de términos de la serie a calcular (u0, u1, ..., u_{n_terms-1})
        simplify: si es True, simplifica las expresiones simbólicas en cada paso.
        """
        self.n_terms = n_terms
        self.simplify = simplify

    def solve(self, equation: Equation, **kwargs) -> sp.Expr:
        """
        Retorna la solución aproximada como suma de u0 + u1 + ... + u_{n_terms-1}
        en forma simbólica.
        """
        # Extraer componentes
        L = equation.L
        R = equation.R
        N = equation.N
        g = equation.g
        var = equation.var
        dep_var = equation.dep_var
        conditions = equation.conditions

        # Asumimos que es una EDO con una sola variable independiente
        if len(var) != 1:
            raise NotImplementedError("Por ahora solo se soportan EDOs (una variable independiente).")
        t = var[0]  # variable independiente

        # Determinar el orden de L (derivada más alta)
        # Para simplificar, suponemos que L es de la forma d^m/dt^m (operador principal)
        # Buscamos la derivada de mayor orden en L
        order = equation.order()
        if order == 0:
            raise ValueError("No se pudo determinar el orden del operador L.")

        # Construir la inversa de L con condiciones iniciales
        # Necesitamos condiciones para la función y sus derivadas hasta order-1 en el punto inicial.
        # Extraemos las condiciones iniciales (is_initial=True) y las ordenamos por orden de derivada.
        init_conds = [c for c in conditions if c.is_initial]
        # Verificar que haya al menos una condición (punto base)
        if not init_conds:
            # Si no hay, asumimos punto base = 0 y condiciones cero (solución particular)
            point = sp.S(0)
            # Creamos condiciones dummy para la integración
            # En realidad, necesitamos las constantes que surgen de la integración múltiple.
            # Usaremos un enfoque: integrales indefinidas + constantes a determinar.
            # Por simplicidad, implementaremos el caso con condiciones en t0.
            raise ValueError("Se requieren condiciones iniciales para la EDO.")
        else:
            # Tomamos el punto base de la primera condición (todas deberían ser iguales)
            point = init_conds[0].at_point

        # Definimos la función que calcula L^{-1} (integral múltiple desde point)
        def L_inverse(expr):
            # Integrar expr desde point hasta t, order veces
            res = expr
            for _ in range(order):
                res = sp.Integral(res, (t, point, t))
            return res

        # Construcción de u0: u0 = phi + L^{-1}(g) - L^{-1}(R(u0))? No, la recursión canónica es:
        # u0 = phi + L^{-1}(g)
        # u_{m+1} = - L^{-1}(R(u_m)) - L^{-1}(A_m)
        # donde phi es la solución de L(phi)=0 que satisface las condiciones iniciales.
        # phi es un polinomio en t de grado order-1 cuyos coeficientes se determinan con las condiciones.
        # Calculamos phi: es la solución homogénea con las condiciones dadas.
        # Construimos phi como combinación lineal de potencias (t-point)^k, k=0..order-1
        # y resolvemos los coeficientes con las condiciones.
        # Condiciones: para cada derivada j (0 <= j < order), se tiene que la j-ésima derivada de u en t=point
        # es algún valor dado por las condiciones iniciales.
        # Extraemos los valores de las condiciones:
        init_vals = {}
        for cond in init_conds:
            # Determinar el orden de derivada: si cond.var es dep_var (función), orden 0.
            # Si cond.var es Derivative, extraer orden.
            # Por simplicidad, asumimos que el usuario pasó condiciones como "u(point)=a", "u'(point)=b", etc.
            # Vamos a detectar: si cond.var es exactamente dep_var, orden 0.
            # Si cond.var es Derivative(dep_var, t, n), orden n.
            if cond.var == dep_var:
                order_deriv = 0
            elif isinstance(cond.var, Derivative) and cond.var.expr == dep_var and cond.var.variables[0] == t:
                order_deriv = len(cond.var.variables)
            else:
                raise ValueError(f"No se pudo interpretar la condición {cond}")
            init_vals[order_deriv] = cond.value

        # Aseguramos que tenemos valores para 0..order-1
        for k in range(order):
            if k not in init_vals:
                init_vals[k] = sp.S(0)  # por defecto cero

        # Construimos phi como suma_{k=0}^{order-1} C_k * (t - point)^k
        C = sp.symbols(f'C0:{order}')
        phi = sum(C[k] * (t - point)**k for k in range(order))
        # Las derivadas de phi en t=point dan: phi^{(j)}(point) = j! * C_j
        eqs = []
        for j in range(order):
            deriv_j = phi.diff(t, j).subs(t, point)
            eqs.append(sp.Eq(deriv_j, init_vals[j]))
        # Resolver para C
        sol = sp.solve(eqs, C)
        phi = phi.subs(sol)

        # Ahora calculamos u0 = phi + L^{-1}(g)
        u0 = phi + L_inverse(g)
        if self.simplify:
            u0 = sp.simplify(u0)

        # Inicializar lista de componentes
        u_components = [u0]

        # Precalcular L^{-1} aplicada a R y a los polinomios de Adomian
        # La recursión: u_{m+1} = - L^{-1}(R(u_m)) - L^{-1}(A_m)
        # Nota: R y N pueden depender de t y de u, y posiblemente de derivadas de u.
        # En la formulación estándar, R es lineal en u (y sus derivadas), N es no lineal.
        # Para aplicar L^{-1} a expresiones que contienen u_m, debemos sustituir.
        # Por simplicidad, asumimos que R(u) es lineal y que N(u) es una función no lineal de u únicamente
        # (no de derivadas). Eso es común en ADM.

        # Verificar que N depende solo de dep_var, no de derivadas
        # (se podría extender después)

        for m in range(1, self.n_terms):
            # Calcular R(u_{m-1})
            # R es una expresión en dep_var y sus derivadas. Sustituimos dep_var por u_components[m-1]
            R_um = R.subs(dep_var, u_components[m-1])
            # Para derivadas, necesitamos derivar u_components[m-1] correctamente.
            # Si R contiene derivadas como Derivative(dep_var, t), debemos convertirlas en Derivative(u_comp, t)
            # Hacemos una sustitución más inteligente: reemplazar dep_var por u_comp y luego evaluar derivadas
            # Usamos el método .subs no es suficiente para derivadas. Mejor usar la función `derivative_sub` personalizada.
            # Creamos una función auxiliar.
            def replace_dep_var(expr, u_expr):
                # Recorremos el árbol de expresión reemplazando dep_var por u_expr,
                # y Derivative(dep_var, t, k) por Derivative(u_expr, t, k)
                if expr == dep_var:
                    return u_expr
                elif isinstance(expr, Derivative) and expr.expr == dep_var:
                    new_expr = u_expr
                    for var in expr.variables:
                        new_expr = sp.Derivative(new_expr, var)
                    return new_expr
                else:
                    # Recursión sobre argumentos
                    return expr.func(*[replace_dep_var(arg, u_expr) for arg in expr.args])
            R_um = replace_dep_var(R, u_components[m-1])
            # Calcular A_{m-1} (polinomio de Adomian para índice m-1)
            A_m1 = AdomianPolynomialsCalculator.compute(N, u_components, m-1, dep_var)
            # Aplicar L^{-1}
            term1 = L_inverse(R_um)
            term2 = L_inverse(A_m1)
            u_m = - term1 - term2
            if self.simplify:
                u_m = sp.simplify(u_m)
            u_components.append(u_m)

        # Sumar todos los componentes
        approx_solution = sum(u_components)
        if self.simplify:
            approx_solution = sp.simplify(approx_solution)

        return approx_solution