"""
Aplicación del método de Adomian a varios problemas de EDOs y EDPs.
Uso del módulo eqsolver desarrollado.
"""

N_termns = 10

def print_title(title: str) -> None:
    print("\n\n\n" + "="*80 + "\n" + title + "\n" + "="*80)
    

def equation_a():
    print_title("(a) EDO: y' = x^2 + y^2, y(0)=0")
    
    x = sp.Symbol('x')
    y = sp.Function('y')(x)
    
    L = y.diff(x)
    R = sp.S(0)
    N = -y**2
    g = x**2
    
    conditions = [Condition(var=y, value=0, at_point=0, is_initial=True)]
    eq = Equation(L, R, N, g, [x], y, conditions)
    
    solver = AdomianMethod(n_terms=N_termns, simplify=True)
    sol = solver.solve(eq)
    print(f"Solución ADM ({N_termns} términos):")
    sp.pprint(sol)
    return sol
    

def equation_b():
    print_title("(b) EDP: u_t = x^2 - (u_x)^2 / 4, u(x,0)=0")

    t, x = sp.symbols('t x')
    u = sp.Function('u')(t, x)
    
    L = u.diff(t)
    R = sp.S(0)
    N = (1/4)*u.diff(x)**2
    g = x**2

    conditions = [Condition(var=u, value=0, at_point={t:0, x:x}, is_initial=True)]
    eq = Equation(L, R, N, g, [t, x], u, conditions)
    
    solver = AdomianMethod(n_terms=N_termns, simplify=True)
    sol = solver.solve(eq)
    print(f"Solución ADM (primeros {N_termns} términos en t):")
    sp.pprint(sol)
    return sol

def equation_c():
    print_title("(c) EDP: u_t + u^2 u_x = 0, u(x,0)=3x")
    
    t, x = sp.symbols('t x')
    u = sp.Function('u')(t, x)
    
    L = u.diff(t)
    R = sp.S(0)
    N = u**2 * u.diff(x)
    g = sp.S(0)
    
    conditions = [Condition(var=u, value=3*x, at_point={t:0, x:x}, is_initial=True)]
    eq = Equation(L, R, N, g, [t, x], u, conditions)
    
    solver = AdomianMethod(n_terms=N_termns, simplify=True)
    sol = solver.solve(eq)
    print(f"Solución ADM (primeros {N_termns} términos en t):")
    sp.pprint(sol)
    return sol

def system_equations_d():
    print_title("(d) EDO's syatem")
    t = sp.symbols('t')
    u = sp.Function('u')(t)
    v = sp.Function('v')(t)

    # Ecuación 1: u'' + v = 0
    L1 = u.diff(t, t)
    R1 = sp.S(0)
    N1 = v
    g1 = sp.S(0)

    # Ecuación 2: v'' + u = 0
    L2 = v.diff(t, t)
    R2 = sp.S(0)
    N2 = u
    g2 = sp.S(0)

    # Condiciones iniciales globales
    conditions = [
        Condition(var=u, value=sp.S(0), at_point=sp.S(0), is_initial=True),
        Condition(var=u.diff(t), value=sp.S(1), at_point=sp.S(0), is_initial=True),
        Condition(var=v, value=sp.S(0), at_point=sp.S(0), is_initial=True),
        Condition(var=v.diff(t), value=sp.S(-1), at_point=sp.S(0), is_initial=True),
    ]

    # Crear las ecuaciones individuales (sin condiciones internas)
    eq1 = Equation(L1, R1, N1, g1, [t], u, conditions=[])
    eq2 = Equation(L2, R2, N2, g2, [t], v, conditions=[])

    # Sistema
    system = SystemEquation(equations=[eq1, eq2], dep_vars=[u, v], conditions=conditions)

    # Resolver con Adomian (6 términos para buena precisión)
    solver = AdomianSystemSolver(n_terms=N_termns, simplify=True)
    sol_u, sol_v = solver.solve(system)

    print("Solución aproximada para u(t):")
    sp.pprint(sol_u)
    print("\nSolución aproximada para v(t):")
    sp.pprint(sol_v)

    return sol_u, sol_v


def equation_e():
    print_title("(e) EDP: u_{tt}=u_{xx}")
    x, t = sp.symbols('x t')
    n_termns_fourier = 4

    # Condiciones iniciales expandidas en series de Fourier
    # sin^3(x) = (3/4) sin(x) - (1/4) sin(3x)
    # sin(2x) ya está en la base
    a = {1: sp.Rational(3,4), 3: sp.Rational(-1,4)}   # coeficientes para u(x,0)
    b = {2: 1}                                         # coeficientes para u_t(x,0)
    
    u_approx = 0
    # Para cada modo n que tenga coeficiente no nulo hasta n_max
    for n in range(1, n_termns_fourier+1):
        an = a.get(n, 0)
        bn = b.get(n, 0)
        if an == 0 and bn == 0: continue
        
        # Variable dependiente T_n(t)
        T = sp.Function(f'T{n}')(t)
        # Ecuación: T'' + n^2 T = 0
        L = T.diff(t, t)
        R = (n**2) * T
        N = 0
        g = 0

        # Condiciones iniciales
        conds = [
            Condition(var=T, value=an, at_point=0, is_initial=True),
            Condition(var=T.diff(t), value=bn, at_point=0, is_initial=True)
        ]

        eq = Equation(L, R, N, g, [t], T, conds)
        solver = AdomianMethod(n_terms=N_termns, simplify=True)
        T_series = solver.solve(eq)
        # Añadir el modo a la solución
        u_approx += T_series * sp.sin(n*x)

    u_approx = sp.sympify(u_approx)
    sp.pprint(u_approx)
    
    return u_approx
    
if __name__ == "__main__":
    from eqsolver import Equation, SystemEquation, Condition,\
        AdomianMethod, AdomianSystemSolver
    import sympy as sp

    equation_a()
    equation_b()
    equation_c()
    system_equations_d()
    equation_e()