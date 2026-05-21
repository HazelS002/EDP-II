"""
Aplicación del método de Adomian a varios problemas de EDOs y EDPs.
Uso del módulo eqsolver desarrollado.
"""

import sympy as sp
from eqsolver import Equation, Condition, AdomianMethod

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
    
if __name__ == "__main__":
    equation_a()
    equation_b()
    equation_c()