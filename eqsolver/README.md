# eqsolver

Módulo de Python para la resolución simbólica de ecuaciones diferenciales
ordinarias (EDOs) y parciales (EDPs) mediante el
**método de descomposición de Adomian (ADM)**. Actualmente soporta problemas de
valor inicial (condiciones iniciales en el tiempo) y sistemas de ecuaciones.

---

## Características principales

- **Simbólico**: utiliza `sympy` para obtener expresiones exactas en forma de 
  series.
- **Descomposición explícita**: el usuario debe proporcionar la ecuación en la forma  
  `L(u) + R(u) + N(u) = g`, donde:
  - `L` es el operador lineal invertible (típicamente la derivada temporal de mayor orden).
  - `R` es el resto lineal (términos lineales de orden inferior).
  - `N` es el término no lineal (puede incluir derivadas).
  - `g` es el término fuente.
- **Manejo de EDOs y EDPs**: soporta una variable independiente (EDO) o varias
  (EDP), siempre que la evolución sea temporal.
- **Condiciones iniciales**: se especifican en un punto base
  (generalmente `t=0`).
- **Sistemas de ecuaciones**: resuelve sistemas acoplados mediante
  `AdomianSystemSolver`.

---

## Estructura del módulo

```
eqsolver/
├── core/
│   ├── equation.py         # Clases Equation y Condition
│   ├── solver_base.py      # Clase abstracta Solver
│   └── system_equation.py  # Clase SystemEquation (para sistemas)
├── methods/
│   └── adomian/
│       ├── adomian_solver.py             # Adomian para una ecuación
│       ├── adomian_system_solver.py      # Adomian para sistemas
│       ├── adomian_polynomials.py        # Polinomios de Adomian (una variable)
│       └── adomian_polynomials_system.py # Polinomios para sistemas
├── utils/
│   ├── symbolic_helpers.py   # Inversa de operadores, integración con condiciones
│   └── conditions.py         # Manejo de condiciones iniciales
├── __init__.py 
└── README.md
```

---

## Base teórica del método de Adomian

El método de descomposición de Adomian (ADM) busca una solución en serie de la forma  
`u = u_0 + u_1 + u_2 + ...` para una ecuación diferencial escrita como:

```
L(u) + R(u) + N(u) = g
```

donde:
- `L` es un operador lineal invertible (normalmente la derivada de mayor orden, `d^m/dt^m`).
- `R` es el operador lineal restante.
- `N` es un operador no lineal.
- `g` es una función conocida.

### 1. Operador inverso
Se define el operador inverso `L^{-1}`, que es la integral múltiple (con las condiciones iniciales).  
Por ejemplo, si `L = d/dt`, entonces `L^{-1}(f) = ∫_{t_0}^t f(s) ds`.

### 2. Descomposición de la solución y de la no linealidad
Se supone:
```
u = Σ_{n=0}^∞ u_n
```
y la no linealidad se expande en polinomios de Adomian:
```
N(u) = Σ_{n=0}^∞ A_n(u_0, u_1, ..., u_n)
```
Los polinomios se calculan mediante la fórmula:
```
A_n = 1/n! · d^n/dλ^n N( Σ_{k=0}^∞ u_k λ^k ) |_{λ=0}
```

### 3. Recursión
Aplicando `L^{-1}` a ambos lados de la ecuación se obtiene:
```
u = φ + L^{-1}(g) - L^{-1}(R(u)) - L^{-1}(N(u))
```
donde `φ` es la solución de la ecuación homogénea `L(u)=0` que satisface las condiciones iniciales.  
La recursión canónica es:
```
u_0 = φ + L^{-1}(g)
u_{m+1} = - L^{-1}(R(u_m)) - L^{-1}(A_m),   m ≥ 0
```
La solución aproximada se toma como la suma parcial `Σ_{n=0}^{M} u_n`.

### 4. Convergencia
Para problemas con operadores lineales acotados y no linealidades analíticas, la serie converge en un intervalo (generalmente pequeño). En la práctica, con unos pocos términos se obtienen buenas aproximaciones para valores de la variable no muy alejados del punto inicial.


## Ejemplos de uso

### Ejemplo 1: EDO lineal de primer orden

```python
from eqsolver import Equation, Condition, AdomianMethod
import sympy as sp

t = sp.Symbol('t')
u = sp.Function('u')(t)

# u' = u, u(0)=1
L = u.diff(t)
R = -u
N = 0
g = 0
conditions = [Condition(var=u, value=1, at_point=0, is_initial=True)]

eq = Equation(L, R, N, g, [t], u, conditions)
solver = AdomianMethod(n_terms=5)
sol = solver.solve(eq)
print(sol)   # 1 + t + t**2/2 + t**3/6 + t**4/24
```

### Ejemplo 2: EDO no lineal `u' = u^2`, `u(0)=1`

```python
L = u.diff(t)
R = 0
N = -u**2      # porque u' - u^2 = 0
g = 0
conditions = [Condition(u, 1, 0)]
eq = Equation(L, R, N, g, [t], u, conditions)
solver = AdomianMethod(n_terms=4)
sol = solver.solve(eq)
print(sol)   # 1 + t + t**2 + t**3 + ...
```

### Ejemplo 3: EDP de calor lineal

```python
t, x = sp.symbols('t x')
u = sp.Function('u')(t, x)

# u_t - u_xx = 0, u(0,x)=cos(x)
L = u.diff(t)
R = -u.diff(x, x)
N = 0
g = 0
conditions = [Condition(u, sp.cos(x), {t:0, x:x}, is_initial=True)]

eq = Equation(L, R, N, g, [t, x], u, conditions)
solver = AdomianMethod(n_terms=4)
sol = solver.solve(eq)
print(sol)   # (1 - t + t**2/2 - t**3/6)*cos(x)
```

### Ejemplo 4: Sistema de EDOs

```python
from eqsolver import Equation, Condition, SystemEquation, AdomianSystemSolver

t = sp.Symbol('t')
u = sp.Function('u')(t)
v = sp.Function('v')(t)

# u'' + v = 0, v'' + u = 0
# condiciones: u(0)=0, u'(0)=1, v(0)=0, v'(0)=-1
L1 = u.diff(t, t); N1 = v; eq1 = Equation(L1, 0, N1, 0, [t], u, [])
L2 = v.diff(t, t); N2 = u; eq2 = Equation(L2, 0, N2, 0, [t], v, [])

conditions = [
    Condition(u, 0, 0), Condition(u.diff(t), 1, 0),
    Condition(v, 0, 0), Condition(v.diff(t), -1, 0)
]

system = SystemEquation([eq1, eq2], [u, v], conditions)
solver = AdomianSystemSolver(n_terms=6)
sol_u, sol_v = solver.solve(system)
print(sol_u)   # serie de sinh(t)-sin(t)
```

---

## Limitaciones actuales

- **Condiciones de contorno**: solo se admiten condiciones **iniciales** (en `t=t0`). No se soportan condiciones de contorno espaciales.
- **Operador L**: debe ser una derivada pura respecto al tiempo (`d^m/dt^m`). No se permiten operadores más generales.
- **No linealidades**: se soportan, pero la implementación actual puede ser lenta para órdenes altos.
- **Convergencia**: el método es de tipo serie, válido en un entorno del punto inicial. Para valores alejados pueden ser necesarios muchos términos.
- **Sistemas**: la implementación es experimental y asume que todas las ecuaciones tienen el mismo orden temporal.

---

## Trabajo futuro

- Implementación de condiciones de contorno mediante funciones de Green y traslado.
- Soporte para EDPs con condiciones de contorno espaciales (por ejemplo, mediante transformadas de Fourier).
- Mejora de la eficiencia simbólica (simplificación selectiva, evitando explosión de términos).
- Integración con cálculo numérico para series largas.

---

## Contribución

Este módulo es de código abierto. Las contribuciones son bienvenidas.

---
