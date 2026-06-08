import ufl
from dolfinx import fem, default_scalar_type

def epsilon(u): return ufl.sym(ufl.grad(u))

def sigma(u, p, mu): return 2.0 * mu * epsilon(u) - p * ufl.Identity(len(u))

def create_variational_forms(V, Q, dt, mu, rho):
    """Ensambla las formas de los tres pasos del esquema de proyección.

    Returns:
        a1, L1, a2, L2, a3, L3: Formas de UFL.
        u_n, p_n: Funciones de prueba iniciales.
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    # Funciones para almacenar la solución en el paso anterior
    u_n = fem.Function(V)
    p_n = fem.Function(Q)
    u_ = fem.Function(V)
    p_ = fem.Function(Q)

    n = ufl.FacetNormal(V.mesh)
    f = ufl.Constant(V.mesh, default_scalar_type((0.0, 0.0)))
    k = ufl.Constant(V.mesh, dt)
    mu_c = ufl.Constant(V.mesh, mu)
    rho_c = ufl.Constant(V.mesh, rho)

    # Paso 1: velocidad tentativa
    U_avg = 0.5 * (u_n + u)
    F1 = (rho_c * ufl.inner((u - u_n) / k, v) * ufl.dx
          + rho_c * ufl.inner(ufl.dot(u_n, ufl.grad(u_n)), v) * ufl.dx
          + ufl.inner(sigma(U_avg, p_n, mu_c), ufl.grad(v)) * ufl.dx
          + ufl.dot(p_n * n, v) * ufl.ds
          - ufl.dot(mu_c * ufl.grad(U_avg) * n, v) * ufl.ds
          - ufl.inner(f, v) * ufl.dx)
    a1 = ufl.lhs(F1)
    L1 = ufl.rhs(F1)

    # Paso 2: corrección de presión
    a2 = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
    L2 = (ufl.inner(ufl.grad(p_n), ufl.grad(q)) * ufl.dx
          - (1.0 / k) * ufl.div(u_) * q * ufl.dx)

    # Paso 3: corrección de velocidad
    a3 = ufl.inner(u, v) * ufl.dx
    L3 = (ufl.inner(u_, v) * ufl.dx
          - k * ufl.inner(ufl.grad(p_ - p_n), v) * ufl.dx)

    return (a1, L1, a2, L2, a3, L3), (u_n, p_n, u_, p_)