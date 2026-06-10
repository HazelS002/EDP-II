from dolfin import TrialFunction, TestFunction,\
    Function, FacetNormal, Constant, dx
from dolfin import dot, nabla_grad, inner, ds, dx, lhs, rhs, div

from .helpers import epsilon, sigma


def assemble_forms(V, Q, dt, mu, rho, mesh):
    # Funciones y variables variacionales
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    u_n = Function(V)
    u_ = Function(V)
    p_n = Function(Q)
    p_ = Function(Q)

    # Expresiones auxiliares
    U = 0.5*(u_n + u)
    n = FacetNormal(mesh)
    f = Constant((0, 0))
    k = Constant(dt)
    mu_c = Constant(mu)
    rho_c = Constant(rho)


    # Paso 1: velocidad tentativa
    F1 = rho_c*dot((u - u_n)/k, v)*dx \
        + rho_c*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
        + inner(sigma(U, p_n, mu_c), epsilon(v))*dx \
        + dot(p_n*n, v)*ds - dot(mu_c*nabla_grad(U)*n, v)*ds \
        - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Paso 2: corrección de presión
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Paso 3: corrección de velocidad
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    return (a1, L1), (a2, L2), (a3, L3), (u, v), (p, q), (u_n, u_), (p_n, p_)