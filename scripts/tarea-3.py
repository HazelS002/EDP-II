from heateq.findiffs import *
from heateq.conditions import *
from heateq.utils import create_mesh

from visualization.one_dimensional_time_eqs.complete_plot import show_solution

from dolfinx import fem
import ufl
from ns_cylinder import *


def ns_cylinder():
    # Parámetros
    T = 5.0
    num_steps = 5000
    dt = T / num_steps
    mu = 0.001
    rho = 1.0

    # Generar malla
    domain, facet_tags = create_mesh()

    # Espacios de funciones (Taylor-Hood P2-P1)
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    V = fem.FunctionSpace(domain, P2)
    Q = fem.FunctionSpace(domain, P1)

    # Condiciones de contorno
    bcu, bcp = define_boundary_conditions(domain, V, Q, facet_tags)

    # Formas variacionales
    forms, init_funcs = create_variational_forms(V, Q, dt, mu, rho)

    # Bucle temporal
    time_loop(domain, V, Q, bcu, bcp, forms, init_funcs, dt, num_steps,
              output_dir="navier_stokes_cylinder")
    return

if __name__ == "__main__":
    # # Condiciones de dirichlet
    # A, u0 = dirichletCondition()
    # linear_frames = solveEq(A, u0)
    # X, T, U = create_mesh(linear_frames)
    # show_solution(X, T, U, "Heat Equation - Dirichlet Conditions")

    
    # # Condiciones de Neumann
    # A, u0 = neumannCondition()
    # linear_frames = solveEq(A, u0)
    # X, T, U = create_mesh(linear_frames)
    # show_solution(X, T, U, "Heat Equation - Neumann Conditions")


    # # Condiciones periodicas
    # A, u0 = periodicCondition()
    # linear_frames = solvePeriodicEq(A, u0)
    # X, T, U = create_mesh(linear_frames)
    # show_solution(X, T, U, "Heat Equation - Periodic Conditions")

    ns_cylinder()
