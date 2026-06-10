from heateq.findiffs import *
from heateq.conditions import *
from heateq.utils import create_mesh
from visualization import show_solution

from ns_cylinder import solve_simulation, visualize_simulation

def ns_cylinder():
    T = 5.0
    num_steps = 5000
    mu = 0.001
    rho = 1

    dir = "navier_stokes_cylinder"
    solve_simulation(T, num_steps, mu, rho, output_dir=dir)
    visualize_simulation(output_dir=dir)
    


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
