from heateq.findiffs import *
from heateq.conditions import *
from heateq.utils import create_mesh
from visualization import show_solution

from ns_cylinder import solve_simulation
from ns_cylinder.config import nsc_default_output_dirname, meshfilename,\
    useriesfilename

from reaction_system import solve_reaction_simulation
from reaction_system.config import rs_default_output_dirname

def ns_cylinder():
    T = 5.0
    num_steps = 5000
    mu = 0.001
    rho = 1

    solve_simulation(T, num_steps, mu, rho, output_dir=\
                     nsc_default_output_dirname)    

def reaction_system():
    T = 5.0
    num_steps = 500
    eps = 0.01
    K = 10.0

    solve_reaction_simulation(T, num_steps, eps, K, nsc_default_output_dirname,
                              meshfilename, useriesfilename, output_dir=\
                                rs_default_output_dirname)


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

    reaction_system()
