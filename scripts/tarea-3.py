from heateq.findiffs import *
from heateq.conditions import *
from heateq.utils import create_mesh

from visualization.one_dimensional_time_eqs.complete_plot import show_solution

if __name__ == "__main__":
    # Condiciones de dirichlet
    A, u0 = dirichletCondition()
    linear_frames = solveEq(A, u0)
    X, T, U = create_mesh(linear_frames)
    show_solution(X, T, U, "Heat Equation - Dirichlet Conditions")

    
    # Condiciones de Neumann
    A, u0 = neumannCondition()
    linear_frames = solveEq(A, u0)
    X, T, U = create_mesh(linear_frames)
    show_solution(X, T, U, "Heat Equation - Neumann Conditions")


    # Condiciones periodicas
    A, u0 = periodicCondition()
    linear_frames = solvePeriodicEq(A, u0)
    X, T, U = create_mesh(linear_frames)
    show_solution(X, T, U, "Heat Equation - Periodic Conditions")
