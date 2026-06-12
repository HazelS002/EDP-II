import os
from dolfin import File, VectorFunctionSpace, FunctionSpace, plot
from matplotlib.pyplot import show

from heateq.findiffs import *
from heateq.conditions import *
from heateq.utils import create_mesh
from visualization import show_solution, show_simulation

from ns_cylinder import solve_simulation
from ns_cylinder.config import nsc_default_output_dirname, meshfilename,\
    useriesfilename, pseriesfilename, mesh_resolution
from ns_cylinder.mesh import get_mesh

from reaction_system import solve_reaction_simulation
from reaction_system.config import rs_default_output_dirname

def ns_cylinder():
    # parametros fisicos y numericos
    T = 5.0
    num_steps = 5000
    mu = 0.001
    rho = 1

    if not os.path.exists(nsc_default_output_dirname):    # para guardar calculos
        os.makedirs(nsc_default_output_dirname)           # crear directorio
        print("Output dir created:\t" + nsc_default_output_dirname)

    print("Navies Stokes Cylinder\nCreating mesh...")
    mesh = get_mesh(mesh_resolution)                           # crear malla
    plot(mesh, title="Created Mesh"); show()                   # graficar malla
    File(nsc_default_output_dirname + meshfilename) << mesh    # Guardar malla

    V = VectorFunctionSpace(mesh, 'P', 2)    # Espacios de funciones
    Q = FunctionSpace(mesh, 'P', 1)

    # hacer calculos para simulación
    print("Solving problem...")
    solve_simulation(V, Q, mesh, T, num_steps, mu, rho, output_dir=\
                     nsc_default_output_dirname)
    
    # leer datos de simulación y graficar
    print("Showing simulations...")
    show_simulation(V, nsc_default_output_dirname + useriesfilename,
                    title="Velocity")
    show_simulation(Q, nsc_default_output_dirname + pseriesfilename,
                    title="Pressure")
    
    return
    


def reaction_system():
    T = 5.0
    num_steps = 500
    eps = 0.01
    K = 10.0

    solve_reaction_simulation(T, num_steps, eps, K, nsc_default_output_dirname,
                              meshfilename, useriesfilename, output_dir=\
                                rs_default_output_dirname)


if __name__ == "__main__":
    # Ecuacion de calor: Condiciones de dirichlet
    A, u0 = dirichletCondition()              # generar sistema
    linear_frames = solveEq(A, u0)            # resolver sistema
    X, T, U = create_mesh(linear_frames)      # formato para graficar
    show_solution(X, T, U, "Heat Equation - Dirichlet Conditions")    # graficar

    
    # Ecuacion de calor: Condiciones de Neumann
    A, u0 = neumannCondition()                # generar sistema
    linear_frames = solveEq(A, u0)            # resolver sistema
    X, T, U = create_mesh(linear_frames)      # formato para graficar
    show_solution(X, T, U, "Heat Equation - Neumann Conditions")    # graficar


    # Ecuacion de calor: Condiciones periodicas
    A, u0 = periodicCondition()               # generar sistema
    linear_frames = solvePeriodicEq(A, u0)    # resolver sistema
    X, T, U = create_mesh(linear_frames)      # formato para graficar 
    show_solution(X, T, U, "Heat Equation - Periodic Conditions")    # graficar

    # Ejercicio 2 (mas info en: Solvin PDEs in Python - The FEniCS Tutorial I)
    ns_cylinder()       # realiza calculos y simulación
    reaction_system()   # solo realiza calculos
