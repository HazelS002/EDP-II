from __future__ import print_function
from fenics import File, TimeSeries, solve
import os

from .helpers import read_mesh, setup_spaces, setup_functions, save_solution
from .variational import define_variational_form
from .config import rs_default_output_dirname, u1filename, u2filename, u3filename

def solve_reaction_simulation(T, num_steps, eps, K, input_dir, meshfilename,
                              useriesfilename, output_dir\
                                =rs_default_output_dirname):
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    mesh = read_mesh(input_dir + meshfilename)
    dt = T / num_steps

    # Setup spaces and functions
    W, V = setup_spaces(mesh)
    w, u, u_n = setup_functions(V, W)

    # Define source terms and variational form
    F = define_variational_form(V, dt, eps, K, w, u, u_n)

    # Create output files
    vtkfile_u_1 = File(output_dir + u1filename)
    vtkfile_u_2 = File(output_dir + u2filename)
    vtkfile_u_3 = File(output_dir + u3filename)

    vtkfiles = [vtkfile_u_1, vtkfile_u_2, vtkfile_u_3]

    # Time series for velocity data
    timeseries_w = TimeSeries(input_dir + useriesfilename)

    # Time-stepping
    t = 0.0
    for _ in range(num_steps):
        t += dt
        timeseries_w.retrieve(w.vector(), t)   # read velocity at current time
        solve(F == 0, u)
        save_solution(u, t, vtkfiles)          # save to VTK
        u_n.assign(u)