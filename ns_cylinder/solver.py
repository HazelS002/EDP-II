from __future__ import print_function

from dolfin import assemble, solve, XDMFFile, TimeSeries

from .boundary import get_boundary_conditions
from .variational import assemble_forms
from .config import ufilename, pfilename, useriesfilename, pseriesfilename,\
    nsc_default_output_dirname, save_every


def solve_simulation(V, Q, mesh, T, num_steps, mu, rho,
                     output_dir = nsc_default_output_dirname):
    dt = T / num_steps    # tamaño de paso

    # definir las condiciones de frontera para velocidad y presion
    bcu, bcp = get_boundary_conditions(V, Q)

    # Obtener formas débiles
    (a1, L1), (a2, L2), (a3, L3), _, _, (u_n, u_), (p_n, p_) =\
        assemble_forms(V, Q, dt, mu, rho, mesh)

    # Ensamblaje de matrices (independientes del tiempo)
    A1, A2, A3 = assemble(a1), assemble(a2), assemble(a3)

    # Aplicar condiciones de contorno a matrices
    [ bc.apply(A1) for bc in bcu ]; [ bc.apply(A2) for bc in bcp]

    # Archivos de salida XDMF
    xdmffile_u = XDMFFile(output_dir + ufilename)
    xdmffile_p = XDMFFile(output_dir + pfilename)
    xdmffile_u.parameters["flush_output"] = True
    xdmffile_p.parameters["flush_output"] = True

    # Crear TimeSeries (para usar en reaction_system)
    timeseries_u = TimeSeries(output_dir + useriesfilename)
    timeseries_p = TimeSeries(output_dir + pseriesfilename)

    # Bucle temporal
    t = 0
    for n in range(num_steps):
        t += dt

        # volver a aplicar pasos para cada instante
        # Paso 1
        b1 = assemble(L1)
        [ bc.apply(b1) for bc in bcu ]
        solve(A1, u_.vector(), b1, 'bicgstab', 'default')

        # Paso 2
        b2 = assemble(L2)
        [ bc.apply(b2) for bc in bcp ]
        solve(A2, p_.vector(), b2, 'bicgstab', 'default')

        # Paso 3
        b3 = assemble(L3)
        solve(A3, u_.vector(), b3, 'cg', 'sor')

        # Guardar cada ciertos pasos
        if n % save_every == 0 or n == num_steps-1:
            xdmffile_u.write(u_, t)
            xdmffile_p.write(p_, t)

            # Save nodal values to file
            timeseries_u.store(u_.vector(), t)
            timeseries_p.store(p_.vector(), t)

        u_n.assign(u_); p_n.assign(p_)    # Actualizar soluciones anteriores

        # Mostrar progreso y velocidad máxima cada 100 pasos
        print(f"Step {n}/{num_steps}, t={t:.3f}s, |u|_max = {u_.vector().norm('linf'):.4f}")

    print(f"Data simulation saved in: {output_dir}")
