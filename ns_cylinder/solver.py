from __future__ import print_function
import os

from dolfin import VectorFunctionSpace, FunctionSpace
from dolfin import assemble, solve
from dolfin import XDMFFile, File

from .boundary import get_boundary_conditions
from .variational import assemble_forms
from .config import mesh_resolution, save_every
from .mesh import get_mesh


def solve_simulation(T, num_steps, mu, rho, output_dir = "navier_stokes_cylinder"):
    # Crear directorio para resultados
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dt = T / num_steps
    mesh = get_mesh(mesh_resolution)


    # Espacios de funciones
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)

    bcu, bcp = get_boundary_conditions(V, Q)

    # Obtener formas débiles
    (a1, L1), (a2, L2), (a3, L3), (u, v), (p, q), (u_n, u_), (p_n, p_) =\
        assemble_forms(V, Q, dt, mu, rho, mesh)

    # Ensamblaje de matrices (independientes del tiempo)
    A1, A2, A3 = assemble(a1), assemble(a2), assemble(a3)

    # Aplicar condiciones de contorno a matrices
    [ bc.apply(A1) for bc in bcu ]; [ bc.apply(A2) for bc in bcp]

    # Archivos de salida XDMF
    xdmffile_u = XDMFFile(os.path.join(output_dir, "velocity.xdmf"))
    xdmffile_p = XDMFFile(os.path.join(output_dir, "pressure.xdmf"))
    xdmffile_u.parameters["flush_output"] = True
    xdmffile_p.parameters["flush_output"] = True

    # Guardar malla
    File(os.path.join(output_dir, "cylinder.xml.gz")) << mesh

    # Bucle temporal
    t = 0
    for n in range(num_steps):
        t += dt

        # Paso 1
        b1 = assemble(L1)
        [ bc.apply(b1) for bc in bcu ]
        solve(A1, u_.vector(), b1, 'bicgstab', 'default')  # 'hypre_amg' si está disponible

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

        # Actualizar soluciones anteriores
        u_n.assign(u_); p_n.assign(p_)

        # Mostrar progreso y velocidad máxima cada 100 pasos
        print(f"Step {n}/{num_steps}, t = {t:.3f} s, |u|_max = {u_.vector().norm('linf'):.4f}")

    print(f"Data simulation saved in: {output_dir}")
