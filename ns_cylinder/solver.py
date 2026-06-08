import os
from dolfinx import fem, io, la
from dolfinx.fem.petsc import LinearProblem

def save_xdmf(file, func, t, name):
    """Escribe una función en un archivo XDMF con el tiempo dado."""
    func.name = name
    file.write_function(func, t)

def time_loop(domain, V, Q, bcu, bcp, forms, initial_functions,
              dt, num_steps, output_dir="results"):
    """Ejecuta el bucle temporal del esquema de proyección.

    Args:
        domain: Malla.
        V, Q: Espacios de funciones.
        bcu, bcp: Condiciones de contorno.
        forms: Tupla (a1, L1, a2, L2, a3, L3).
        initial_functions: Tupla (u_n, p_n, u_, p_).
        dt: Paso de tiempo.
        num_steps: Número de pasos.
        output_dir: Directorio para guardar resultados.
    """
    a1, L1, a2, L2, a3, L3 = forms
    u_n, p_n, u_, p_ = initial_functions

    # Preparar archivos XDMF
    os.makedirs(output_dir, exist_ok=True)
    xdmf_u = io.XDMFFile(domain.comm, os.path.join(output_dir, "velocity.xdmf"), "w")
    xdmf_p = io.XDMFFile(domain.comm, os.path.join(output_dir, "pressure.xdmf"), "w")
    xdmf_u.write_mesh(domain)
    xdmf_p.write_mesh(domain)

    # Crear solvers lineales (reutilizables)
    problem1 = LinearProblem(a1, L1, bcs=bcu, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem2 = LinearProblem(a2, L2, bcs=bcp, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem3 = LinearProblem(a3, L3, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "sor"})

    t = 0.0
    for step in range(num_steps):
        t += dt
        # Actualizar constantes (dt puede cambiar, pero es fijo aquí)
        # En DOLFINx, si dt no cambia, no es necesario recompilar.

        # Paso 1
        u_.vector[:] = problem1.solve().vector
        # Paso 2
        p_.vector[:] = problem2.solve().vector
        # Paso 3
        u_.vector[:] = problem3.solve().vector

        # Guardar cada 50 pasos
        if step % 50 == 0 or step == num_steps - 1:
            save_xdmf(xdmf_u, u_, t, "velocity")
            save_xdmf(xdmf_p, p_, t, "pressure")

        # Actualizar funciones para el siguiente paso
        u_n.vector[:] = u_.vector
        p_n.vector[:] = p_.vector

        if step % 100 == 0:
            max_vel = u_.vector.max()
            print(f"Paso {step}/{num_steps}, t = {t:.3f} s, |u|_max = {max_vel:.4f}")

    xdmf_u.close()
    xdmf_p.close()
    print("Simulación completada.")