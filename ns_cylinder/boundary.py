import numpy as np
from dolfinx import fem, default_scalar_type

def inflow_velocity_expression(x):
    """Perfil de velocidad parabólico en la entrada."""
    u_max = 1.5
    return np.array([
        4.0 * u_max * x[1] * (0.41 - x[1]) / (0.41**2),
        np.zeros_like(x[0])
    ])

def define_boundary_conditions(domain, V, Q, facet_tags):
    """Define las condiciones de Dirichlet para velocidad y presión.

    Args:
        domain (Mesh): La malla.
        V (FunctionSpace): Espacio para velocidad (vectorial).
        Q (FunctionSpace): Espacio para presión (escalar).
        facet_tags (MeshTags): Etiquetas de las fronteras (1:inflow, 2:outflow, 3:walls, 4:cylinder).

    Returns:
        bcu (list): Lista de condiciones de Dirichlet para velocidad.
        bcp (list): Lista de condiciones de Dirichlet para presión.
    """
    # Velocidad de entrada
    inflow_vel = fem.Function(V)
    inflow_vel.interpolate(inflow_velocity_expression)

    # Localización de DOFs en las fronteras
    inflow_dofs = fem.locate_dofs_topological(V, 1, facet_tags.find(1))
    walls_dofs = fem.locate_dofs_topological(V, 1, facet_tags.find(3))
    cylinder_dofs = fem.locate_dofs_topological(V, 1, facet_tags.find(4))
    outflow_dofs_p = fem.locate_dofs_topological(Q, 1, facet_tags.find(2))

    zero_velocity = fem.Constant(domain, default_scalar_type((0.0, 0.0)))
    zero_pressure = fem.Constant(domain, default_scalar_type(0.0))

    bcu = [
        fem.dirichletbc(inflow_vel, inflow_dofs),
        fem.dirichletbc(zero_velocity, walls_dofs),
        fem.dirichletbc(zero_velocity, cylinder_dofs)
    ]
    bcp = [fem.dirichletbc(zero_pressure, outflow_dofs_p)]

    return bcu, bcp