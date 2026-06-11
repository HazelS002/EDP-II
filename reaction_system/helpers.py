from fenics import Mesh, triangle
from fenics import VectorFunctionSpace, FunctionSpace, Function,\
    FiniteElement, MixedElement


def read_mesh(dir_mesh):
    return Mesh(dir_mesh)


def setup_spaces(mesh):
    """Define function spaces for velocity and concentrations."""
    W = VectorFunctionSpace(mesh, 'P', 2)
    P1 = FiniteElement('P', triangle, 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(mesh, element)
    return W, V


def setup_functions(V, W):
    """Create functions for velocity and concentrations."""
    w = Function(W)
    u = Function(V)
    u_n = Function(V)
    return w, u, u_n


def save_solution(u, t, vtkfiles):
    """Split the mixed solution and save each component to VTK."""
    vtkfile_u_1, vtkfile_u_2, vtkfile_u_3 = vtkfiles
    u_1, u_2, u_3 = u.split()
    vtkfile_u_1 << (u_1, t)
    vtkfile_u_2 << (u_2, t)
    vtkfile_u_3 << (u_3, t)