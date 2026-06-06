import numpy as np

from ..config import a, b, Nx
# from ..config import t0, tf, Nt
from ..config import dt
from ..config import initial_condition

# def create_mesh():
#     x, t = np.linspace(a, b, Nx), np.linspace(t0, tf, Nt)
#     X_mesh, T_mesh = np.meshgrid(x, t, indexing='ij')

#     return (X_mesh, T_mesh), (x, t)

def get_mats(A, theta):
    I = np.eye(A.shape[0])
    M1, M2 = I - theta*dt*A, I + (1 - theta)*dt*A
    return M1, M2

def get_u0():
    return initial_condition(np.linspace(a, b, Nx))