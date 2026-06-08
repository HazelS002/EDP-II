import numpy as np

from ..config import Nx, dx
from ..utils.helpers import get_u0

def neumannCondition():
    A = np.zeros((Nx, Nx))
    u0 = get_u0()

    # Interior
    for i in range(1, Nx-1):
        A[i, i-1] = 1/dx**2
        A[i, i]   = -2/dx**2
        A[i, i+1] = 1/dx**2

    A[0, 0] = -2/dx**2; A[0, 1] =  2/dx**2         # Borde izquierdo (u_x=0)
    A[-1, -2] =  2/dx**2;  A[-1, -1] = -2/dx**2    # Borde derecho (u_x=0)

    return A, u0