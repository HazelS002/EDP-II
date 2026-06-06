import numpy as np

from ..config import Nx, dx
from ..utils.helpers import get_u0

def dirichletCondition():
    A = np.zeros((Nx, Nx))
    
    u0 = get_u0()
    u0[0] = 0.0; u0[-1] = 0.0

    for i in range(1, Nx-1):        # Interiores
        A[i, i-1] = 1/dx**2
        A[i, i]   = -2/dx**2
        A[i, i+1] = 1/dx**2
    
    A[0, 0] = 1.0; A[-1, -1] = 1.0    # Bordes: u=0
    return A, u0