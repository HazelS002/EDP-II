import numpy as np

from ..config import Nx, dx 
from ..utils.helpers import get_u0

def periodicCondition():
    # Usamos Nx-1 incógnitas (u0,...,u_{Nx-2}), con u_{Nx-1}=u0
    N = Nx - 1
    A = np.zeros((N, N))

    u0 = get_u0()
    avg = (u0[0] + u0[-1])/2
    u0[0] = avg; u0[-1] = avg

    # Interior (i=1,...,N-2)
    for i in range(1, N-1):
        A[i, i-1] = 1/dx**2
        A[i, i]   = -2/dx**2
        A[i, i+1] = 1/dx**2
    # i=0
    A[0, 0]   = -2/dx**2
    A[0, 1]   =  1/dx**2
    A[0, N-1] = 1/dx**2   # conecta con el último nodo independiente (índice N-1 = Nx-2)
    # Para i=N-1 (último nodo independiente, i = Nx-2):
    # (u_{Nx-3} - 2u_{Nx-2} + u_{Nx-1})/dx^2, con u_{Nx-1}=u0.
    # Entonces A[N-1, N-2] = 1/dx^2, A[N-1, N-1] = -2/dx^2, A[N-1, 0] = 1/dx^2.
    A[N-1, N-2] = 1/dx**2
    A[N-1, N-1] = -2/dx**2
    A[N-1, 0]   = 1/dx**2
    
    return A, u0