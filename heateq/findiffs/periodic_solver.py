import numpy as np
from scipy.linalg import solve

from ..utils.helpers import get_mats
from ..config import Nt, Nx

def solvePeriodicEq(A, u0, theta=0.5):
    M1, M2 = get_mats(A, theta=theta)
    linear_frames = []

    linear_frames.append(u0.copy())

    u = u0[:Nx - 1].copy()
    for _ in range(Nt):
        u = solve(M1, M2 @ u)        
        linear_frames.append(np.append(u, u[0]))

    return linear_frames
