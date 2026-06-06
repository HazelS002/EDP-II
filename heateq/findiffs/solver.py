from scipy.linalg import solve

from ..utils.helpers import get_mats
from ..config import Nt

def solveEq(A, u0, theta=0.5):
    M1, M2 = get_mats(A, theta=theta)
    linear_frames = []

    u = u0.copy(); linear_frames.append(u)

    for _ in range(Nt):
        u = solve(M1, M2 @ u)
        linear_frames.append(u)
        
    return linear_frames
