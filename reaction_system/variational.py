from fenics import Expression, Constant, TestFunctions
from fenics import split, dx
from fenics import dot, grad


def define_variational_form(V, dt, eps, K, w, u, u_n):
    """Build the variational form F for the system."""
    f_1 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.1,2)<0.05*0.05 ? 0.1 : 0', degree=1)
    f_2 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.3,2)<0.05*0.05 ? 0.1 : 0', degree=1)
    f_3 = Constant(0)

    v_1, v_2, v_3 = TestFunctions(V)
    u_1, u_2, u_3 = split(u)
    u_n1, u_n2, u_n3 = split(u_n)

    k = Constant(dt)
    Kc = Constant(K)
    epsc = Constant(eps)

    F = ((u_1 - u_n1) / k) * v_1 * dx + dot(w, grad(u_1)) * v_1 * dx \
        + epsc * dot(grad(u_1), grad(v_1)) * dx + Kc * u_1 * u_2 * v_1 * dx \
        + ((u_2 - u_n2) / k) * v_2 * dx + dot(w, grad(u_2)) * v_2 * dx \
        + epsc * dot(grad(u_2), grad(v_2)) * dx + Kc * u_1 * u_2 * v_2 * dx \
        + ((u_3 - u_n3) / k) * v_3 * dx + dot(w, grad(u_3)) * v_3 * dx \
        + epsc * dot(grad(u_3), grad(v_3)) * dx - Kc * u_1 * u_2 * v_3 * dx + Kc * u_3 * v_3 * dx \
        - f_1 * v_1 * dx - f_2 * v_2 * dx - f_3 * v_3 * dx
        
    return F