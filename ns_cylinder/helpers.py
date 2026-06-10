from dolfin import sym, nabla_grad, Identity

# Operadores de deformación y tensión
def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p, mu_c):
    return 2*mu_c*epsilon(u) - p*Identity(len(u))

__all__ = [
    "epsilon", "sigma"
]