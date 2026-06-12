from dolfin import SubDomain, near, DOLFIN_EPS
from dolfin import DirichletBC, Constant, Expression

# Definición de fronteras (más precisa para el cilindro)
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 2.2)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0) or near(x[1], 0.41))

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ( (x[0]-0.2)**2 + (x[1]-0.2)**2 < 0.05**2 + DOLFIN_EPS)

def get_boundary_conditions(V, Q):
    # Inicializar marcadores
    inflow = Inflow()
    outflow = Outflow()
    walls = Walls()
    cylinder = Cylinder()

    # velocidad de entrada
    inflow_profile =\
        Expression(('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0'), degree=2)

    # Condiciones de contorno
    bcu_inflow = DirichletBC(V, inflow_profile, inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)

    bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
    bcp = [bcp_outflow]

    return bcu, bcp


__all__ = [
    "Inflow", "Outflow", "Walls", "Cylinder", "get_boundary_conditions"
]