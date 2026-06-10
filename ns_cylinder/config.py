from dolfin import Point
from mshr import Rectangle, Circle


channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)


mesh_resolution = 20    # 64
save_every = 10


# file names

ufilename = "velocity.xdmf"
pfilename = "pressure.xdmf"
meshfilename = "cylinder.xml.gz"    
default_output_dirname = "navier_stokes_cylinder"