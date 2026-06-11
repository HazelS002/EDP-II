from dolfin import Point
from mshr import Rectangle, Circle


channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)


mesh_resolution = 20    # 64
save_every = 10


# file names
nsc_default_output_dirname = "nsc_datasimulation/"

ufilename = "velocity.xdmf"
pfilename = "pressure.xdmf"
meshfilename = "cylinder.xml.gz"    

useriesfilename = "velocity_series"
pseriesfilename = "pressure_series"