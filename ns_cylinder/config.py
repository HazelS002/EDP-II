from dolfin import Point
from mshr import Rectangle, Circle


channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)


mesh_resolution = 20    # 64
save_every = 10