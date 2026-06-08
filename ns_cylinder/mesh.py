import gmsh
from mpi4py import MPI
import dolfinx.io.gmshio

def create_mesh(lx=2.2, ly=0.41, circle_center=(0.2, 0.2), circle_radius=0.05, lc=0.02):
    """Genera la malla 2D de un canal rectangular con un cilindro circular.
    
    Returns:
        domain (dolfinx.mesh.Mesh): La malla.
        facet_tags (dolfinx.mesh.MeshTags): Marcadores de las fronteras.
    """
    gmsh.initialize()
    gmsh.model.add("cylinder_channel")

    # Puntos del rectángulo
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(lx, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(lx, ly, 0, lc, 3)
    gmsh.model.geo.addPoint(0, ly, 0, lc, 4)

    # Líneas del rectángulo
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 5)

    # Círculo
    cx, cy = circle_center
    r = circle_radius
    gmsh.model.geo.addPoint(cx, cy, 0, lc/2, 10)
    gmsh.model.geo.addPoint(cx + r, cy, 0, lc/2, 11)
    gmsh.model.geo.addPoint(cx, cy + r, 0, lc/2, 12)
    gmsh.model.geo.addPoint(cx - r, cy, 0, lc/2, 13)
    gmsh.model.geo.addPoint(cx, cy - r, 0, lc/2, 14)

    gmsh.model.geo.addCircleArc(11, 10, 12, 10)
    gmsh.model.geo.addCircleArc(12, 10, 13, 11)
    gmsh.model.geo.addCircleArc(13, 10, 14, 12)
    gmsh.model.geo.addCircleArc(14, 10, 11, 13)

    gmsh.model.geo.addCurveLoop([10, 11, 12, 13], 15)

    # Superficie
    gmsh.model.geo.addPlaneSurface([5, 15], 1)

    # Grupos físicos (para identificar fronteras)
    gmsh.model.addPhysicalGroup(1, [1], 1)      # inflow (x=0)
    gmsh.model.addPhysicalGroup(1, [2], 2)      # outflow (x=lx)
    gmsh.model.addPhysicalGroup(1, [3, 4], 3)   # walls (y=0, y=ly)
    gmsh.model.addPhysicalGroup(1, [10, 11, 12, 13], 4)  # cilindro
    gmsh.model.addPhysicalGroup(2, [1], 5)      # dominio

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )

    gmsh.finalize()
    return domain, facet_tags