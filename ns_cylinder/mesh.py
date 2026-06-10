from mshr import generate_mesh

from .config import channel, cylinder


def get_mesh(resolution):
    domain = channel - cylinder
    mesh = generate_mesh(domain, resolution)
    return mesh