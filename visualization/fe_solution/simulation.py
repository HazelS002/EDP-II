from dolfin import plot
from matplotlib import pyplot as plt

from .utils.helpers import get_timeserie
from .config import FIG_KWARGS


def show_simulation(space, dirname):    
    times, us = get_timeserie(space, dirname)

    fig, ax = plt.subplots(1, 2, figsize=FIG_KWARGS["figsize"])

    for t, u in zip(times, us):
        artist = plot(u, title=rf"$t={t:.3f}$")
        plt.colorbar(artist)

        plt.draw()
        plt.pause(0.002)
        plt.clf()