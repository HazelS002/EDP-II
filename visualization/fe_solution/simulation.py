import matplotlib.pyplot as plt
from dolfin import plot

from .utils.helpers import get_timeserie
from .config import FIG_KWARGS, pause

def show_simulation(space, dirname, title=""):
    times, us = get_timeserie(space, dirname)
    fig, ax = plt.subplots(figsize=FIG_KWARGS["figsize"])

    colorbar = None
    for t, u in zip(times, us):
        ax.clear()

        plot_title = title + rf" - $t={t:.3f}$" if title else rf"$t={t:.3f}$"
        artist = plot(u, title=plot_title, axes=ax)

        if colorbar is None:
            colorbar = fig.colorbar(artist, ax=ax)
        else:
            colorbar.update_normal(artist)

        plt.draw()
        plt.pause(pause)

    plt.show()