from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from ..config import REPEAT


def create_2d_plot(X, T, U, ax, cmap):
    """  Gráfico 2D con mapa de colores. El eje x representa x, el eje y
    representa t, y el color representa u(x,t).
    Parámetros:
    - X, T: mallas 2D para x y t respectivamente (shape: (len(x), len(t)))
    - U: solución calculada (shape: (len(x), len(t)))
    - ax: objeto de eje de matplotlib donde se dibujará la gráfica
    - cmap: mapa de colores a usar para la gráfica
    Retorna: im, el objeto de la gráfica 2D. """
    im = ax.pcolormesh(X, T, U, shading='auto', cmap=cmap)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$t$')
    return im

def create_3d_plot(X, T, U, ax, cmap):
    """ Gráfico 3D de superficie. El eje x representa x, el eje y representa t,
    y el eje z representa u(x,t).
    Parámetros:
    - X, T: mallas 2D para x y t respectivamente (shape: (len(x), len(t)))
    - U: solución calculada (shape: (len(x), len(t)))
    - ax: objeto de eje de matplotlib donde se dibujará la gráfica
    - cmap: mapa de colores a usar para la gráfica
    Retorna: surf, el objeto de la gráfica 3D. """
    surf = ax.plot_surface(X, T, U, cmap=cmap, linewidth=0, antialiased=True)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$t$')
    return surf

def create_animation(X, T, U, ax, cmap, interval=100):
    x, t = X[:, 0], T[0, :]

    # limpiar ejes
    for place in ['top', 'right', 'left']: ax.spines[place].set_visible(False)
    ax.set_xlabel(r'$x$'); ax.set_yticks([])

    # imagen inicial
    img_data = U[:, 0].reshape(1, -1)
    im = ax.imshow(img_data, extent=[x.min(), x.max(), -0.05, 0.05],
                   aspect='auto', origin='lower', cmap=cmap)
    ax.set_ylim(-1, 1)
    time_text = ax.text(0.35, 0.7, '', transform=ax.transAxes, fontsize=15)

    def init():
        im.set_array(U[:, 0].reshape(1, -1))
        time_text.set_text(rf'$t={t[0]:.4f}$')
        return [im, time_text]

    def update(frame):
        idx = frame % len(t)
        im.set_array(U[:, idx].reshape(1, -1))
        time_text.set_text(rf'$t={t[idx]:.4f}$')
        return [im, time_text]

    anim = FuncAnimation(ax.figure, update, frames=len(t), init_func=init,
        interval=interval, repeat=REPEAT, blit=True)

    return anim


if __name__ == "__main__": pass