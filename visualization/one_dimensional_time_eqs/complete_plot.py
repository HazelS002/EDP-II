from ..utils.builders import create_2d_plot, create_3d_plot, create_animation
from matplotlib import pyplot as plt

FIG_SIZE = (10, 4)

def show_solution(X, T, U, title='Solucion', cmap='viridis'):
    """  Muestra la solución calculada en tres formas: un gráfico 2D con mapa
    de colores, un gráfico 3D de superficie, y una animación de la solución
    como una barra de colores horizontal.
     Parámetros:
    - X, T: mallas 2D para x y t respectivamente (shape: (len(x), len(t)))
    - U: solución calculada (shape: (len(x), len(t)))
    - title: título general para la figura
    - cmap: mapa de colores a usar para las gráficas
    Retorna: im, surf, anim, cbar donde im es el objeto de la gráfica 2D, surf
    es el objeto de la gráfica 3D, anim es el objeto de la animación, y cbar
    es la colorbar. """
    
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1])   # 1 fila, 3 columnas
    
    # Grafica 2D
    ax1 = fig.add_subplot(gs[0, 0])
    im = create_2d_plot(X, T, U, ax1, cmap=cmap)
    
    # Grafica 3D
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    surf = create_3d_plot(X, T, U, ax2, cmap=cmap)
    
    # Animacion 
    ax3 = fig.add_subplot(gs[0, 2])
    anim = create_animation(X, T, U, ax3, cmap=cmap)

    # Crea la colorbar horizontal compartida por los tres ejes
    cbar = fig.colorbar(im, ax=[ax1, ax2, ax3], orientation='horizontal',
                        fraction=0.05, pad=0.05, label=r'$u(x,t)$')
    
    fig.suptitle(title)  # Título general para la figura
    plt.show()

    return im, surf, anim, cbar