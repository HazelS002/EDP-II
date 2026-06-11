import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

################################################################################
#            PARAMETROS AJUSTABLES PARA LAS GRAFICAS Y ANIMACIONES
################################################################################

# Parametros globales para las graficas
plt.rcParams['figure.constrained_layout.use'] = True    # Ajuste automatico
FIG_SIZE = (10, 4)                                      # Tamaño de las figuras

CALOR_CMAP = 'jet'     # Mapa de colores para la solución de calor
WAVE_CMAP  = 'rainbow' # Mapa de colores para la solución de onda

# Parametros de animacion
INTERVAL = 100   # Intervalo entre frames en milisegundos
REPEAT = True    # Repetir la animación al finalizar

################################################################################


################################################################################
#                 CALCULO DE LAS SOLUCIONES DE LOS EJERCICIOS
################################################################################

# Calculo de terminos de la serie de Fourier:
#     Para cada ejercicio, se define una funcion que calcula los terminos de la
#     serie. La nomenclatura de estas funciones, sigue la siguiente convencion:
#         terms_function_<numero de ejercicio><indice del ejercicio>
#     Ejemplo: Para los terminos de la solución del inciso a del ejercicio 1, la
#     función se llamara terms_function_1a

# Definir los terminos de la serie de Fourier de las soluciones de cada
# ejercicio. Estas funciones deben aceptar tres argumentos: n, X, T, donde n es
# el índice del término, X y T son mallas 2D para x y t respectivamente. La
# función debe retornar un array con shape (len(x), len(t), N) que contenga los
# terminos de la serie para cada n, x y t. (Usar funciones vectorizadas de
# numpy para evitar loops)

def terms_function_1a(n, X, T) -> np.ndarray:
    """ Funcion de terminos de Ejercicio 1a """
    return ((-1)**(n+1) * (4/((2*n-1)*np.pi))**2 *
            np.sin((2*n-1)*np.pi*X/4) *
            np.exp(-((2*n-1)*np.pi)**2 * T / 8))

def terms_function_1b(n, X, T, L=1, alpha=1) -> np.ndarray:
    """ Funcion de terminos de Ejercicio 1b """
    return (-1)**n*16*n*(12*n**2-19)/(np.pi*(4*n**2-25)*(4*n**2-81))\
        *np.sin(n*np.pi*X/L)*np.exp(-(alpha*n*np.pi/L)**2*T)

def terms_function_2b(n, X, T, alpha=1) -> np.ndarray:
    """ Funcion de terminos de Ejercicio 2b """
    return np.sin(n*np.pi*X/np.pi)*((4*(-1)**(n+1)+4)/(n**3*np.pi)*np\
            .cos(n*np.pi*alpha/np.pi*T)+ (-1)**n*2*np.sin(7*np.pi**2)\
                /(np.pi*alpha*(49*np.pi**2-n**2))*np.sin(n*np.pi*alpha/np.pi*T))



# Funcion para calcular la solucion sumando los terminos de la serie de Fourier
def sum_terms(N: int, x: np.ndarray, t: np.ndarray, terms_function: callable)\
    -> np.ndarray:
    """ Suma los terminos de la serie de Fourier para calcular la solución.
    Parámetros:
    - N: numero de terminos de la serie a sumar
    - x: array de puntos en el eje x
    - t: array de puntos en el eje t
    - terms_function: funcion que calcula los terminos de la serie, debe aceptar
      tres argumentos: n, X, T, donde n es el índice del término, X y T son
      mallas 2D para x y t respectivamente. La función debe retornar un array con
      shape (len(x), len(t), N) que contenga los términos de la serie para cada
      n, x y t.
      Retorna: X, T, U donde X y T son las mallas 2D para x y t respectivamente,
      y U es la solución calculada sumando los terminos de la serie
      (shape: (len(x), len(t))).
    """

    n = np.arange(1, N + 1)    # Índices de los términos de la serie
    X_mesh, T_mesh = np.meshgrid(x, t, indexing='ij')  # Crear malla 2D para x y t
    
    # Agregar una nueva dimensión para los términos de la serie
    X = X_mesh[..., np.newaxis]   # shape: (len(x), len(t), 1)
    T = T_mesh[..., np.newaxis]   # shape: (len(x), len(t), 1)
    n = n[np.newaxis, np.newaxis, :]  # shape: (1, 1, N)
    
    # Calcular todos los términos
    term = terms_function(n, X, T)
    
    U = np.sum(term, axis=2)  # Sumar sobre el eje de los terminos
    print_solution_info(X_mesh[:, 0], T_mesh[0, :], U)
    return X_mesh, T_mesh, U


# Funcion auxiliar para imprimir información
def print_solution_info(X, T, sol):
    """ Imprime información relevante sobre la solución calculada, como el
    rango de x y t, y la forma de la solución. """
    print("Info of the solution:")
    print(f"\tRange of x: {X.min()} to {X.max()} with shape {X.shape}")
    print(f"\tRange of t: {T.min()} to {T.max()} with shape {T.shape}")
    print(f"\tShape of the solution: {sol.shape}" + "\n\n" + 80*"=")


if __name__ == "__main__":
    """ Ejecuta los ejercicios de la tarea. Para cada ejercicio, se pueden
    ajustar los parámetros"""

    from visualization import show_solution

    ############################################################################
    #             PARAMETROS AJUSTABLES PARA LOS EJERCICIOS 
    ############################################################################

    N = 10  # Numero de terminos de la serie de Fourier para cada ejercicio

    # Ejercicio 1a (No tiene parametros ajustables)

    # Ejercicio 1b
    param_1b = {
        'L': 1,
        'alpha': 0.1
    }

    # El ejercicio 2a no estan definidas las funciones \phi y
    # \psi por lo cual no lo consideramos para la implementacion.

    # Ejercicio 2b
    alpha = 2

    ############################################################################
    #     EJERCICIOS DE LA TAREA: CALCULO Y VISUALIZACION DE LAS SOLUCIONES
    ############################################################################

    print(80*"=")

    # Ejercicio 1a
    x, t = np.linspace(0,4,250), np.linspace(0,2,250)
    X, T, sol = sum_terms(N, x, t, terms_function_1a)
    title=rf'Ejercicio 1a $(N={N})$'
    show_solution(X, T, sol, title=title, cmap=CALOR_CMAP)

    # Ejercicio 1b
    x, t = np.linspace(0, param_1b['L'], 250), np.linspace(0, 2, 250)
    terms_function = lambda n, X, T: terms_function_1b(n, X, T, L=param_1b['L'],
                                                       alpha=param_1b['alpha'])
    X, T, sol = sum_terms(N, x, t, terms_function)
    title=rf'Ejercicio 1b $(N={N},L={param_1b["L"]},\alpha={param_1b["alpha"]})$'
    show_solution(X, T, sol, title=title, cmap=CALOR_CMAP)
    
    # Ejercicio 2b
    x, t = np.linspace(0, np.pi, 250), np.linspace(0, 6, 250)
    terms_function = lambda n, X, T: terms_function_2b(n, X, T, alpha=alpha)
    X, T, sol = sum_terms(N, x, t, terms_function)
    title=rf'Ejercicio 2b $(N={N},L=3.14,\alpha={alpha})$'
    show_solution(X, T, sol, title=title, cmap=WAVE_CMAP)