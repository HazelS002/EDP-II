import numpy as np
from matplotlib import pyplot as plt


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


# Funcion para calcular la solución sumando los términos de la serie de Fourier

def sum_terms(N: int, x: np.ndarray, t: np.ndarray, terms_function) -> np.ndarray:
    """ Calcula la solución sumando los términos de la serie de Fourier. La
    función terms_function debe aceptar tres argumentos: n, X, T, donde n es el
    índice del término, X y T son mallas 2D para x y t respectivamente. La
    función debe retornar un array con shape (len(x), len(t), N) que contenga
    los términos de la serie para cada n, x y t."""

    n = np.arange(1, N + 1)    # Índices de los términos de la serie
    X, T = np.meshgrid(x, t, indexing='ij')  # Crear malla 2D para x y t
    
    # Agregar una nueva dimensión para los términos de la serie
    X = X[..., np.newaxis]   # shape: (len(x), len(t), 1)
    T = T[..., np.newaxis]   # shape: (len(x), len(t), 1)
    n = n[np.newaxis, np.newaxis, :]  # shape: (1, 1, N)
    
    # Calcular todos los términos
    term = terms_function(n, X, T)
    
    U = np.sum(term, axis=2)  # Sumar sobre el eje de los terminos
    print_solution_info(X, T, U)
    return X, T, U


# Funcion auxiliar para imprimir información
def print_solution_info(X, T, sol):
    """Imprime información sobre los rangos y shapes de X, T y la solución."""
    print(f"Range of x: {X.min()} to {X.max()} with shape {X.shape}")
    print(f"Range of t: {T.min()} to {T.max()} with shape {T.shape}")
    print(f"Shape of the solution: {sol.shape}")

# Función para mostrar la solución usando un mapa de colores
def show_solution(X, T, sol, title='Solution'):
    """ Muestra la solución usando un mapa de colores. X y T deben ser mallas
    2D con shape (len(x), len(t))."""
    plt.imshow(sol.T, extent=(X.min(), X.max(), T.min(), T.max()),
               aspect='auto', origin='lower')
    plt.colorbar(label=r'$u(x,t)$')
    plt.xlabel(r'$x$') ; plt.ylabel(r'$t$')
    plt.title(title)
    plt.show()



if __name__ == "__main__":
    """ Ejecuta los ejercicios de la tarea. Para cada ejercicio, se pueden
    ajustar los parámetros"""

    # Ejercicio 1a
    x, t = np.linspace(0,4,250), np.linspace(0,2,250)
    N = 10

    X, T, sol = sum_terms(N, x, t, terms_function_1a)
    show_solution(X, T, sol, title=f'Solucion de Ejercicio 1a con {N} términos')



    # Ejercicio 1b
    L, alpha = 1, 0.1
    x, t = np.linspace(0, L, 250), np.linspace(0, 2, 250)
    N = 10

    X, T, sol = sum_terms(N, x, t, lambda n, X, T:\
                          terms_function_1b(n, X, T, L=L, alpha=alpha))
    show_solution(X, T, sol, title=f'Solucion de Ejercicio 1b con {N} terminos, L={L}, alpha={alpha}')