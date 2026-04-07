import numpy as np
from matplotlib import pyplot as plt


# Calculo de terminos de la serie de Fourier
#     Para la nomenclatura de las funciones, se sigue la siguiente convención:
#     terms_<numero de ejercicio><indice del ejercicio>
#     Ejemplo: para el inciso a del ejercicio 1, la función se llamará terms_1a


def sol_1a(N: int, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calcula la solución del Ejercicio 1a usando N términos de la serie de Fourier."""
    n = np.arange(1, N + 1)    # Índices de los términos de la serie
    X, T = np.meshgrid(x, t, indexing='ij')  # Crear malla 2D para x y t
    
    # Agregar una nueva dimensión para los términos de la serie
    X = X[..., np.newaxis]   # shape: (len(x), len(t), 1)
    T = T[..., np.newaxis]   # shape: (len(x), len(t), 1)
    n = n[np.newaxis, np.newaxis, :]  # shape: (1, 1, N)
    
    # Calcular todos los términos
    term = ((-1)**(n+1) * (4/((2*n-1)*np.pi))**2 *
            np.sin((2*n-1)*np.pi*X/4) *
            np.exp(-((2*n-1)*np.pi)**2 * T / 8))
    
    U = np.sum(term, axis=2)  # Sumar sobre el eje de los terminos
    print_solution_info(X, T, U)
    return X, T, U


# Funciones auxiliares para 
def print_solution_info(X, T, sol):
    """Imprime información sobre los rangos y shapes de X, T y la solución."""
    print(f"Range of x: {X.min()} to {X.max()} with shape {X.shape}")
    print(f"Range of t: {T.min()} to {T.max()} with shape {T.shape}")
    print(f"Shape of the solution: {sol.shape}")

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
    x = np.linspace(0,4,250)
    t = np.linspace(0,2,250)
    N = 10

    X, T, sol = sol_1a(N,x,t)
    show_solution(X, T, sol, title=f'Solucion de Ejercicio 1a con {N} términos')