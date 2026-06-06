# parametros ajustables

a, b = 0.0, 1.0      # rango espacial
t0, tf = 0.0, 0.2    # rango temporal

Nx = 200    # numero de puntos espaciales
Nt = 200    # numero de puntos temporales

initial_condition = lambda x: 4*x + 4*x**2





# parametros calculables

dt = (tf - t0) / Nt
dx = (b - a) / Nx